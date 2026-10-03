#!/usr/bin/env bash
#
# Orchestrates the GPS fusion evaluation runs against a ROS2 bag.
#
#   ./run_eval.sh                 # run every variant
#   ./run_eval.sh B_gps           # run one variant
#   ./run_eval.sh gt              # (re)generate ground truth only
#   REPEATS=3 ./run_eval.sh B_gps # run one variant 3x -> B_gps_run1, _run2, _run3
#
# REPEATS exists because these runs are NOT reproducible at the level that matters. Every run
# processes an identical amount of data (same pose and frame counts), but threading in the GPS drain
# path means delayed init can land on different geometry, and transform quality varies run to run.
# A single run therefore cannot distinguish "this setting is better" from "this run got lucky".
#
# Each run gets its own output directory holding the exact config it ran with, the recorded
# trajectory, and the estimator log. Nothing is shared between runs except the ground-truth file, so
# results stay reproducible and self-documenting.
#
# Required environment (or set the variable of the same name directly):
#   BAG          ROS2 bag directory containing IMU, camera(s) and a NavSatFix topic
#   BASE_CONFIG  estimator config directory for that dataset (holds estimator_config.yaml)
#   WS           colcon workspace with ov_msckf built (needs $WS/install/setup.bash)
#   OUT_ROOT     where results are written
# Optional: GRACE_SECS, REPEATS, RUN_TAG, GPS_TOPIC, POSE_TOPIC, PLAYBACK_RATE,
#           DROPOUT_START, DROPOUT_LENGTHS (see below).
# AI Generated
set -euo pipefail

BAG="${BAG:-${DATASET_DIR:-}}"
WS="${WS:-${WORKSPACE:-}}"
BASE_CONFIG="${BASE_CONFIG:-}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_PATH:-}}"
# NavSatFix topic carrying the GNSS track used as ground truth (passed to make_gt.py).
GPS_TOPIC="${GPS_TOPIC:-/imu/gnss}"
# Topic the estimator publishes its pose on, which the recorder subscribes to. /poseimu is what
# `ros2 run` gives (no namespace); see record_traj.py.
POSE_TOPIC="${POSE_TOPIC:-/poseimu}"
# Keep at 1.0 unless you know your QoS tolerates faster playback (see README).
PLAYBACK_RATE="${PLAYBACK_RATE:-1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Seconds to keep the estimator alive after playback ends, so it can finish anything still queued.
GRACE_SECS="${GRACE_SECS:-8}"
# How many times to repeat each requested variant (see the note at the top on why this matters).
REPEATS="${REPEATS:-1}"
# Optional label inserted into the output directory name, e.g. RUN_TAG=rigorous -> D_dropout60_rigorous_run1.
# Use it when re-running a variant under a changed config so the previous results are not overwritten
# and both sets can be analyzed side by side.
RUN_TAG="${RUN_TAG:-}"

# Variant table: NAME | gps_enabled | dropout_start | dropout_end
# Dropout times are seconds relative to the FIRST GPS fix (see VioManager::feed_measurement_gps).
# Pick DROPOUT_START and DROPOUT_LENGTHS to suit the bag: the outage should begin after the filter has
# converged and end with enough bag left to observe re-convergence. Unaided distance is roughly
# platform speed x outage length. Defaults suit a bag a few minutes long; set DROPOUT_LENGTHS="" to
# run only the A/B variants.
DROPOUT_START="${DROPOUT_START:-120}"
DROPOUT_LENGTHS="${DROPOUT_LENGTHS-30 60 90}"
VARIANTS=(
  "A_vio_only|false|-1|-1"
  "B_gps|true|-1|-1"
)
# Dropout variants are lettered C, D, E, ... in the order given.
letters=(C D E F G H I J K L)
i=0
for len in $DROPOUT_LENGTHS; do
  VARIANTS+=("${letters[$i]}_dropout${len}|true|${DROPOUT_START}|$((DROPOUT_START + len))")
  i=$((i + 1))
done

log() { printf '\033[1;36m[eval]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[eval] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ESTIMATOR_PAT="run_subscribe_msckf"
RECORDER_PAT="gps_eval/record_traj.py"

# PIDs of real estimator/recorder processes matching a pattern.
#
# Plain `pgrep -f` is not usable here: it also matches any *shell* whose command line happens to
# mention the name -- including this script and the terminal that launched it -- which would make the
# preflight check refuse to run against itself. Filtering by comm keeps only actual processes.
# `pgrep -x` is not an option either: comm is truncated to 15 chars, so "run_subscribe_msckf" never
# matches exactly.
proc_pids() {
  local pat="$1" p comm
  pgrep -f "$pat" 2>/dev/null | while read -r p; do
    comm="$(ps -p "$p" -o comm= 2>/dev/null || true)"
    case "$comm" in
      bash | sh | zsh | dash | pgrep | pkill | ps | grep | "") ;;
      *) echo "$p" ;;
    esac
  done
  # MUST return 0. "No matches" is the normal, healthy case, but pgrep signals it with exit status 1,
  # and `set -o pipefail` propagates that out of the pipeline. A caller doing `pids="$(proc_pids ...)"`
  # then inherits status 1 from the command substitution, which `set -e` turns into a silent exit --
  # i.e. the script would abort exactly when the system is clean.
  return 0
}

any_running() { [ -n "$(proc_pids "$1")" ]; }

# Shut the estimator down and PROVE it is gone, escalating INT -> TERM -> KILL.
#
# Two properties of this stack make the naive approach silently wrong, and the failure mode is data
# corruption rather than a crash:
#
#   1. run_subscribe_msckf does not exit on SIGINT. It finishes its work, prints "TIME: ... seconds",
#      then wedges in teardown (the class_loader "attempting to unload library while objects created by
#      this loader exist" warnings) with its worker threads parked. A bare `wait` blocks forever.
#   2. `ros2 run` is a Python wrapper around the real binary. Signalling only the PID that bash reports
#      can leave the child orphaned and running -- and an orphaned node KEEPS PUBLISHING on the pose topic,
#      so the next run's recorder interleaves two trajectories into one file. That produces a plausible
#      looking file that is actually two runs zipped together.
#
# So this matches by name and verifies afterwards, rather than trusting a PID. Escalating to KILL is
# safe because by the time we call this the run is complete and the trajectory is already on disk.
stop_estimator() {
  local sig pids
  for sig in INT TERM KILL; do
    pids="$(proc_pids "$ESTIMATOR_PAT")"
    [ -n "$pids" ] || return 0
    # shellcheck disable=SC2086
    kill -"$sig" $pids 2>/dev/null || true
    for _ in $(seq 1 6); do
      any_running "$ESTIMATOR_PAT" || return 0
      sleep 1
    done
    [ "$sig" = KILL ] || log "estimator still alive after SIG$sig, escalating"
  done
  any_running "$ESTIMATOR_PAT" && die "estimator survived SIGKILL -- refusing to continue, results would be corrupted"
  return 0
}

stop_recorder() {
  local pid="$1"
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 8); do
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# Refuse to start on top of an existing node. This is not tidiness: a second publisher on the pose topic
# corrupts the recording invisibly. Not auto-killed because a pre-existing node may be deliberate.
preflight() {
  local pids
  pids="$(proc_pids "$ESTIMATOR_PAT") $(proc_pids "$RECORDER_PAT")"
  if [ -n "${pids// /}" ]; then
    printf '\033[1;31m[eval] ERROR:\033[0m an OpenVINS node or recorder is already running:\n' >&2
    # shellcheck disable=SC2086
    ps -o pid,etime,args -p $pids 2>/dev/null >&2 || true
    printf '\nA second publisher on the pose topic silently interleaves two trajectories into one recording.\n' >&2
    printf 'Stop it first:  pkill -f %s; pkill -f record_traj.py\n' "$ESTIMATOR_PAT" >&2
    exit 1
  fi
}

# Replace `key: ...` in an OpenCV-flavoured YAML file, appending the key if it is not already there.
# A real YAML library is not usable here: these configs start with `%YAML:1.0`, which is the OpenCV
# FileStorage dialect and not valid YAML.
set_key() {
  local file="$1" key="$2" val="$3"
  if grep -qE "^${key}:" "$file"; then
    sed -i -E "s|^${key}:.*|${key}: ${val}|" "$file"
  else
    printf '%s: %s\n' "$key" "$val" >> "$file"
  fi
}

[ -n "$BAG" ] || die "BAG is not set (ROS2 bag directory)"
[ -n "$WS" ] || die "WS is not set (colcon workspace)"
[ -n "$BASE_CONFIG" ] || die "BASE_CONFIG is not set (estimator config directory)"
[ -n "$OUT_ROOT" ] || die "OUT_ROOT is not set (results directory)"
[ -d "$BAG" ] || die "bag not found: $BAG"
[ -d "$BASE_CONFIG" ] || die "base config dir not found: $BASE_CONFIG"
[ -f "$WS/install/setup.bash" ] || die "workspace not built: $WS/install/setup.bash"

# The ROS/colcon setup scripts reference unset variables (COLCON_TRACE, AMENT_TRACE_SETUP_FILES, ...),
# so `set -u` has to come off across the source or they abort. Restored immediately after.
set +u
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

mkdir -p "$OUT_ROOT"

GT_FILE="$OUT_ROOT/groundtruth.txt"

make_gt() {
  log "generating ground truth from $BAG"
  # Pin the datum to the estimator's, so the GT file and the filter's ENU frame share an origin by
  # construction rather than by both independently picking the first fix.
  local datum_arg=()
  local datum
  datum="$(grep -E '^gps_datum:' "$BASE_CONFIG/estimator_config.yaml" 2>/dev/null | sed -E 's|.*\[([^]]*)\].*|\1|' | tr -d ' ' || true)"
  if [ -n "$datum" ]; then
    datum_arg=(--datum "$datum")
    log "using fixed datum from config: $datum"
  fi
  python3 "$SCRIPT_DIR/make_gt.py" "$BAG" "$GT_FILE" --topic "$GPS_TOPIC" "${datum_arg[@]}"
}

run_variant() {
  local spec="$1" suffix="${2:-}"
  IFS='|' read -r name gps_en drop_start drop_end <<< "$spec"
  name="${name}${RUN_TAG:+_$RUN_TAG}${suffix}"

  local out="$OUT_ROOT/$name"
  rm -rf "$out"
  mkdir -p "$out"

  # Copy the whole config directory: estimator_config.yaml refers to the calibration files by *relative*
  # path (relative_config_imu / relative_config_imucam), so they have to travel with it.
  cp -r "$BASE_CONFIG/." "$out/config/"
  local cfg="$out/config/estimator_config.yaml"

  set_key "$cfg" "gps_enabled" "$gps_en"
  set_key "$cfg" "gps_dropout_start_secs" "$drop_start"
  set_key "$cfg" "gps_dropout_end_secs" "$drop_end"

  # Keep the total-state dump, but per run, so runs do not clobber each other in /tmp. It carries the
  # calibration evolution (cam intrinsics/extrinsics, IMU intrinsics) -- note it predates GPS and does
  # NOT contain the lever arm or the E-to-G transform, which only appear in run.log.
  set_key "$cfg" "filepath_est" "\"$out/total_state_est.txt\""
  set_key "$cfg" "filepath_std" "\"$out/total_state_std.txt\""
  set_key "$cfg" "filepath_gt" "\"$out/total_state_gt.txt\""

  log "=== $name === gps_enabled=$gps_en dropout=[$drop_start, $drop_end]"

  # Run the node directly rather than via subscribe.launch.py, matching how this is run by hand.
  # This is deliberate: the launch file unconditionally declares use_stereo / max_cameras /
  # save_total_state as node parameters, and YamlParser::parse_config() checks ROS parameters BEFORE
  # the YAML file -- so those launch defaults would silently override the config (notably
  # use_stereo, whose launch-file default of true overrides a config that sets it to false). run_subscribe_msckf sets
  # automatically_declare_parameters_from_overrides(true), so only the parameters we actually pass on
  # the command line exist, and everything else comes from the config as written.
  # Started plainly, NOT under setsid: setsid forks when its caller is already a process-group leader,
  # in which case $! names the short-lived setsid rather than the node -- so the shutdown check sees
  # "already gone" and the real node is orphaned and left publishing. stop_estimator() matches by name
  # instead, which does not depend on getting a PID right.
  preflight
  ros2 run ov_msckf run_subscribe_msckf --ros-args -p config_path:="$cfg" \
      > "$out/run.log" 2>&1 &
  local launch_pid=$!

  # Default topic is /poseimu because `ros2 run` applies no namespace and ROS2Visualizer publishes with
  # relative names. It would be /ov_msckf/poseimu under subscribe.launch.py. Passed explicitly so the
  # coupling to the launch method is visible here rather than buried in the recorder's default.
  python3 "$SCRIPT_DIR/record_traj.py" "$out/traj_est.txt" --topic "$POSE_TOPIC" > "$out/record.log" 2>&1 &
  local rec_pid=$!

  # Let the node come up and subscribe before any data flows, otherwise the opening frames are lost.
  sleep 6
  pgrep -f "$ESTIMATOR_PAT" >/dev/null 2>&1 || { tail -30 "$out/run.log"; die "$name: estimator died during startup"; }

  # Directly assert the condition that matters: exactly one publisher on the topic we record. This
  # catches a stale node that preflight missed (e.g. one that appeared between the check and now), and
  # it is checked BEFORE playback so a corrupted run costs seconds rather than four minutes.
  local npub
  npub="$(timeout 15 ros2 topic info "$POSE_TOPIC" 2>/dev/null | sed -n 's/^Publisher count: //p')"
  if [ "${npub:-0}" != "1" ]; then
    stop_recorder "$rec_pid"
    stop_estimator
    die "$name: expected exactly 1 publisher on $POSE_TOPIC, found '${npub:-unknown}'. Two publishers interleave two trajectories into one recording."
  fi

  log "$name: playing bag at rate $PLAYBACK_RATE"
  # Do not raise the playback rate above 1.0. Subscriptions use SensorDataQoS (best-effort, shallow
  # queue), so playing faster drops frames non-deterministically and makes runs incomparable.
  ros2 bag play "$BAG" --rate "$PLAYBACK_RATE" >> "$out/run.log" 2>&1

  log "$name: playback done, ${GRACE_SECS}s grace for queued frames"
  sleep "$GRACE_SECS"

  stop_recorder "$rec_pid"
  stop_estimator
  wait "$launch_pid" 2>/dev/null || true
  sleep 1

  local n
  n=$(grep -vc '^#' "$out/traj_est.txt" 2>/dev/null || echo 0)
  log "$name: recorded $n poses -> $out/traj_est.txt"
  [ "$n" -gt 100 ] || log "WARNING: $name recorded only $n poses -- check $out/run.log"
}

target="${1:-all}"

if [ "$target" = "gt" ]; then
  make_gt
  exit 0
fi

[ -f "$GT_FILE" ] || make_gt

# With REPEATS>1 each variant runs N times into <name>_run1..N. The suffix stays empty at REPEATS=1
# so single runs keep their plain directory names and stay comparable with earlier results.
run_repeats() {
  local spec="$1" i
  if [ "$REPEATS" -le 1 ]; then
    run_variant "$spec"
    return
  fi
  for i in $(seq 1 "$REPEATS"); do
    log "repeat $i/$REPEATS of ${spec%%|*}"
    run_variant "$spec" "_run$i"
  done
}

if [ "$target" = "all" ]; then
  for spec in "${VARIANTS[@]}"; do run_repeats "$spec"; done
else
  found=0
  for spec in "${VARIANTS[@]}"; do
    [ "${spec%%|*}" = "$target" ] && { run_repeats "$spec"; found=1; }
  done
  [ "$found" = 1 ] || die "unknown variant '$target' (have: $(printf '%s ' "${VARIANTS[@]%%|*}"))"
fi

log "done. analyze with:"
log "  python3 $SCRIPT_DIR/analyze.py $OUT_ROOT"
