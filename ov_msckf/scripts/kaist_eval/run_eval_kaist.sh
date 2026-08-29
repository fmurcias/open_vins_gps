#!/usr/bin/env bash
#
# Orchestrates the deterministic serial KAIST evaluation runs (ov_msckf/src/run_serial_kaist.cpp).
#
#   ./run_eval_kaist.sh gt          # (re)generate ground truth only
#   ./run_eval_kaist.sh smoke       # 60s truncated run of A and B, for a quick sanity check
#   ./run_eval_kaist.sh determinism # run B_gps twice and diff the outputs -- see README
#   ./run_eval_kaist.sh A_vio_only  # one named variant
#   ./run_eval_kaist.sh             # all variants
#
# Unlike ov_msckf/scripts/gps_eval/run_eval.sh (which plays a ROS2 bag in real time and has to
# manage a background node/recorder pair, QoS publisher counts, PID tracking, etc.), this harness
# has none of that: run_serial_kaist is a single blocking, synchronous, in-process call per run --
# there is no background process to orchestrate, and no real-time playback to wait out.
#
# Override via environment: DATASET_DIR, GT_CSV, BASE_CONFIG, BIN, OUT_ROOT.
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-/media/paco/datasets/kaist/urban27-dongtan_data/urban27-dongtan}"
GT_CSV="${GT_CSV:-/media/paco/datasets/kaist/ground_truth/urban27-dongtan/global_pose.csv}"
WS="${WS:-$HOME/VIO/ros2_ws}"
BASE_CONFIG="${BASE_CONFIG:-$WS/src/open_vins/config/kaist_gps}"
BIN="${BIN:-$WS/build/ov_msckf/run_serial_kaist}"
OUT_ROOT="${OUT_ROOT:-$HOME/VIO/kaist_eval_results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The ov_eval auto-discovery layout (error_dataset.cpp) is <results_root>/<algo_name>/<gt_stem>/*.txt
# -- GT_STEM must match the ground-truth file's basename (without extension) exactly.
# Overridable so a second sequence (e.g. urban28) can be scored without clobbering the first's
# results: set GT_STEM together with DATASET_DIR/GT_CSV/BASE_CONFIG/OUT_ROOT.
GT_STEM="${GT_STEM:-urban27}"
GT_TUM="$OUT_ROOT/${GT_STEM}.txt"

# $BIN itself never calls a ROS API, but when built inside the ROS2 colcon workspace it links
# against ov_msckf_lib, which (in that build) also compiles in ROS2Visualizer and pulls in ROS2's
# shared libraries transitively -- so it still needs the ROS2 environment sourced to *run*, even
# though it does no ROS work. Source it if present; do not hard-fail if it's not (a binary built via
# the standalone `cmake ../ov_msckf/` no-ROS path has zero ROS linkage and needs none of this). Same
# `set +u`/`set -u` dance as gps_eval/run_eval.sh: the ROS setup scripts reference variables that
# don't exist under `set -u`.
if [ -f "$WS/install/setup.bash" ]; then
  set +u
  # shellcheck disable=SC1091
  source "$WS/install/setup.bash"
  set -u
fi

# Variant table: NAME | gps_enabled | dropout_start | dropout_end (seconds relative to the FIRST
# GPS fix, see gps_dropout_start_secs/end_secs doc in UpdaterGPSOptions.h). The sequence is ~1157s
# over ~5.25km; t+400 sits comfortably mid-run with plenty of runway left for the longest outage.
VARIANTS=(
  "A_vio_only|false|-1|-1"
  "B_gps|true|-1|-1"
  "C_dropout30|true|400|430"
  "D_dropout60|true|400|460"
  "E_dropout90|true|400|490"
)

log() { printf '\033[1;36m[eval]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[eval] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$DATASET_DIR/sensor_data" ] || die "dataset not found: $DATASET_DIR/sensor_data"
[ -e "$DATASET_DIR/image" ] || die "no image/ under $DATASET_DIR -- symlink it from the *_img archive first, e.g.:" \
  "  ln -s <...>_img-001/urban27-dongtan/image $DATASET_DIR/image"
[ -f "$BASE_CONFIG/estimator_config.yaml" ] || die "base config not found: $BASE_CONFIG/estimator_config.yaml"
[ -x "$BIN" ] || die "run_serial_kaist binary not found/executable at $BIN (build it first)"

mkdir -p "$OUT_ROOT"

# Replace `key: ...` in an OpenCV-flavoured YAML file (same helper/same caveat as gps_eval/run_eval.sh:
# these configs start with %YAML:1.0, the OpenCV FileStorage dialect, not valid plain YAML).
set_key() {
  local file="$1" key="$2" val="$3"
  if grep -qE "^${key}:" "$file"; then
    sed -i -E "s|^${key}:.*|${key}: ${val}|" "$file"
  else
    printf '%s: %s\n' "$key" "$val" >> "$file"
  fi
}

make_gt() {
  log "generating ground truth from $GT_CSV"
  python3 "$SCRIPT_DIR/make_gt_kaist.py" "$GT_CSV" "$GT_TUM"
}

run_variant() {
  local spec="$1"
  IFS='|' read -r name gps_en drop_start drop_end <<< "$spec"

  local out_dir="$OUT_ROOT/$name/$GT_STEM"
  mkdir -p "$out_dir"
  local cfg_dir="$OUT_ROOT/$name/config"
  rm -rf "$cfg_dir"
  cp -r "$BASE_CONFIG/." "$cfg_dir/"
  local cfg="$cfg_dir/estimator_config.yaml"

  set_key "$cfg" "gps_enabled" "$gps_en"
  set_key "$cfg" "gps_dropout_start_secs" "$drop_start"
  set_key "$cfg" "gps_dropout_end_secs" "$drop_end"

  log "=== $name === gps_enabled=$gps_en dropout=[$drop_start, $drop_end]"
  "$BIN" "$cfg" "$DATASET_DIR" "$out_dir/run1.txt"

  local n
  n=$(grep -vc '^#' "$out_dir/run1.txt" 2>/dev/null || echo 0)
  log "$name: wrote $n poses -> $out_dir/run1.txt"
  [ "$n" -gt 100 ] || log "WARNING: $name wrote only $n poses -- check the run's stdout above"
}

target="${1:-all}"

if [ "$target" = "gt" ]; then
  make_gt
  exit 0
fi

[ -f "$GT_TUM" ] || make_gt

if [ "$target" = "smoke" ]; then
  # Quick sanity check on a truncated window, before committing to the full ~19min run: exercises the
  # merge ordering, Bayer demosaic, and output format without the full runtime cost.
  #
  # 120s, not 60s: the vehicle sits stationary (ZUPT engaged, no rotation/translation) for roughly the
  # first 50-60s of this sequence, and VIO-only (A) needs slightly more accumulated motion after that
  # to clear its dynamic-init threshold than the GPS-fused config (B) does -- observed directly: at a
  # 60s cutoff A_vio_only writes 0 poses (never initializes in time) while B_gps writes 151, purely
  # because the init event falls just past the 60s boundary for A and just before it for B. That is a
  # smoke-window sizing issue, not a bug -- at 120s both variants initialize and produce comparable
  # pose counts (A: 487/1201 frames, B: similar). Keep the window long enough to clear this reliably.
  for spec in "A_vio_only|false|-1|-1" "B_gps|true|-1|-1"; do
    IFS='|' read -r name gps_en drop_start drop_end <<< "$spec"
    local_out="$OUT_ROOT/smoke_$name"
    mkdir -p "$local_out"
    cfg_dir="$local_out/config"
    rm -rf "$cfg_dir"
    cp -r "$BASE_CONFIG/." "$cfg_dir/"
    set_key "$cfg_dir/estimator_config.yaml" "gps_enabled" "$gps_en"
    log "=== smoke $name (first 120s) ==="
    "$BIN" "$cfg_dir/estimator_config.yaml" "$DATASET_DIR" "$local_out/run1.txt" 120
  done
  exit 0
fi

if [ "$target" = "determinism" ]; then
  # Required gate before trusting any A/B comparison -- see plan verification step 3.
  cfg_dir="$OUT_ROOT/determinism_check/config"
  rm -rf "$cfg_dir"; mkdir -p "$cfg_dir"
  cp -r "$BASE_CONFIG/." "$cfg_dir/"
  set_key "$cfg_dir/estimator_config.yaml" "gps_enabled" "true"
  log "=== determinism check: running B_gps twice on a 120s window ==="
  "$BIN" "$cfg_dir/estimator_config.yaml" "$DATASET_DIR" "$OUT_ROOT/determinism_check/run1.txt" 120
  "$BIN" "$cfg_dir/estimator_config.yaml" "$DATASET_DIR" "$OUT_ROOT/determinism_check/run2.txt" 120
  if diff -q "$OUT_ROOT/determinism_check/run1.txt" "$OUT_ROOT/determinism_check/run2.txt" > /dev/null; then
    log "PASS: run1.txt and run2.txt are byte-identical"
  else
    die "FAIL: run1.txt and run2.txt differ -- see $OUT_ROOT/determinism_check/"
  fi
  exit 0
fi

if [ "$target" = "all" ]; then
  for spec in "${VARIANTS[@]}"; do run_variant "$spec"; done
else
  found=0
  for spec in "${VARIANTS[@]}"; do
    [ "${spec%%|*}" = "$target" ] && { run_variant "$spec"; found=1; }
  done
  [ "$found" = 1 ] || die "unknown variant '$target' (have: gt smoke determinism $(printf '%s ' "${VARIANTS[@]%%|*}"))"
fi

log "done. score with, e.g.:"
log "  \$WS/build/ov_eval/error_singlerun posyaw $GT_TUM $OUT_ROOT/B_gps/$GT_STEM/run1.txt"
log "  \$WS/build/ov_eval/error_dataset posyaw $GT_TUM $OUT_ROOT"
