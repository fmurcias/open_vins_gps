# GPS Fusion Evaluation Harness

Three evaluations of the GPS fusion (`ov_msckf/src/update/UpdaterGPS.h`) against a
ROS 2 bag: VIO-only accuracy, GPS-fused accuracy, and behaviour across a simulated GPS outage.

The harness is dataset-agnostic. It needs a bag with the sensor topics your estimator config
expects plus a `sensor_msgs/NavSatFix` topic, and an estimator config directory for that dataset.

## The ground-truth caveat — read this first

When a dataset has **no dedicated ground-truth file or topic**, the GNSS track is the only absolute
position reference available, so it serves as ground truth. That is honest in some places and
circular in others, and the distinction decides what each test actually proves:

| Run | GNSS independent? | What its ATE means |
|---|---|---|
| `A_vio_only` | Yes | Real accuracy. The filter never saw these fixes. |
| `B_gps` | **No** | Tracking tightness only — this run consumed these exact fixes. Not accuracy. |
| `C/D/E/...` outage window | **Yes** | Real accuracy. Fixes are withheld from the filter but still score it. |

So the outage windows are the strongest result in the set, not merely a robustness check. Two further
metrics avoid the circularity entirely:

- **Loop-closure error.** If the platform ends near where it started (the report prints the GNSS
  start-to-end distance next to the estimate's), how far the *estimate* drifts over the same loop
  measures accumulated drift regardless of what was fused. Only meaningful for closed-loop
  trajectories.
- **Run A vs run B.** A relative comparison, immune to the shared reference.

If your dataset *does* have independent ground truth, you can use that instead: write it in the
`ov_eval` format (`timestamp tx ty tz qx qy qz qw`) as `$OUT_ROOT/groundtruth.txt`, in the same
frame convention as the GNSS ENU track or accept that `analyze.py` aligns yaw and translation only.
The circularity caveats above then no longer apply.

One more limitation: GNSS carries no attitude, so ground-truth quaternions are identity. Position
metrics are valid; **any orientation error `ov_eval` reports is meaningless.**

## Usage

Set the required environment (or export these beforehand):

| Variable | Meaning |
|---|---|
| `BAG` | ROS 2 bag directory (IMU, camera(s), NavSatFix) |
| `BASE_CONFIG` | Estimator config directory for the dataset (contains `estimator_config.yaml` and the files it references) |
| `WS` | colcon workspace with `ov_msckf` built (`$WS/install/setup.bash`) |
| `OUT_ROOT` | Where results are written |

Optional: `GPS_TOPIC` (default `/imu/gnss`), `POSE_TOPIC` (default `/poseimu`), `PLAYBACK_RATE`
(default `1.0`), `GRACE_SECS`, `REPEATS`, `RUN_TAG`, `DROPOUT_START`, `DROPOUT_LENGTHS`.

```bash
cd ov_msckf/scripts/gps_eval

export BAG=/path/to/bag BASE_CONFIG=/path/to/config_dir WS=/path/to/ws OUT_ROOT=/path/to/results
export GPS_TOPIC=/your/navsatfix/topic

./run_eval.sh gt          # ground truth only (fast, verifies the ENU conversion)
./run_eval.sh             # all variants, one bag duration each at real-time playback
./run_eval.sh B_gps       # one variant
REPEATS=3 ./run_eval.sh B_gps   # repeat a variant (runs are not bit-reproducible)

python3 analyze.py <results/path>
```

Outputs land in `$OUT_ROOT`: `REPORT.md`, `error_vs_time.png`, and per run a `config/` directory
holding the exact config it ran with, `traj_est.txt`, and `run.log`.

### Verifying the ground truth

`make_gt.py` prints trajectory statistics: pose count, duration and rate, path length, maximum
distance from start, end-vs-start offset, ENU extent, and the datum used. Check them against what you
know about the dataset (does the duration match the bag? is the path length plausible? does a route
that should close actually close?). If they are off, the ENU conversion or the datum is wrong and every
downstream error number is wrong with it. This is a real check, not decoration.

The datum is taken from `gps_datum` in `estimator_config.yaml` when set, so the ground truth and the
filter share an ENU origin by construction; otherwise both use the first valid fix.

## The runs

| Run | Config |
|---|---|
| `A_vio_only` | `gps_enabled: false` |
| `B_gps` | `gps_enabled: true` |
| `C_dropout<N>`, `D_dropout<N>`, ... | `gps_enabled: true` + an outage of N seconds starting at `DROPOUT_START` |

Dropout variants are generated from two settings (defaults in parentheses):

- `DROPOUT_START` (`120`): outage start, in seconds.
- `DROPOUT_LENGTHS` (`"30 60 90"`): one variant per length, lettered C, D, E, … in order. Set to an
  empty string to run only A and B.

Outage times are seconds relative to the **first GPS fix**, not absolute epoch time, so they transfer
between bags. Choose them for your dataset: start after the filter has converged, leave enough bag
after the outage to observe re-convergence, and note the unaided distance is roughly platform speed ×
outage length.

`analyze.py` picks up any run directory in the results root, so custom variants, repeats and tagged
re-runs (`RUN_TAG`) need no changes to it.

## How the outage is simulated

`gps_dropout_start_secs` / `gps_dropout_end_secs` are applied in
`VioManager::feed_measurement_gps()`, which drops the fix before it reaches the queue or the updater.
Nothing downstream is told the fix existed, so no `try_update()` runs and no chi2 rejections
accumulate — the rejection-streak reset in `UpdaterGPS` therefore cannot fire *during* the gap. What
gets measured on the far side is purely how the filter copes when fixes reappear against a state that
has drifted, which is the behaviour under test.

Set both to `-1` (the default) to disable. These are evaluation-only knobs.

## Gotchas worth knowing

**Do not use `subscribe.launch.py` for evaluation runs.** `YamlParser::parse_config()` checks ROS
parameters *before* the YAML file, and the launch file unconditionally declares `use_stereo`
(default `true`), `max_cameras`, and `save_total_state`. Those defaults therefore silently override
the config — notably a config that sets `use_stereo: false` would run in stereo mode anyway.

`run_eval.sh` runs the node directly instead:

```bash
ros2 run ov_msckf run_subscribe_msckf --ros-args -p config_path:=<cfg>
```

`run_subscribe_msckf` sets `automatically_declare_parameters_from_overrides(true)`, so only the
parameters actually passed on the command line exist and everything else comes from the config as
written. This matches how the node is normally run by hand, which keeps scripted and manual runs
comparable.

**Playback rate should stay at 1.0.** Subscriptions use `rclcpp::SensorDataQoS()` — best-effort with a
shallow queue — so faster playback drops frames non-deterministically and makes runs incomparable.
`analyze.py` warns if the associated-pose counts vary by more than 5% across runs; treat that warning
as invalidating the cross-run comparison rather than as noise.

**Topic names depend on how the node is started.** `ROS2Visualizer` publishes with relative names, so
`ros2 run` gives `/poseimu` while `subscribe.launch.py` gives `/ov_msckf/poseimu`. `run_eval.sh` uses
`ros2 run`; override `POSE_TOPIC` if you change that. `record_traj.py` reports the pose-like topics it
can see if nothing arrives within a few seconds.

**Clocks must match.** Estimate timestamps are on the IMU clock. If your GNSS stamps are offset from
it, set `gps_toff` appropriately (or correct the ground-truth file), otherwise every error figure
carries the offset.

**`record_traj.py` exists because `pose_to_file` does not.** `ov_eval/src/pose_to_file.cpp` is
ROS1-only (`ros::init`) and commented out of `ov_eval/cmake/ROS2.cmake`, so there is no ROS 2 path to
a scorable trajectory without it. Note that it swaps the covariance blocks: `publish_state()` emits
`[position | orientation]` per ROS convention, while `ov_eval`'s `Loader` expects orientation first.
Getting that backwards silently corrupts NEES rather than failing.

**Frozen-filter detection is tuned empirically.** `analyze.py` flags a run as failed if the
accelerometer bias stays bit-identical for `--freeze-frames` consecutive camera frames (default 500),
and flags truncated runs relative to the cohort median pose count. The frame threshold scales with
camera rate; on a new dataset check the `max freeze` column of known-good runs and adjust.

## Cross-checking with ov_eval

The standard tool works on these files directly:

```bash
ros2 run ov_eval error_singlerun posyaw \
    $OUT_ROOT/groundtruth.txt \
    $OUT_ROOT/A_vio_only/traj_est.txt
```

Its position ATE/RPE should match `analyze.py`. Ignore its orientation columns.
