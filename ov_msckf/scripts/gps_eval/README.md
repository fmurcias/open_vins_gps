# GPS Fusion Evaluation Harness

Three evaluations of the GPS fusion (`ov_msckf/src/update/UpdaterGPS.h`) against a
real bag: VIO-only accuracy, GPS-fused accuracy, and behaviour across a simulated GPS outage.

## The ground-truth caveat — read this first

This dataset has **no ground-truth file and no GT topic**. The GNSS track is the only absolute
position reference available, so it serves as ground truth. That is honest in some places and
circular in others, and the distinction decides what each test actually proves:

| Run | GNSS independent? | What its ATE means |
|---|---|---|
| `A_vio_only` | Yes | Real accuracy. The filter never saw these fixes. |
| `B_gps` | **No** | Tracking tightness only — this run consumed these exact fixes. Not accuracy. |
| `C/D/E` outage window | **Yes** | Real accuracy. Fixes are withheld from the filter but still score it. |

So the outage windows are the strongest result in the set, not merely a robustness check. Two further
metrics avoid the circularity entirely:

- **Loop-closure error.** The platform returns to within 0.74 m horizontally / 0.97 m vertically of
  its start, so how far the *estimate* drifts over the same loop measures accumulated drift
  regardless of what was fused.
- **Run A vs run B.** A relative comparison, immune to the shared reference.

One more limitation: GNSS carries no attitude, so ground-truth quaternions are identity. Position
metrics are valid; **any orientation error `ov_eval` reports is meaningless.**

## Usage

```bash
cd ov_msckf/scripts/gps_eval

./run_eval.sh gt          # ground truth only (fast, verifies the ENU conversion)
./run_eval.sh             # all five runs, ~25 min at real-time playback
./run_eval.sh B_gps       # one variant

python3 analyze.py <results/path>
```

Override paths via environment: `BAG`, `BASE_CONFIG`, `OUT_ROOT`, `WS`, `GRACE_SECS`.

Outputs land in `$OUT_ROOT`: `REPORT.md`, `error_vs_time.png`, and per run a `config/` directory
holding the exact config it ran with, `traj_est.txt`, and `run.log`.

### Verifying the ground truth

`make_gt.py` prints trajectory statistics. For the `40_4_ros2` bag these must come out as:

```
poses 967 | 241.5 s @ 4.00 Hz | path 713.8 m | max 95.9 m from start
end vs start: 0.74 m horiz, -0.97 m vert
ENU extent: x [-95.8, 43.0]  y [-36.1, 26.2]  z [-1.0, 37.6]
```

If they differ, the ENU conversion or the datum is wrong and every downstream error number is wrong
with it. This is a real check, not decoration.

## The runs

| Run | Config |
|---|---|
| `A_vio_only` | `gps_enabled: false` |
| `B_gps` | `gps_enabled: true` |
| `C_dropout30` | + outage t+120 → t+150 s (~90 m unaided) |
| `D_dropout60` | + outage t+120 → t+180 s (~180 m unaided) |
| `E_dropout90` | + outage t+120 → t+210 s (~270 m unaided) |

Outage times are seconds relative to the **first GPS fix**, not absolute epoch time, so they
transfer to other bags unchanged.

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
the config — notably `use_stereo`, which is `false` in `fgi_masala` but `true` in the launch file.

`run_eval.sh` runs the node directly instead:

```bash
ros2 run ov_msckf run_subscribe_msckf --ros-args -p config_path:=<cfg>
```

`run_subscribe_msckf` sets `automatically_declare_parameters_from_overrides(true)`, so only the
parameters actually passed on the command line exist and everything else comes from the config as
written. This matches how the node is normally run by hand, which keeps scripted and manual runs
comparable.

**Playback rate must stay at 1.0.** Subscriptions use `rclcpp::SensorDataQoS()` — best-effort with a
shallow queue — so faster playback drops frames non-deterministically and makes runs incomparable.
`analyze.py` warns if the associated-pose counts vary by more than 5% across runs; treat that warning
as invalidating the cross-run comparison rather than as noise.

**`record_traj.py` exists because `pose_to_file` does not.** `ov_eval/src/pose_to_file.cpp` is
ROS1-only (`ros::init`) and commented out of `ov_eval/cmake/ROS2.cmake`, so there is no ROS 2 path to
a scorable trajectory without it. Note that it swaps the covariance blocks: `publish_state()` emits
`[position | orientation]` per ROS convention, while `ov_eval`'s `Loader` expects orientation first.
Getting that backwards silently corrupts NEES rather than failing.

## Cross-checking with ov_eval

The standard tool works on these files directly:

```bash
ros2 run ov_eval error_singlerun posyaw \
    ~/VIO/gps_eval_results/groundtruth.txt \
    ~/VIO/gps_eval_results/A_vio_only/traj_est.txt
```

Its position ATE/RPE should match `analyze.py`. Ignore its orientation columns.
