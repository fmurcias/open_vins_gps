#!/usr/bin/env python3
"""
Convert the KAIST Complex Urban Dataset's global_pose.csv into an ov_eval-format ground-truth
trajectory.

Unlike the other GPS-eval dataset (ov_msckf/scripts/gps_eval/), this is REAL 6-DoF ground truth --
global_pose.csv is KAIST's own globally-optimized reference trajectory (full rotation, not just a
GNSS position track), so orientation error is meaningful here, not just position.

Input format (no header, comma-separated), one pose per row:
    stamp_ns, R00,R01,R02,tx, R10,R11,R12,ty, R20,R21,R22,tz
i.e. a row-major 3x4 [R|t] matrix, position in a local UTM-like frame.

Output is the ov_eval Loader format (space separated, see ov_eval/src/utils/Loader.cpp):
    timestamp(s) tx ty tz qx qy qz qw

Timestamp conversion (stamp_ns -> seconds) uses the exact same formula
(`t_sec = stamp_ns * 1e-9`) as ov_msckf/src/run_serial_kaist.cpp -- both sides of any comparison
ov_eval does (association within a max_dt window) depend on this being identical, not just close.

Usage:
    make_gt_kaist.py <global_pose.csv> <out_gt.txt>

AI Generated
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation


def ns_to_sec(stamp_ns):
    # Kept bit-identical in spirit to run_serial_kaist.cpp's ns_to_sec(): a single multiply, no
    # separate integer/fractional split that could round differently.
    return stamp_ns * 1e-9


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <global_pose.csv> <out_gt.txt>", file=sys.stderr)
        sys.exit(1)
    path_in, path_out = sys.argv[1], sys.argv[2]

    times = []
    positions = []
    quats = []  # (x, y, z, w)
    n_bad_rotation = 0

    with open(path_in) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 13:
                continue
            # Parse the leading nanosecond-epoch timestamp as an int directly from the string, NOT
            # via float(): KAIST timestamps (e.g. 1544582648748380970) have up to 19 significant
            # digits, but a Python float (IEEE754 double) is only exact up to 2^53 (~16 digits) --
            # routing it through float() first would silently round it by up to a few hundred
            # nanoseconds before ns_to_sec() ever runs. Must match run_serial_kaist.cpp exactly.
            stamp_ns = int(fields[0])
            vals = [float(x) for x in fields[1:]]  # R00,R01,R02,tx, R10,R11,R12,ty, R20,R21,R22,tz
            R = np.array(
                [
                    [vals[0], vals[1], vals[2]],
                    [vals[4], vals[5], vals[6]],
                    [vals[8], vals[9], vals[10]],
                ]
            )
            t = np.array([vals[3], vals[7], vals[11]])

            # Sanity check: a genuine rotation matrix has det ~= 1. Catches a column/row transposition
            # bug immediately instead of silently producing a subtly-wrong ground truth.
            det = np.linalg.det(R)
            if abs(det - 1.0) > 1e-3:
                n_bad_rotation += 1
                continue

            q = Rotation.from_matrix(R).as_quat()  # scipy: (x, y, z, w)
            times.append(ns_to_sec(stamp_ns))
            positions.append(t)
            quats.append(q)

    if n_bad_rotation > 0:
        print(f"WARNING: {n_bad_rotation} row(s) had |det(R)-1| > 1e-3, skipped", file=sys.stderr)
    if not times:
        print("ERROR: no valid poses parsed", file=sys.stderr)
        sys.exit(1)

    positions = np.array(positions)
    with open(path_out, "w") as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
        for t, p, q in zip(times, positions, quats):
            f.write(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")

    # Sanity-check summary, same intent as gps_eval/make_gt.py's printed stats for the other dataset:
    # if the extent/path-length/rate don't look physically plausible for an urban driving sequence, the
    # LLA/rotation conversion (or a column-order mixup) is wrong and every downstream error number is
    # wrong with it.
    duration = times[-1] - times[0]
    rate = (len(times) - 1) / duration if duration > 0 else 0.0
    deltas = np.diff(positions, axis=0)
    path_length = np.sum(np.linalg.norm(deltas, axis=1))
    dists_from_start = np.linalg.norm(positions - positions[0], axis=1)
    end_vs_start = positions[-1] - positions[0]
    print(f"poses {len(times)} | {duration:.1f} s @ {rate:.2f} Hz | path {path_length:.1f} m | "
          f"max {dists_from_start.max():.1f} m from start")
    print(f"end vs start: {np.linalg.norm(end_vs_start[:2]):.2f} m horiz, {end_vs_start[2]:.2f} m vert")
    print(f"extent: x [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]  "
          f"y [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]  "
          f"z [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")
    print(f"wrote {path_out}")


if __name__ == "__main__":
    main()
