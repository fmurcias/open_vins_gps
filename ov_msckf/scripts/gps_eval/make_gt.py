#!/usr/bin/env python3
"""
Build an ov_eval-format ground-truth trajectory from the GNSS track in a ROS2 bag.

The GNSS fixes are the only absolute position reference available for this dataset, so they serve as
ground truth. Read the caveat in README.md before interpreting anything: scoring a *GPS-fused* run
against this file is circular. It is honest ground truth for a VIO-only run, and for the outage
window of a dropout run (where the filter never saw these fixes).

Output is the ov_eval Loader format (space separated):

    timestamp(s) tx ty tz qx qy qz qw

GNSS carries no attitude, so the quaternion is identity on every row. That is fine for the "posyaw"
alignment mode -- AlignTrajectory::align_posyaw uses positions only when aligning over all poses --
but it does mean any *orientation* error ov_eval reports is meaningless. Use the position columns.

Usage:
    make_gt.py <bag_dir> <out_gt.txt> [--topic /imu/gnss] [--datum LAT,LON,ALT]

AI Generated
"""

import argparse
import math
import sys

# WGS84, kept bit-identical to ov_core/src/utils/gps_conv.h so the ENU frame here and the ENU frame
# the estimator fuses in are the same frame. Do not "simplify" these to spherical-earth formulas.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla2ecef(lat_deg, lon_deg, alt_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return (
        (n + alt_m) * cos_lat * cos_lon,
        (n + alt_m) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
    )


def ecef2enu(ecef, ecef0, lat0_deg, lon0_deg):
    lat0, lon0 = math.radians(lat0_deg), math.radians(lon0_deg)
    sin_lat0, cos_lat0 = math.sin(lat0), math.cos(lat0)
    sin_lon0, cos_lon0 = math.sin(lon0), math.cos(lon0)
    d = (ecef[0] - ecef0[0], ecef[1] - ecef0[1], ecef[2] - ecef0[2])
    return (
        -sin_lon0 * d[0] + cos_lon0 * d[1],
        -sin_lat0 * cos_lon0 * d[0] - sin_lat0 * sin_lon0 * d[1] + cos_lat0 * d[2],
        cos_lat0 * cos_lon0 * d[0] + cos_lat0 * sin_lon0 * d[1] + sin_lat0 * d[2],
    )


def read_fixes(bag_dir, topic):
    """Return [(t, lat, lon, alt, status)] for every NavSatFix on `topic`, in bag order."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import NavSatFix

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    out = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        m = deserialize_message(data, NavSatFix)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        out.append((t, m.latitude, m.longitude, m.altitude, m.status.status))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir")
    ap.add_argument("out_gt")
    ap.add_argument("--topic", default="/imu/gnss")
    ap.add_argument(
        "--datum",
        default=None,
        help="Fixed ENU origin as LAT,LON,ALT. Should match gps_datum in the estimator config so both "
        "sides share an origin by construction. Defaults to the first valid fix (what the estimator "
        "does when gps_datum is unset).",
    )
    args = ap.parse_args()

    fixes = read_fixes(args.bag_dir, args.topic)
    if not fixes:
        sys.exit("ERROR: no messages on {} in {}".format(args.topic, args.bag_dir))

    # Mirror ROS2Visualizer::callback_gps, which rejects anything below STATUS_FIX (0) before it ever
    # reaches the estimator. Keeping a fix here that the estimator discarded would silently score the
    # filter against data it was never given.
    valid = [f for f in fixes if f[4] >= 0]
    n_dropped = len(fixes) - len(valid)
    if not valid:
        sys.exit("ERROR: no fixes with status >= STATUS_FIX")

    if args.datum:
        lat0, lon0, alt0 = (float(v) for v in args.datum.split(","))
    else:
        lat0, lon0, alt0 = valid[0][1], valid[0][2], valid[0][3]

    ecef0 = lla2ecef(lat0, lon0, alt0)
    rows = []
    for t, lat, lon, alt, _ in valid:
        e = ecef2enu(lla2ecef(lat, lon, alt), ecef0, lat0, lon0)
        rows.append((t, e))

    with open(args.out_gt, "w") as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
        for t, e in rows:
            f.write("%.9f %.6f %.6f %.6f 0.0 0.0 0.0 1.0\n" % (t, e[0], e[1], e[2]))

    # Report the same figures the plan recorded from the bag. If these drift, the ENU conversion or the
    # datum is wrong and every downstream error number is wrong with it -- so this is a real check, not
    # decoration.
    pts = [r[1] for r in rows]
    path = sum(math.dist(pts[i], pts[i - 1]) for i in range(1, len(pts)))
    dur = rows[-1][0] - rows[0][0]
    end_h = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    end_v = pts[-1][2] - pts[0][2]
    print("wrote %s" % args.out_gt)
    print("  poses            : %d (%d dropped for status < STATUS_FIX)" % (len(rows), n_dropped))
    print("  duration         : %.1f s  (%.2f Hz)" % (dur, len(rows) / dur if dur > 0 else 0.0))
    print("  path length      : %.1f m" % path)
    print("  max dist from t0 : %.1f m" % max(math.hypot(p[0], p[1]) for p in pts))
    print("  end vs start     : %.2f m horiz, %.2f m vert" % (end_h, end_v))
    print("  datum            : %.8f, %.8f, %.3f%s" % (lat0, lon0, alt0, "" if args.datum else " (first valid fix)"))
    print("  ENU extent       : x [%.1f, %.1f]  y [%.1f, %.1f]  z [%.1f, %.1f]"
          % (min(p[0] for p in pts), max(p[0] for p in pts),
             min(p[1] for p in pts), max(p[1] for p in pts),
             min(p[2] for p in pts), max(p[2] for p in pts)))


if __name__ == "__main__":
    main()
