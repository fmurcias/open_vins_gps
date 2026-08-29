#!/usr/bin/env python3
"""
Record the estimated trajectory to an ov_eval-format file.

This is the ROS2 stand-in for ov_eval/src/pose_to_file.cpp, which is ROS1-only (it calls ros::init /
nh.subscribe) and is commented out of ov_eval/cmake/ROS2.cmake -- so there is currently no way to get
a scorable trajectory out of a ROS2 OpenVINS run.

Output is the ov_eval Loader format (space separated, see ov_eval/src/utils/Loader.cpp):

    timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33

IMPORTANT -- the covariance blocks are swapped relative to the ROS message. ROS2Visualizer::
publish_state() builds its marginal covariance with statevars = {p(), q()} and fills the message
"position then orientation as per ros convention", so the incoming 6x6 is [[P_pp, P_pq], [P_qp, P_qq]].
The Loader wants the *orientation* upper-triangle (Pr) first and the *position* upper-triangle (Pt)
second. Getting this backwards does not fail loudly -- it silently corrupts NEES -- so the mapping is
done explicitly below.

Timestamps need no correction: publish_state() stamps with state->_timestamp + calib_dt_CAMtoIMU (IMU
clock) and gps_toff is 0 for this dataset, so estimate and GNSS ground truth share a clock.

TOPIC NAMING -- the default is `/poseimu`, NOT `/ov_msckf/poseimu`. ROS2Visualizer creates its
publishers with relative names ("poseimu"), so the resolved topic depends entirely on how the node was
started: `ros2 run` applies no namespace and you get `/poseimu`, whereas subscribe.launch.py sets
namespace=ov_msckf and you get `/ov_msckf/poseimu`. run_eval.sh uses `ros2 run`, hence this default.
If nothing arrives within a few seconds this script says so and lists the pose topics it can actually
see, rather than silently recording an empty file for the length of a bag.

Usage:
    record_traj.py <out_est.txt> [--topic /poseimu]

Writes incrementally and flushes on shutdown, so a Ctrl-C or a killed launch still leaves a usable file.

AI Generated
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped

HEADER = ("# timestamp(s) tx ty tz qx qy qz qw "
          "Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33\n")


class TrajRecorder(Node):
    def __init__(self, out_path, topic):
        super().__init__("gps_eval_traj_recorder")
        self.count = 0
        self.f = open(out_path, "w")
        self.f.write(HEADER)

        # The publisher side uses a depth-2 reliable queue; match reliability so we do not silently
        # drop poses, and keep some depth so a slow disk cannot cost us samples.
        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.topic = topic
        self.sub = self.create_subscription(PoseWithCovarianceStamped, topic, self.callback, qos)
        self.get_logger().info("recording %s -> %s" % (topic, out_path))

        # Fail loudly and early on a topic mismatch. Subscribing to a name nobody publishes is silent
        # in ROS, so without this the first sign of trouble is an empty file after a full bag replay.
        self.warned = False
        self.timer = self.create_timer(12.0, self.check_alive)

    def check_alive(self):
        if self.count > 0 or self.warned:
            self.timer.cancel()
            return
        self.warned = True
        names = ["%s [%s]" % (n, ",".join(t)) for n, t in self.get_topic_names_and_types()
                 if "pose" in n.lower() or "odom" in n.lower()]
        self.get_logger().error(
            "NO MESSAGES on '%s' after 12s. Pose-like topics currently visible: %s. "
            "Note ROS2Visualizer uses relative topic names, so 'ros2 run' gives /poseimu while "
            "subscribe.launch.py gives /ov_msckf/poseimu -- pass --topic to match how the node was started."
            % (self.topic, ", ".join(names) if names else "(none)"))

    def callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        c = msg.pose.covariance  # 6x6 row-major, ROS order: [position | orientation]

        def blk(r0, c0):
            """Upper triangle (11 12 13 22 23 33) of the 3x3 block at (r0, c0) of the 6x6."""
            return [c[6 * (r0 + i) + (c0 + j)] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]

        pr = blk(3, 3)  # orientation block -> Loader's Pr columns
        pt = blk(0, 0)  # position block    -> Loader's Pt columns

        self.f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f %s %s\n" % (
            t, p.x, p.y, p.z, q.x, q.y, q.z, q.w,
            " ".join("%.9e" % v for v in pr),
            " ".join("%.9e" % v for v in pt),
        ))
        self.count += 1
        if self.count % 500 == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()
        self.get_logger().info("wrote %d poses" % self.count)
        print("[record_traj] wrote %d poses" % self.count, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_est")
    ap.add_argument("--topic", default="/poseimu",
                    help="default suits `ros2 run` (no namespace); use /ov_msckf/poseimu with subscribe.launch.py")
    args, unknown = ap.parse_known_args()

    rclpy.init(args=unknown)
    node = TrajRecorder(args.out_est, args.topic)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # ExternalShutdownException is the normal path when we are SIGINT'd: rclpy tears the context
        # down underneath spin(). Both mean "stop cleanly", not "something went wrong".
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
