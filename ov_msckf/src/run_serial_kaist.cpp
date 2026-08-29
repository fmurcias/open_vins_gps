/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// Deterministic, ROS-free serial player for the raw KAIST Complex Urban Dataset
// (sensor_data/*.csv + image/stereo_{left,right}/*.png -- no rosbag involved anywhere).
//
// This exists because playing a converted rosbag in real time (the approach used for the other
// GPS-eval dataset, see ov_msckf/scripts/gps_eval/) is not deterministic: ROS subscriptions there
// use SensorDataQoS (best-effort, shallow queue), so faster-than-1x playback drops frames
// non-deterministically, and thread timing in the GPS drain path can land delayed init on different
// geometry run to run (see that harness's own README on why it needs REPEATS). OpenVINS's estimator
// core (VioManager/State/Propagator/updaters) has no ROS dependency at all -- only ov_msckf/src/ros/
// and the run_subscribe_msckf/ros1_serial_msckf entry points touch ROS -- so there is nothing
// stopping a direct, in-process, single-threaded feed loop that reads the dataset files and calls
// VioManager::feed_measurement_*() itself, with no message passing, no queues, and no real-time
// throttling at all. That is what this file does.
//
// Determinism has two parts, and both matter:
//  (1) Correct ordering: all IMU/GPS/camera measurements are merged into ONE strictly
//      timestamp-ordered stream (see the 3-way merge in main() below) before being fed. This is not
//      cosmetic -- VioManager::feed_measurement_camera() does NOT buffer or wait for IMU; it assumes
//      every IMU sample covering [state timestamp, this camera timestamp] has already been fed via
//      feed_measurement_imu(). Feeding in strict global timestamp order guarantees that invariant by
//      construction rather than by luck.
//  (2) No internal parallelism: VioManagerOptions has THREE threading knobs
//      (num_opencv_threads, use_multi_threading_pubs, use_multi_threading_subs) that run_simulation.cpp
//      and test_gps_sim.cpp already force to 0/false/false "for repeatability" -- same here. There is a
//      FOURTH, easy to miss: InertialInitializerOptions::init_dyn_mle_max_threads feeds directly into
//      ceres::Solver::Options.num_threads (see ov_init/src/dynamic/DynamicInitializer.cpp:629,1010).
//      Ceres does not guarantee a fixed reduction order across threads, so a value >1 there is a real
//      determinism risk independent of the other three. config/kaist sets it to 6; we force it to 1.
//
// Usage: run_serial_kaist <config_path> <kaist_dataset_dir> <output_traj_path> [max_relative_secs]
//   config_path        Path to an estimator_config.yaml (e.g. config/kaist_gps/estimator_config.yaml)
//   kaist_dataset_dir  Path to the raw KAIST sequence root, i.e. the directory that directly contains
//                      sensor_data/ and image/ (e.g. .../urban27-dongtan_data/urban27-dongtan, with
//                      image/stereo_{left,right}/ symlinked or copied in from the *_img archive)
//   output_traj_path   Where to write the estimated trajectory, in the same 20-column space-separated
//                      format ov_msckf/scripts/gps_eval/record_traj.py writes (t tx ty tz qx qy qz qw
//                      + 6 orientation-cov + 6 position-cov upper-triangle values), so
//                      ov_eval::Loader::load_data() gets full pose+NEES support, not just poses.
//   max_relative_secs  Optional. If >0, stop feeding once a measurement's timestamp exceeds
//                      (first IMU timestamp + max_relative_secs). For quick smoke tests only.
// AI Generated
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "utils/colors.h"
#include "utils/gps_conv.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

namespace {

/// One row of sensor_data/xsens_imu.csv that we care about (gyro + accel columns only).
struct ImuRow {
  double timestamp; // seconds
  Eigen::Vector3d wm, am;
};

/// One row of sensor_data/gps.csv.
struct GpsRow {
  double timestamp; // seconds
  double lat, lon, alt;
  Eigen::Matrix3d cov;
};

/// One row of sensor_data/stereo_stamp.csv. Images are loaded lazily by filename when the merge
/// loop actually reaches this frame -- the KAIST image set is tens of GB, never preload it.
struct CamRow {
  double timestamp; // seconds
  std::string path_left, path_right;
};

/// KAIST timestamps are nanosecond Unix epoch integers. We convert with a single, identical
/// formula everywhere it is needed (also duplicated in the ground-truth converter script) --
/// double has ~15-17 significant decimal digits, and at this magnitude (~1.5e9) that leaves
/// sub-microsecond quantization noise, four to five orders of magnitude below the smallest real
/// gap between distinct samples (IMU period ~10ms) -- so it never affects ordering or association,
/// as long as both sides of any comparison used the same formula.
double ns_to_sec(long long stamp_ns) { return static_cast<double>(stamp_ns) * 1e-9; }

/// Splits off the leading comma-separated integer timestamp WITHOUT ever routing it through a
/// double: KAIST nanosecond-epoch timestamps (e.g. 1544582648748380970) have up to 19 significant
/// digits, but a double's 52-bit mantissa is only exact up to 2^53 (~16 digits) -- parsing the
/// timestamp token with `stream >> double` first (as the rest of the line is parsed) would silently
/// round it by up to a few hundred nanoseconds *before* ns_to_sec() ever runs, defeating the point
/// of ns_to_sec() using a single exact formula. Every other field on the line is small enough that
/// double parsing is exact for our purposes, so only this leading column needs special handling.
bool parse_leading_stamp_ns(const std::string &line, long long &stamp_ns, std::string &rest) {
  size_t comma = line.find(',');
  if (comma == std::string::npos)
    return false;
  try {
    stamp_ns = std::stoll(line.substr(0, comma));
  } catch (...) {
    return false;
  }
  rest = line.substr(comma + 1);
  return true;
}

/// Parses a comma-separated line into a vector of doubles (KAIST CSVs have no header rows).
bool parse_csv_line(const std::string &line, std::vector<double> &out) {
  out.clear();
  if (line.empty())
    return false;
  std::string tmp = line;
  std::replace(tmp.begin(), tmp.end(), ',', ' ');
  std::istringstream ss(tmp);
  double v;
  while (ss >> v)
    out.push_back(v);
  return !out.empty();
}

std::vector<ImuRow> load_imu(const std::string &path) {
  std::vector<ImuRow> rows;
  std::ifstream f(path);
  if (!f.is_open()) {
    PRINT_ERROR(RED "[KAIST]: unable to open IMU file %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line, rest;
  std::vector<double> v;
  long long stamp_ns;
  double last_ts = -std::numeric_limits<double>::infinity();
  size_t n_dropped = 0;
  while (std::getline(f, line)) {
    // Columns after the timestamp: qx,qy,qz,qw, ex,ey,ez, gx,gy,gz(rad/s), ax,ay,az(m/s^2), mx,my,mz
    // -- 16 total (17 with the timestamp). Column layout confirmed against the reference converter
    // https://github.com/rpng/kaist2bag (imu_converter.cpp's fscanf format string). Quaternion/
    // euler/mag are unused by OpenVINS.
    if (!parse_leading_stamp_ns(line, stamp_ns, rest))
      continue;
    if (!parse_csv_line(rest, v) || v.size() < 16)
      continue;
    ImuRow row;
    row.timestamp = ns_to_sec(stamp_ns);
    row.wm << v[7], v[8], v[9];
    row.am << v[10], v[11], v[12];
    // Propagator::select_imu_readings() assumes strictly increasing IMU timestamps; drop (not
    // reorder) any non-increasing sample rather than silently violate that.
    if (row.timestamp <= last_ts) {
      n_dropped++;
      continue;
    }
    last_ts = row.timestamp;
    rows.push_back(row);
  }
  if (n_dropped > 0) {
    PRINT_WARNING(YELLOW "[KAIST]: dropped %zu non-increasing/duplicate IMU timestamp(s)\n" RESET, n_dropped);
  }
  return rows;
}

std::vector<GpsRow> load_gps(const std::string &path) {
  std::vector<GpsRow> rows;
  std::ifstream f(path);
  if (!f.is_open()) {
    PRINT_ERROR(RED "[KAIST]: unable to open GPS file %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line, rest;
  std::vector<double> v;
  long long stamp_ns;
  double last_ts = -std::numeric_limits<double>::infinity();
  size_t n_dropped = 0;
  while (std::getline(f, line)) {
    // Columns after the timestamp: lat_deg, lon_deg, alt_m, cov[9] (row-major 3x3) -- 12 total
    // (13 with the timestamp). Confirmed against kaist2bag's gps_converter.cpp, which passes
    // position_covariance straight through.
    if (!parse_leading_stamp_ns(line, stamp_ns, rest))
      continue;
    if (!parse_csv_line(rest, v) || v.size() < 12)
      continue;
    GpsRow row;
    row.timestamp = ns_to_sec(stamp_ns);
    row.lat = v[0];
    row.lon = v[1];
    row.alt = v[2];
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        row.cov(r, c) = v[3 + 3 * r + c];
    if (row.timestamp <= last_ts) {
      n_dropped++;
      continue;
    }
    last_ts = row.timestamp;
    rows.push_back(row);
  }
  if (n_dropped > 0) {
    PRINT_WARNING(YELLOW "[KAIST]: dropped %zu non-increasing/duplicate GPS timestamp(s)\n" RESET, n_dropped);
  }
  return rows;
}

std::vector<CamRow> load_cam(const std::string &stamp_path, const std::string &left_dir, const std::string &right_dir) {
  std::vector<CamRow> rows;
  std::ifstream f(stamp_path);
  if (!f.is_open()) {
    PRINT_ERROR(RED "[KAIST]: unable to open stereo stamp file %s\n" RESET, stamp_path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  double last_ts = -std::numeric_limits<double>::infinity();
  size_t n_dropped = 0;
  while (std::getline(f, line)) {
    // Strip any trailing whitespace/CR so the timestamp string used to build the image filename
    // below is byte-exact -- reusing a line with stray characters would silently fail cv::imread.
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
      line.pop_back();
    if (line.empty())
      continue;
    long long stamp_ns = std::stoll(line);
    double ts = ns_to_sec(stamp_ns);
    if (ts <= last_ts) {
      n_dropped++;
      continue;
    }
    last_ts = ts;
    CamRow row;
    row.timestamp = ts;
    row.path_left = left_dir + "/" + line + ".png";
    row.path_right = right_dir + "/" + line + ".png";
    rows.push_back(row);
  }
  if (n_dropped > 0) {
    PRINT_WARNING(YELLOW "[KAIST]: dropped %zu non-increasing/duplicate camera timestamp(s)\n" RESET, n_dropped);
  }
  return rows;
}

} // namespace

int main(int argc, char **argv) {

  if (argc < 4) {
    PRINT_ERROR(RED "Usage: run_serial_kaist <config_path> <kaist_dataset_dir> <output_traj_path> [max_relative_secs]\n" RESET);
    return EXIT_FAILURE;
  }
  std::string config_path = argv[1];
  std::string dataset_dir = argv[2];
  std::string output_path = argv[3];
  double max_relative_secs = (argc >= 5) ? std::atof(argv[4]) : -1.0;

  // ---- Load config. No ROS anywhere in this file, so no set_node_handler() call at all (unlike
  // test_sim_meas.cpp's #if ROS_AVAILABLE==1 guarded call) -- this builds identically whether or not
  // ROS is present, since it never touches a ROS API. ----
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);

  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);

  // Determinism, see the file header comment for why all four of these are required.
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  params.init_options.init_dyn_mle_max_threads = 1;

  // gps_datum is not part of GPSOptions (it is a runtime datum choice, not an estimator parameter --
  // see the comment on ov_core::GpsData) -- the ROS visualizers parse this same key locally in
  // setup_subscribers(); we do the same here since there is no ROS layer to do it for us.
  std::vector<double> gps_datum;
  parser->parse_config("gps_datum", gps_datum, false);

  if (!parser->successful()) {
    PRINT_ERROR(RED "[KAIST]: unable to parse all required config parameters, please fix %s\n" RESET, config_path.c_str());
    return EXIT_FAILURE;
  }

  auto sys = std::make_shared<VioManager>(params);

  // ---- Load all three sensor streams. IMU/GPS rows are tiny (row data only); camera rows are just
  // timestamps + filenames -- the actual pixel data is read lazily in the merge loop below. ----
  std::string sensor_dir = dataset_dir + "/sensor_data";
  std::vector<ImuRow> imu_rows = load_imu(sensor_dir + "/xsens_imu.csv");
  std::vector<GpsRow> gps_rows = params.gps.enabled ? load_gps(sensor_dir + "/gps.csv") : std::vector<GpsRow>();
  std::vector<CamRow> cam_rows =
      load_cam(sensor_dir + "/stereo_stamp.csv", dataset_dir + "/image/stereo_left", dataset_dir + "/image/stereo_right");

  if (imu_rows.empty() || cam_rows.empty()) {
    PRINT_ERROR(RED "[KAIST]: no IMU or no camera data found under %s\n" RESET, dataset_dir.c_str());
    return EXIT_FAILURE;
  }

  // GPS datum: explicitly pinned via config is strongly preferred over the first-fix fallback below.
  // Reason: outage-simulation variants (gps_dropout_start_secs/end_secs) skip different leading fixes
  // depending on the dropout window, so an auto-picked "first fix" datum would silently differ between
  // variants -- putting each one's GPS residuals, and the estimated E-to-G yaw/position, in a different
  // ENU frame. ov_eval's posyaw/se3 alignment corrects estimate-vs-ground-truth, not this -- it does
  // NOT make a datum mismatch between variants moot. Pin gps_datum in the config when comparing runs.
  double datum_lat = 0, datum_lon = 0, datum_alt = 0;
  if (gps_datum.size() == 3) {
    datum_lat = gps_datum.at(0);
    datum_lon = gps_datum.at(1);
    datum_alt = gps_datum.at(2);
    PRINT_INFO(GREEN "[KAIST]: using pinned GPS datum lat=%.8f, lon=%.8f, alt=%.3f\n" RESET, datum_lat, datum_lon, datum_alt);
  } else if (!gps_rows.empty()) {
    datum_lat = gps_rows.front().lat;
    datum_lon = gps_rows.front().lon;
    datum_alt = gps_rows.front().alt;
    PRINT_WARNING(YELLOW "[KAIST]: gps_datum not pinned in config, using first fix [%.8f, %.8f, %.3f] -- pin it "
                         "explicitly (same key) before comparing multiple variants against each other\n" RESET,
                  datum_lat, datum_lon, datum_alt);
  }

  // max_relative_secs truncation MUST happen before the "trim trailing camera" step below, not
  // after: IMU (100Hz) and camera (10Hz) samples never share exact timestamps, so truncating both
  // independently to the same wall-clock cutoff can leave the last surviving camera frame's
  // timestamp past the last surviving IMU sample's -- reintroducing exactly the "camera outruns
  // IMU" problem the trim below exists to prevent, but at the new truncated boundary instead of the
  // dataset's real end. Doing the trim once, after any truncation, handles both cases uniformly.
  if (max_relative_secs > 0) {
    double cutoff = imu_rows.front().timestamp + max_relative_secs;
    auto imu_end = std::upper_bound(imu_rows.begin(), imu_rows.end(), cutoff,
                                    [](double t, const ImuRow &r) { return t < r.timestamp; });
    imu_rows.erase(imu_end, imu_rows.end());
    auto gps_end = std::upper_bound(gps_rows.begin(), gps_rows.end(), cutoff,
                                    [](double t, const GpsRow &r) { return t < r.timestamp; });
    gps_rows.erase(gps_end, gps_rows.end());
    auto cam_end = std::upper_bound(cam_rows.begin(), cam_rows.end(), cutoff,
                                    [](double t, const CamRow &r) { return t < r.timestamp; });
    cam_rows.erase(cam_end, cam_rows.end());
    PRINT_INFO(CYAN "[KAIST]: smoke-test truncation to first %.1fs -> %zu IMU, %zu GPS, %zu camera\n" RESET, max_relative_secs,
               imu_rows.size(), gps_rows.size(), cam_rows.size());
    if (imu_rows.empty()) {
      PRINT_ERROR(RED "[KAIST]: max_relative_secs=%.1f left zero IMU samples, nothing to do\n" RESET, max_relative_secs);
      return EXIT_FAILURE;
    }
  }

  // Trailing camera frames past the last (possibly just-truncated) IMU sample cannot be propagated
  // to correctly: Propagator::select_imu_readings() would silently EXTRAPOLATE off the last two IMU
  // samples (PRINT_WARNING only, does not fail) instead of the normal bracketing interpolation.
  // Truncate rather than accept degraded propagation for the run's tail.
  // (The mirror-image case -- camera frames before any IMU -- is NOT a problem and needs no
  // handling: do_feature_propagate_update() only runs once is_initialized_vio, which itself
  // requires an IMU window to exist first.)
  double last_imu_t = imu_rows.back().timestamp;
  size_t n_cam_total = cam_rows.size();
  while (!cam_rows.empty() && cam_rows.back().timestamp > last_imu_t) {
    cam_rows.pop_back();
  }
  if (cam_rows.size() != n_cam_total) {
    PRINT_WARNING(YELLOW "[KAIST]: dropped %zu trailing camera frame(s) past the last IMU sample (t=%.6f)\n" RESET,
                  n_cam_total - cam_rows.size(), last_imu_t);
  }

  PRINT_INFO(GREEN "[KAIST]: loaded %zu IMU, %zu GPS, %zu camera measurements from %s\n" RESET, imu_rows.size(), gps_rows.size(),
             cam_rows.size(), dataset_dir.c_str());

  std::ofstream out(output_path);
  if (!out.is_open()) {
    PRINT_ERROR(RED "[KAIST]: unable to open output file %s\n" RESET, output_path.c_str());
    return EXIT_FAILURE;
  }
  out << "# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33\n";
  out << std::setprecision(9) << std::fixed;

  // ---- Deterministic 3-way merge: always feed whichever stream's next timestamp is smallest.
  // Tie-break at equal timestamps: IMU, then GPS, then Camera.
  //  - IMU-before-camera is a HARD correctness requirement: feed_measurement_camera() /
  //    track_image_and_update() does not buffer or wait for IMU, it assumes everything up to that
  //    timestamp was already fed (see the file header comment).
  //  - GPS-vs-camera tie order does not matter: feed_measurement_gps() only enqueues (thread-safe),
  //    and the actual fusion/draining happens lazily by timestamp inside the next camera call
  //    (VioManager::drain_gps_queue()), regardless of which was enqueued first at an exact tie.
  //
  // IMU LOOKAHEAD -- required, not an optimization. Feeding in *strict* timestamp order is not
  // sufficient: to propagate to time1, Propagator::select_imu_readings() needs an IMU sample
  // STRICTLY AFTER time1 so its "END OF THE INTEGRATION PERIOD" case can interpolate the final
  // partial step and close the interval (see Propagator.cpp). With only samples <= time1, prop_data
  // ends short, dt_summed < (time1 - time0), and propagate_and_clone()'s
  //   assert(std::abs((time1 - time0) - dt_summed) < 1e-4)
  // aborts the process. This bites the GPS path in particular: drain_gps_queue() propagates to each
  // fix's own timestamp, and on urban28 1.2% of fixes (157 / 13508) have no IMU sample between the
  // fix and the next camera frame -- so under strict ordering nothing after the fix has been fed yet
  // when it is drained. (urban27 hit the same pattern but its leftover interval happened to stay
  // under the 1e-4 tolerance, which is why it only warned instead of crashing.)
  //
  // So before processing any camera/GPS event we feed IMU until it is covered with margin. This is
  // safe and faithful, not a fudge: feed_measurement_imu() only appends to the propagator/ZUPT/
  // initializer buffers (it never propagates the state), and a live system behaves exactly this way
  // -- IMU arrives continuously and is buffered well ahead of the camera frame being processed.
  // Ordering is still deterministic and still IMU-before-camera; the window is just one sample wider.
  const double kImuLookahead = 0.05; // seconds of IMU to keep fed past the next event
  std::shared_ptr<State> state_ptr = sys->get_state();
  size_t i_imu = 0, i_gps = 0, i_cam = 0;
  size_t n_frames_written = 0, n_frames_processed = 0, n_events_dropped = 0;
  double last_imu_fed = -std::numeric_limits<double>::infinity();
  while (i_gps < gps_rows.size() || i_cam < cam_rows.size()) {
    double t_gps = (i_gps < gps_rows.size()) ? gps_rows[i_gps].timestamp : std::numeric_limits<double>::infinity();
    double t_cam = (i_cam < cam_rows.size()) ? cam_rows[i_cam].timestamp : std::numeric_limits<double>::infinity();
    double t_event = std::min(t_gps, t_cam);

    // Keep IMU fed until it covers this event plus the camera-IMU time offset (which the filter
    // calibrates online, so read it live) and the margin above.
    double t_off = state_ptr->_calib_dt_CAMtoIMU->value()(0);
    double t_required = t_event + t_off + kImuLookahead;
    if (last_imu_fed <= t_required) {
      if (i_imu < imu_rows.size()) {
        const ImuRow &row = imu_rows[i_imu++];
        ov_core::ImuData msg;
        msg.timestamp = row.timestamp;
        msg.wm = row.wm;
        msg.am = row.am;
        sys->feed_measurement_imu(msg);
        last_imu_fed = row.timestamp;
        continue;
      }
      // IMU exhausted and it does not reach past this event -- every remaining event is even later,
      // so none of them can be bracketed either. Stop rather than propagate off the end of the data.
      n_events_dropped = (gps_rows.size() - i_gps) + (cam_rows.size() - i_cam);
      break;
    }

    if (t_gps <= t_cam) {
      const GpsRow &row = gps_rows[i_gps++];
      ov_core::GpsData msg;
      msg.timestamp = row.timestamp;
      msg.meas_ENU = ov_core::gps::lla2enu(row.lat, row.lon, row.alt, datum_lat, datum_lon, datum_alt);
      msg.cov = row.cov;
      sys->feed_measurement_gps(msg);

    } else {
      const CamRow &row = cam_rows[i_cam++];
      // Raw pixel data, IMREAD_UNCHANGED so no implicit conversion happens -- the PNGs on disk are a
      // single 8-bit channel already (that's true whether the content is real grayscale or, as here,
      // Bayer mosaic; PNG has no way to flag which).
      cv::Mat raw_left = cv::imread(row.path_left, cv::IMREAD_UNCHANGED);
      cv::Mat raw_right = cv::imread(row.path_right, cv::IMREAD_UNCHANGED);
      if (raw_left.empty() || raw_right.empty()) {
        PRINT_WARNING(YELLOW "[KAIST]: unable to read image pair at t=%.6f (%s / %s), skipping\n" RESET, row.timestamp,
                      row.path_left.c_str(), row.path_right.c_str());
        continue;
      }
      // These are Bayer BGGR8 mosaic, NOT true grayscale -- confirmed against the reference converter
      // https://github.com/rpng/kaist2bag (stereo_converter.cpp tags every frame BAYER_BGGR8).
      // TrackKLT requires CV_8UC1 (cv::equalizeHist/CLAHE assert single-channel 8-bit) and, more
      // importantly, feeding the raw mosaic straight through would corrupt tracking with the 2x2
      // Bayer checkerboard's high-frequency artifact. Demosaic before histogram equalization
      // (config uses histogram_method: HISTOGRAM) -- same order the pipeline always assumes.
      //
      // NOTE the code letters do NOT match naively: OpenCV's cvtColor Bayer codes name the pattern
      // as read from the SECOND row of the image (its own documented convention), one row offset
      // from how ROS/cv_bridge names encodings (which describe the tile starting at pixel (0,0)).
      // Working the 2x2 tile math through confirms the standard cv_bridge crosswalk: ROS
      // bayer_bggr8 <-> OpenCV COLOR_BayerRG2*, NOT COLOR_BayerBG2* -- the "BG"/"RG" letters swap
      // between the two naming conventions. Using COLOR_BayerBG2GRAY here would silently demosaic
      // with the wrong pattern (no crash, just corrupted color/edges feeding straight into KLT).
      cv::Mat gray_left, gray_right;
      cv::cvtColor(raw_left, gray_left, cv::COLOR_BayerRG2GRAY);
      cv::cvtColor(raw_right, gray_right, cv::COLOR_BayerRG2GRAY);

      ov_core::CameraData msg;
      msg.timestamp = row.timestamp;
      msg.sensor_ids = {0, 1};
      msg.images = {gray_left, gray_right};
      msg.masks = {cv::Mat::zeros(gray_left.rows, gray_left.cols, CV_8UC1), cv::Mat::zeros(gray_right.rows, gray_right.cols, CV_8UC1)};
      sys->feed_measurement_camera(msg);
      n_frames_processed++;

      if (sys->initialized()) {
        std::shared_ptr<State> state = sys->get_state();
        // Same convention as ROS2Visualizer::publish_state(): timestamp shifted into the IMU clock,
        // and the JPL quaternion's (x,y,z,w) written directly -- "since we use JPL we have an
        // implicit conversion to Hamilton when we publish" (verbatim repo comment: a JPL q_GtoI has
        // the same (x,y,z,w) as the Hamilton q_ItoG, no matrix conversion needed).
        double t_out = state->_timestamp + state->_calib_dt_CAMtoIMU->value()(0);
        Eigen::Vector4d q = state->_imu->quat();
        Eigen::Vector3d p = state->_imu->pos();
        // Orientation-then-position order (opposite of the ROS PoseWithCovariance convention
        // ROS2Visualizer::publish_state() uses) to match what ov_eval::Loader::load_data() expects
        // directly: columns 8-13 are the orientation-covariance upper triangle (Pr), 14-19 the
        // position-covariance upper triangle (Pt). Writing it in this order ourselves means we don't
        // need record_traj.py's post-hoc swap (that swap exists only because it receives the ROS
        // message, which is fixed in [position|orientation] order).
        std::vector<std::shared_ptr<ov_type::Type>> statevars = {state->_imu->pose()->q(), state->_imu->pose()->p()};
        Eigen::Matrix<double, 6, 6> cov = StateHelper::get_marginal_covariance(state, statevars);

        out << t_out << " " << p(0) << " " << p(1) << " " << p(2) << " " << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << " "
            << cov(0, 0) << " " << cov(0, 1) << " " << cov(0, 2) << " " << cov(1, 1) << " " << cov(1, 2) << " " << cov(2, 2) << " "
            << cov(3, 3) << " " << cov(3, 4) << " " << cov(3, 5) << " " << cov(4, 4) << " " << cov(4, 5) << " " << cov(5, 5) << "\n";
        n_frames_written++;
      }
    }
  }

  out.close();
  if (n_events_dropped > 0) {
    PRINT_WARNING(YELLOW "[KAIST]: dropped %zu trailing event(s) that the IMU stream could not bracket\n" RESET, n_events_dropped);
  }
  PRINT_INFO(GREEN "[KAIST]: done. processed %zu camera frames, wrote %zu poses (post-init) to %s\n" RESET, n_frames_processed,
             n_frames_written, output_path.c_str());
  return EXIT_SUCCESS;
}
