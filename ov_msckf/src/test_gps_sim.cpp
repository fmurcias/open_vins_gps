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

// Full-simulation integration test for the GPS fusion (see design doc, test_gps_sim
// requirements): (a) the E-to-G transform converges to the truth, (b) ATE with GPS beats ATE without,
// (c) NEES stays within a sane band (consistency), (d) the filter survives IMU timestamp jitter --
// this is the whole reason the design avoids interpolation in the first place, and (e) injected
// outlier fixes are rejected by the chi2 gate without corrupting the state.
//
// This is a self-contained executable (no YAML config needed): VioManagerOptions is built directly in
// C++, pointing at the udel_gore.txt trajectory shipped in ov_data/sim via the OV_DATA_DIR compile
// definition set in CMakeLists.

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "cam/CamRadtan.h"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "sim/Simulator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Vec.h"
#include "update/UpdaterGPS.h"
#include "utils/colors.h"
#include "utils/print.h"

#ifndef OV_DATA_DIR
#define OV_DATA_DIR "../ov_data"
#endif

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

static VioManagerOptions build_params(bool gps_enabled, int seed) {
  VioManagerOptions params;

  // Keep the front-end small and synthetic-only so this test runs quickly
  params.state_options.num_cameras = 1;
  params.use_stereo = false;
  params.use_klt = true;
  params.use_aruco = false;
  params.num_pts = 100;
  params.state_options.max_clone_size = 8;
  params.state_options.max_slam_features = 25;
  params.dt_slam_delay = 3.0;
  params.gravity_mag = 9.81;
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;

  auto cam0 = std::make_shared<ov_core::CamRadtan>(752, 480);
  Eigen::VectorXd cam_calib(8);
  cam_calib << 458.0, 458.0, 376.0, 240.0, 0.0, 0.0, 0.0, 0.0;
  cam0->set_value(cam_calib);
  params.camera_intrinsics.insert({0, cam0});
  Eigen::VectorXd cam_extrinsic(7);
  cam_extrinsic << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0; // identity q_ItoC, zero p_IinC
  params.camera_extrinsics.insert({0, cam_extrinsic});

  // InertialInitializerOptions keeps its own separate copy of the camera calibration (normally
  // populated by print_and_load_state() via a YAML parser); mirror it here since we build everything
  // in code instead.
  params.init_options.num_cameras = 1;
  auto cam0_init = std::make_shared<ov_core::CamRadtan>(752, 480);
  cam0_init->set_value(cam_calib);
  params.init_options.camera_intrinsics.insert({0, cam0_init});
  params.init_options.camera_extrinsics.insert({0, cam_extrinsic});

  // IMU intrinsics: these are raw Eigen members normally populated by the YAML parser and are left
  // uninitialized otherwise -- must set them to the identity/no-op defaults State.cpp itself assumes.
  params.vec_dw << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0; // identity (kalibr lower-triangular column-wise)
  params.vec_da << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
  params.vec_tg.setZero();
  params.q_GYROtoIMU << 0.0, 0.0, 0.0, 1.0; // identity JPL quaternion
  params.q_ACCtoIMU << 0.0, 0.0, 0.0, 1.0;

  params.sim_traj_path = std::string(OV_DATA_DIR) + "/sim/udel_gore.txt";
  params.sim_distance_threshold = 1.0;
  params.sim_freq_cam = 10.0;
  params.sim_freq_imu = 200.0;
  params.sim_seed_state_init = 0;
  params.sim_seed_preturb = 0;
  params.sim_seed_measurements = seed;
  params.sim_do_perturbation = false;

  params.gps.enabled = gps_enabled;
  if (gps_enabled) {
    params.sim_freq_gps = 2.0;
    params.sim_gps_noise_std = 0.3;
    params.sim_gps_true_yaw = 0.4;
    params.sim_gps_true_pos_EinG = Eigen::Vector3d(2.0, -1.0, 0.0);
    params.gps.chi2_multipler = 5.0;
    params.gps.noise_floor = 0.15;
    params.gps.noise_multiplier = 1.0;
    params.gps.toff = 0.0;
    params.gps.leverarm_prior = Eigen::Vector3d(0.05, 0.0, -0.02);
    params.gps.leverarm_prior_std = 0.05;
    params.gps.do_calib_leverarm = false;
    params.gps.init_min_meas = 10;
    params.gps.init_min_baseline = 3.0;
    params.gps.init_min_excursion = 0.5;
    params.gps.max_late_dt = 1.0;
  }

  return params;
}

struct SimResult {
  bool ok = false; // false if we never even got a single valid measurement sample
  double ate = -1.0; // RMSE position error (m), sampled at every camera update after VIO init
  double max_err = -1.0;
  bool has_nan = false;
  bool transform_initialized = false;
  double yaw_err_deg = -1.0;
  double pos_EinG_err = -1.0;
  int gps_accepted = 0;
  int gps_rejected = 0;
  std::vector<double> nees_samples; // 3-dof position NEES, sampled after GPS init
};

// Runs one full simulation. If jitter_imu, IMU timestamps are perturbed by up to +-40% of the nominal
// period and occasionally duplicated (this is exactly the scenario the no-interpolation design targets).
// If n_outliers>0, that many GPS fixes get a gross (30m) offset injected before being fed to the filter.
static SimResult run_sim(VioManagerOptions params, double max_duration, bool jitter_imu, int n_outliers) {

  SimResult result;
  Simulator sim(params);
  auto sys = std::make_shared<VioManager>(params);

  double next_imu_time = sim.current_timestamp() + 1.0 / params.sim_freq_imu;
  Eigen::Matrix<double, 17, 1> imustate;
  if (!sim.get_state(next_imu_time, imustate)) {
    return result;
  }
  imustate(0, 0) -= sim.get_true_parameters().calib_camimu_dt;
  sys->initialize_with_gt(imustate);

  double start_time = sim.current_timestamp();
  double buffer_timecam = -1;
  std::vector<int> buffer_camids;
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> buffer_feats;

  std::mt19937 jitter_gen(12345);
  std::uniform_real_distribution<double> jitter_dist(-0.4, 0.4);
  std::bernoulli_distribution dup_dist(0.03);

  int gps_fix_count = 0;
  int outliers_injected = 0;

  double sum_sq_err = 0.0;
  int n_samples = 0;

  auto sample_error = [&]() {
    if (!sys->initialized())
      return;
    Eigen::Matrix<double, 17, 1> gt;
    if (!sim.get_state(sys->get_state()->_timestamp, gt))
      return;
    Eigen::Vector3d p_true = gt.block(5, 0, 3, 1);
    Eigen::Vector3d p_est = sys->get_state()->_imu->pos();
    if (!p_est.allFinite()) {
      result.has_nan = true;
      return;
    }
    double err = (p_est - p_true).norm();
    sum_sq_err += err * err;
    n_samples++;
    result.max_err = std::max(result.max_err, err);

    // NEES (3-dof position), only meaningful once GPS has actually corrected the global gauge
    if (sys->get_updater_gps() != nullptr && sys->get_updater_gps()->transform_initialized()) {
      Eigen::MatrixXd P = StateHelper::get_marginal_covariance(sys->get_state(), {sys->get_state()->_imu->p()});
      Eigen::Vector3d e = p_est - p_true;
      double nees = e.transpose() * P.llt().solve(e);
      if (std::isfinite(nees)) {
        result.nees_samples.push_back(nees);
      }
    }
  };

  while (sim.ok()) {

    if (sim.current_timestamp() - start_time > max_duration) {
      break;
    }
    if (result.has_nan) {
      break;
    }

    // IMU
    double time_imu;
    Eigen::Vector3d wm, am;
    bool hasimu = sim.get_next_imu(time_imu, wm, am);
    if (hasimu) {
      ov_core::ImuData message_imu;
      message_imu.timestamp = time_imu;
      message_imu.wm = wm;
      message_imu.am = am;
      if (jitter_imu) {
        double dt_nominal = 1.0 / params.sim_freq_imu;
        message_imu.timestamp = time_imu + jitter_dist(jitter_gen) * dt_nominal;
        if (dup_dist(jitter_gen)) {
          sys->feed_measurement_imu(message_imu); // occasional duplicate reading at the same/close timestamp
        }
      }
      sys->feed_measurement_imu(message_imu);
    }

    // GPS
    double time_gps;
    Eigen::Vector3d meas_ENU;
    Eigen::Matrix3d cov_gps;
    bool hasgps = sim.get_next_gps(time_gps, meas_ENU, cov_gps);
    if (hasgps) {
      ov_core::GpsData message_gps;
      message_gps.timestamp = time_gps;
      message_gps.meas_ENU = meas_ENU;
      message_gps.cov = cov_gps;
      gps_fix_count++;
      // Only start injecting once the transform is already live, so this cleanly exercises the
      // per-fix chi2 gate in try_update() rather than the (separately tested, see test_gps_jacobians)
      // Horn-alignment outlier rejection used during delayed init.
      bool transform_live = sys->get_updater_gps() != nullptr && sys->get_updater_gps()->transform_initialized();
      if (n_outliers > 0 && outliers_injected < n_outliers && transform_live && gps_fix_count % 6 == 0) {
        message_gps.meas_ENU += Eigen::Vector3d(30.0, -30.0, 0.0);
        outliers_injected++;
      }
      sys->feed_measurement_gps(message_gps);
    }

    // CAM
    double time_cam;
    std::vector<int> camids;
    std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
    bool hascam = sim.get_next_cam(time_cam, camids, feats);
    if (hascam) {
      if (buffer_timecam != -1) {
        sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);
        sample_error();
      }
      buffer_timecam = time_cam;
      buffer_camids = camids;
      buffer_feats = feats;
    }
  }

  result.ok = (n_samples > 0);
  result.ate = (n_samples > 0) ? std::sqrt(sum_sq_err / n_samples) : -1.0;
  if (sys->get_updater_gps() != nullptr) {
    result.gps_accepted = sys->get_updater_gps()->stat_num_accepted;
    result.gps_rejected = sys->get_updater_gps()->stat_num_rejected;
    result.transform_initialized = sys->get_updater_gps()->transform_initialized();
    if (result.transform_initialized) {
      double psi_est = sys->get_state()->_gps_yaw_EtoG->value()(0);
      Eigen::Vector3d p_est = sys->get_state()->_gps_pos_EinG->value();
      double yaw_err = std::abs(psi_est - params.sim_gps_true_yaw);
      result.yaw_err_deg = yaw_err * 180.0 / M_PI;
      result.pos_EinG_err = (p_est - params.sim_gps_true_pos_EinG).norm();
    }
  }
  return result;
}

int main() {

  bool all_passed = true;
  // The ATE-improvement criterion needs a long enough run for pure-VIO drift to actually show up;
  // the jitter/outlier robustness checks don't need that, so they use a shorter duration to stay fast.
  double max_duration_drift = 150.0;
  double max_duration_robustness = 60.0;

  // ---- (a)+(b): GPS transform convergence, and ATE with GPS vs without ----
  PRINT_INFO("[TEST]: running baseline (no GPS)...\n");
  SimResult no_gps = run_sim(build_params(false, 0), max_duration_drift, false, 0);
  PRINT_INFO("[TEST]: running with GPS...\n");
  SimResult with_gps = run_sim(build_params(true, 0), max_duration_drift, false, 0);

  if (!no_gps.ok || !with_gps.ok) {
    PRINT_ERROR(RED "[TEST]: one of the baseline runs produced no samples (ok_no_gps=%d, ok_with_gps=%d)\n" RESET, no_gps.ok,
                with_gps.ok);
    all_passed = false;
  }
  PRINT_INFO("[TEST]: ATE no-gps=%.3fm, ATE with-gps=%.3fm (max %.3fm / %.3fm)\n", no_gps.ate, with_gps.ate, no_gps.max_err,
             with_gps.max_err);

  if (!with_gps.transform_initialized) {
    PRINT_ERROR(RED "[TEST]: E-to-G transform never initialized in the GPS run\n" RESET);
    all_passed = false;
  } else {
    PRINT_INFO("[TEST]: transform errors: yaw=%.3f deg, p_EinG=%.3fm\n", with_gps.yaw_err_deg, with_gps.pos_EinG_err);
    if (with_gps.yaw_err_deg > 0.5) {
      PRINT_ERROR(RED "[TEST]: yaw error %.3f deg exceeds 0.5 deg tolerance\n" RESET, with_gps.yaw_err_deg);
      all_passed = false;
    }
    if (with_gps.pos_EinG_err > 0.1) {
      PRINT_ERROR(RED "[TEST]: p_EinG error %.3fm exceeds 0.1m tolerance\n" RESET, with_gps.pos_EinG_err);
      all_passed = false;
    }
  }

  if (with_gps.ok && no_gps.ok && with_gps.ate > no_gps.ate) {
    PRINT_ERROR(RED "[TEST]: ATE with GPS (%.3fm) did not improve over ATE without GPS (%.3fm)\n" RESET, with_gps.ate, no_gps.ate);
    all_passed = false;
  }

  // ---- (c): NEES within a (generous, single-run) consistency band ----
  if (with_gps.nees_samples.empty()) {
    PRINT_ERROR(RED "[TEST]: no NEES samples were collected after GPS init\n" RESET);
    all_passed = false;
  } else {
    double mean_nees = 0.0;
    for (double v : with_gps.nees_samples)
      mean_nees += v;
    mean_nees /= (double)with_gps.nees_samples.size();
    // 3-dof NEES should average near 3 for a consistent filter; a single run is noisy, so use a generous band
    PRINT_INFO("[TEST]: mean position NEES = %.2f (over %d samples, expect ~3 for a consistent filter)\n", mean_nees,
               (int)with_gps.nees_samples.size());
    if (mean_nees > 30.0) {
      PRINT_ERROR(RED "[TEST]: mean NEES %.2f is wildly inconsistent (>30, expected ~3)\n" RESET, mean_nees);
      all_passed = false;
    }
  }

  // ---- (d): IMU timestamp jitter (+-40% of nominal period, with occasional duplicates) must not diverge ----
  PRINT_INFO("[TEST]: running with GPS + IMU jitter...\n");
  SimResult jitter = run_sim(build_params(true, 1), max_duration_robustness, true, 0);
  if (jitter.has_nan) {
    PRINT_ERROR(RED "[TEST]: filter produced non-finite state under IMU jitter\n" RESET);
    all_passed = false;
  }
  if (!jitter.ok) {
    PRINT_ERROR(RED "[TEST]: jitter run produced no samples\n" RESET);
    all_passed = false;
  } else {
    PRINT_INFO("[TEST]: jitter run ATE=%.3fm (max %.3fm)\n", jitter.ate, jitter.max_err);
    // "did not diverge": bounded absolute error, not a statistical match to the clean run
    if (jitter.max_err > 25.0) {
      PRINT_ERROR(RED "[TEST]: filter diverged under IMU jitter (max error %.3fm > 25m)\n" RESET, jitter.max_err);
      all_passed = false;
    }
  }

  // ---- (e): injected outlier fixes must be rejected by chi2, without corrupting the state ----
  PRINT_INFO("[TEST]: running with GPS + injected outlier fixes...\n");
  SimResult outliers = run_sim(build_params(true, 2), max_duration_robustness, false, 6);
  PRINT_INFO("[TEST]: outlier run: accepted=%d, rejected=%d, ATE=%.3fm\n", outliers.gps_accepted, outliers.gps_rejected, outliers.ate);
  if (outliers.gps_rejected < 5) { // we inject 6 gross (30m) outliers once the transform is live; expect nearly all rejected
    PRINT_ERROR(RED "[TEST]: too few injected gross outlier fixes were rejected by the chi2 gate (%d/6)\n" RESET, outliers.gps_rejected);
    all_passed = false;
  }
  if (!outliers.ok || outliers.has_nan || outliers.max_err > 25.0) {
    PRINT_ERROR(RED "[TEST]: state appears corrupted after injected outliers (ok=%d, nan=%d, max_err=%.3f)\n" RESET, outliers.ok,
                outliers.has_nan, outliers.max_err);
    all_passed = false;
  }

  if (!all_passed) {
    PRINT_ERROR(RED "[TEST]: test_gps_sim FAILED\n" RESET);
    return EXIT_FAILURE;
  }
  PRINT_INFO(GREEN "[TEST]: test_gps_sim PASSED\n" RESET);
  return EXIT_SUCCESS;
}
