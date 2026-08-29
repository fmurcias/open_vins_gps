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

#include "UpdaterGPS.h"

#include <algorithm>

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterGPS::UpdaterGPS(GPSOptions options, std::shared_ptr<Propagator> prop) : _options(options), _prop(prop) {
  // Initialize the chi squared test table with confidence level 0.95 (same pattern as the other updaters)
  for (int i = 1; i < 1000; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
  // A cap below the minimum-measurement gate would evict pairs faster than try_initialize() could
  // ever see init_min_meas of them at once, silently making delayed init unreachable.
  if (_options.init_max_pairs < _options.init_min_meas) {
    PRINT_WARNING(YELLOW "[GPS]: gps_init_max_pairs (%d) is below gps_init_min_meas (%d), which would make delayed init "
                         "impossible -- raising the cap to match\n" RESET,
                  _options.init_max_pairs, _options.init_min_meas);
    _options.init_max_pairs = _options.init_min_meas;
  }
}

void UpdaterGPS::get_measurement_jacobian(std::shared_ptr<State> state, std::shared_ptr<PoseJPL> clone, const Eigen::Vector3d &meas_ENU,
                                          std::vector<std::shared_ptr<Type>> &H_order, Eigen::MatrixXd &H, Eigen::VectorXd &res) const {

  bool do_fej = state->_options.do_fej;
  bool do_leverarm = state->_options.do_calib_gps_leverarm;

  // Current best-estimate values, used to evaluate the residual h(x)
  double psi_val = state->_gps_yaw_EtoG->value()(0);
  Eigen::Vector3d p_EinG_val = state->_gps_pos_EinG->value();
  Eigen::Vector3d p_ANTinI_val = state->_calib_GPStoIMU->value();
  Eigen::Matrix3d R_GtoI_val = clone->Rot();
  Eigen::Vector3d p_IinG_val = clone->pos();

  Eigen::Matrix3d C_val = rot_z(psi_val).transpose();
  Eigen::Vector3d u_val = p_IinG_val + R_GtoI_val.transpose() * p_ANTinI_val - p_EinG_val;
  res = meas_ENU - C_val * u_val;

  // First-estimate linearization point for the Jacobian (falls back to the current value if FEJ is off)
  double psi_jac = do_fej ? state->_gps_yaw_EtoG->fej()(0) : psi_val;
  Eigen::Vector3d p_EinG_jac = do_fej ? Eigen::Vector3d(state->_gps_pos_EinG->fej()) : p_EinG_val;
  Eigen::Vector3d p_ANTinI_jac = do_fej ? Eigen::Vector3d(state->_calib_GPStoIMU->fej()) : p_ANTinI_val;
  Eigen::Matrix3d R_GtoI_jac = do_fej ? clone->Rot_fej() : R_GtoI_val;
  Eigen::Vector3d p_IinG_jac = do_fej ? clone->pos_fej() : p_IinG_val;
  Eigen::Matrix3d C_jac = rot_z(psi_jac).transpose();
  Eigen::Vector3d u_jac = p_IinG_jac + R_GtoI_jac.transpose() * p_ANTinI_jac - p_EinG_jac;
  Eigen::Vector3d e3(0.0, 0.0, 1.0);

  int sz = 10 + (do_leverarm ? 3 : 0);
  H = Eigen::MatrixXd::Zero(3, sz);
  H.block(0, 0, 3, 3) = -C_jac * R_GtoI_jac.transpose() * skew_x(p_ANTinI_jac); // d/d(delta_theta_I)
  H.block(0, 3, 3, 3) = C_jac;                                                 // d/d(p_IinG)
  H.block(0, 6, 3, 1) = -C_jac * skew_x(e3) * u_jac;                           // d/d(psi)
  H.block(0, 7, 3, 3) = -C_jac;                                                // d/d(p_EinG)
  if (do_leverarm) {
    H.block(0, 10, 3, 3) = C_jac * R_GtoI_jac.transpose(); // d/d(p_ANTinI)
  }

  H_order.clear();
  H_order.push_back(clone);
  H_order.push_back(state->_gps_yaw_EtoG);
  H_order.push_back(state->_gps_pos_EinG);
  if (do_leverarm) {
    H_order.push_back(state->_calib_GPStoIMU);
  }
}

void UpdaterGPS::handle_measurement_gap(std::shared_ptr<State> state, double meas_timestamp) {

  if (_options.gap_drift_rate <= 0.0 || _options.gap_max_sigma <= 0.0) {
    return; // disabled
  }

  // Open a recovery if this fix follows an outage. An in-progress recovery keeps its accounting
  // rather than restarting, so a run of gaps cannot sidestep the cumulative ceiling below.
  if (last_fix_time >= 0) {
    double gap = meas_timestamp - last_fix_time;
    if (gap > _options.gap_threshold_secs) {
      double sigma = std::min(_options.gap_drift_rate * gap, _options.gap_max_sigma);
      if (recovery_sigma < 0) {
        recovery_added_var = 0.0;
        recovery_attempts = 0;
        recovery_capped = false;
        PRINT_INFO(CYAN "[GPS]: %.1fs gap before t=%.3f, opening recovery (%.2f m 1-sigma per fix)\n" RESET, gap, meas_timestamp, sigma);
      }
      recovery_sigma = std::max(recovery_sigma, sigma);
      recovery_gap = std::max(recovery_gap, gap);
    }
  }

  if (recovery_sigma < 0) {
    return; // no outage pending, normal fix
  }

  // Add back the position uncertainty the filter should have accumulated while unaided but did not.
  double max_var = std::pow(_options.gap_max_sigma, 2);
  double add_var = std::min(std::pow(recovery_sigma, 2), max_var - recovery_added_var);
  if (add_var <= 0.0) {
    // Escalating past the ceiling would leave the gate so wide it no longer rejects anything, so let
    // the rejection-streak reset tear the transform down instead.
    if (!recovery_capped) {
      recovery_capped = true;
      PRINT_WARNING(YELLOW "[GPS]: recovery after %.1fs gap hit the %.1f m inflation ceiling after %d fixes without an accepted "
                           "update; leaving further recovery to the rejection-streak reset\n" RESET,
                    recovery_gap, _options.gap_max_sigma, recovery_attempts);
    }
    return;
  }

  recovery_added_var += add_var;
  recovery_attempts++;
  std::vector<std::shared_ptr<Type>> order = {state->_imu->pose()->p()};
  Eigen::MatrixXd add = add_var * Eigen::Matrix3d::Identity();
  StateHelper::inflate_covariance(state, order, add);
  PRINT_INFO(CYAN "[GPS]: inflating position covariance by %.2f m (1-sigma) at t=%.3f, recovery attempt %d after %.1fs gap "
                  "(cumulative %.2f m)\n" RESET,
             std::sqrt(add_var), meas_timestamp, recovery_attempts, recovery_gap, std::sqrt(recovery_added_var));
}

bool UpdaterGPS::try_update(std::shared_ptr<State> state, const ov_core::GpsData &meas, std::shared_ptr<PoseJPL> clone) {

  // reset_transform() can clear this mid-drain, so refuse rather than assert.
  if (!_initialized) {
    PRINT_ERROR(RED "[GPS]: try_update() called before the E-to-G transform is initialized, ignoring fix at t=%.3f\n" RESET,
                meas.timestamp);
    return false;
  }

  // This fix is being judged, so it consumes the gap regardless of the gate outcome. Only place
  // last_fix_time advances: fixes dropped before reaching here must leave it alone.
  last_fix_time = meas.timestamp;

  std::vector<std::shared_ptr<Type>> H_order;
  Eigen::MatrixXd H;
  Eigen::VectorXd res;
  get_measurement_jacobian(state, clone, meas.meas_ENU, H_order, H, res);

  // Floor and inflate the reported measurement covariance
  Eigen::Matrix3d R = meas.cov;
  for (int i = 0; i < 3; i++) {
    R(i, i) = std::max(R(i, i), std::pow(_options.noise_floor, 2));
  }
  R *= _options.noise_multiplier;

  // Chi2 gate (same pattern as UpdaterMSCKF / UpdaterZeroVelocity)
  Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, H_order);
  Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
  double chi2 = res.dot(S.llt().solve(res));

  double chi2_check;
  if (res.rows() < 1000) {
    chi2_check = chi_squared_table[(int)res.rows()];
  } else {
    boost::math::chi_squared chi_squared_dist(res.rows());
    chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
  }

  track_gate_outcome(chi2 <= _options.chi2_multipler * chi2_check, meas.timestamp);

  if (chi2 > _options.chi2_multipler * chi2_check) {
    stat_num_rejected++;
    consecutive_update_rejections++;
    if (first_rejection_time < 0) {
      first_rejection_time = meas.timestamp;
    }
    // Log the residual vector, not just the chi2 scalar: a burst that swings with heading points at
    // the lever arm, a slowly-growing one at the frozen transform or VIO drift. chi2 cannot tell them
    // apart.
    Eigen::Vector3d p_ANTinI = state->_calib_GPStoIMU->value();
    PRINT_WARNING(YELLOW "[GPS]: update rejected at t=%.3f (chi2 %.3f > %.3f) res=[%.2f, %.2f, %.2f] |res|=%.2fm "
                         "p_ANTinI=[%.3f, %.3f, %.3f] streak=%d (%.1fs)\n" RESET,
                  meas.timestamp, chi2, _options.chi2_multipler * chi2_check, res(0), res(1), res(2), res.norm(), p_ANTinI(0),
                  p_ANTinI(1), p_ANTinI(2), consecutive_update_rejections, meas.timestamp - first_rejection_time);

    // A bad transform makes every subsequent fix look like an outlier, and rejected fixes carry no
    // information, so the gate alone can never recover. Tear it down and re-run delayed init.
    // Gated on elapsed time, not fix count -- a count is meaningless without the receiver rate.
    double streak_secs = meas.timestamp - first_rejection_time;
    if (streak_secs >= _options.reset_after_rejection_secs || consecutive_update_rejections >= kMaxConsecutiveUpdateRejections) {
      PRINT_WARNING(YELLOW "[GPS]: %d consecutive rejected updates over %.1fs, resetting the E-to-G transform and re-accumulating\n" RESET,
                    consecutive_update_rejections, streak_secs);
      reset_transform(state);
    }
    return false;
  }

  StateHelper::EKFUpdate(state, H_order, H, res, R);
  stat_num_accepted++;
  consecutive_update_rejections = 0;
  first_rejection_time = -1;
  PRINT_INFO(CYAN "[GPS]: update accepted at t=%.3f (chi2 %.3f < %.3f)\n" RESET, meas.timestamp, chi2,
             _options.chi2_multipler * chi2_check);

  // An acceptance means the filter has re-latched onto GPS, so close any recovery in progress.
  if (recovery_sigma >= 0) {
    PRINT_INFO(CYAN "[GPS]: recovered from %.1fs gap at t=%.3f after %d inflated fix(es), |res|=%.2fm\n" RESET, recovery_gap,
               meas.timestamp, recovery_attempts, res.norm());
    recovery_sigma = -1;
    recovery_added_var = 0.0;
    recovery_attempts = 0;
    recovery_gap = 0.0;
    recovery_capped = false;
  }

  // Otherwise the lever arm is only ever printed on rejection, i.e. never in the healthy runs whose
  // calibration you actually want to read off.
  if (state->_options.do_calib_gps_leverarm && ++accepted_since_leverarm_log >= kLeverarmLogEvery) {
    accepted_since_leverarm_log = 0;
    Eigen::Vector3d la = state->_calib_GPStoIMU->value();
    Eigen::Matrix3d la_cov = StateHelper::get_marginal_covariance(state, {state->_calib_GPStoIMU});
    PRINT_INFO(CYAN "[GPS]: lever arm p_ANTinI = [%.3f, %.3f, %.3f] +/- [%.3f, %.3f, %.3f] (1-sigma) after %d accepted fixes\n" RESET,
               la(0), la(1), la(2), std::sqrt(la_cov(0, 0)), std::sqrt(la_cov(1, 1)), std::sqrt(la_cov(2, 2)), stat_num_accepted);
  }
  return true;
}

void UpdaterGPS::track_gate_outcome(bool accepted, double timestamp) {

  recent_outcomes.push_back(accepted);
  while (recent_outcomes.size() > kRateWindow) {
    recent_outcomes.pop_front();
  }
  if (recent_outcomes.size() < kRateWindow) {
    return;
  }

  int n_rejected = (int)std::count(recent_outcomes.begin(), recent_outcomes.end(), false);
  double frac = (double)n_rejected / (double)recent_outcomes.size();
  if (frac < kRateWarnFraction) {
    return;
  }
  if (last_rate_warning_time > 0 && timestamp - last_rate_warning_time < kRateWarnIntervalSecs) {
    return;
  }
  last_rate_warning_time = timestamp;

  // Otherwise this failure is silent: the filter keeps running, looks healthy, and simply stops using
  // GPS. Almost always a noise_floor set to the receiver spec rather than the real error budget.
  PRINT_WARNING(RED "[GPS]: %d%% of the last %d fixes REJECTED. GPS is contributing little or nothing.\n" RESET, (int)(100 * frac),
                (int)recent_outcomes.size());
  PRINT_WARNING(RED "[GPS]: the usual cause is gps_noise_floor (%.2f m) being smaller than the real error budget -- it must cover "
                    "VIO drift, lever-arm and transform error, not just receiver noise. The gate rejects beyond ~%.2f m.\n" RESET,
                _options.noise_floor, _options.noise_floor * std::sqrt(_options.chi2_multipler * 7.815));
}

void UpdaterGPS::feed_init(std::shared_ptr<State> state, const ov_core::GpsData &meas) {

  std::shared_ptr<PoseJPL> clone;
  bool own_clone = false;
  if (state->_clones_IMU.count(meas.timestamp)) {
    clone = state->_clones_IMU.at(meas.timestamp);
  } else {
    _prop->propagate_and_clone(state, meas.timestamp);
    if (state->_timestamp != meas.timestamp || state->_clones_IMU.count(meas.timestamp) == 0) {
      PRINT_WARNING(YELLOW "[GPS]: unable to propagate to fix time %.6f during delayed init, skipping\n" RESET, meas.timestamp);
      return;
    }
    clone = state->_clones_IMU.at(meas.timestamp);
    own_clone = true;
  }

  // Exact antenna position in G from the clone -- no interpolation anywhere in this pipeline
  Eigen::Vector3d p_ANTinG = clone->pos() + clone->Rot().transpose() * state->_calib_GPStoIMU->value();


  InitPair pair;
  pair.timestamp = meas.timestamp;
  pair.p_ANTinG = p_ANTinG;
  pair.meas_ENU = meas.meas_ENU;
  pair.cov = meas.cov;
  init_pairs.push_back(pair);

  // The cap bounds the O(n^2) baseline loops in try_initialize(), and more importantly how much VIO
  // drift the oldest retained pair can carry into the baseline-consistency check there.
  while ((int)init_pairs.size() > _options.init_max_pairs) {
    init_pairs.pop_front();
  }

  // Holding clones alive for try_initialize_rigorous() does not work: marginalize_old_clone() trims by
  // oldest timestamp and drops at most one per camera frame, so interleaved GPS clones push
  // _clones_IMU past max_clone_size, breaking the window-keyed feature cleanup. See docs/gps-fusion.md.
  if (own_clone) {
    StateHelper::marginalize(state, clone);
    state->_clones_IMU.erase(meas.timestamp);
  }

  try_initialize(state);
}

void UpdaterGPS::reset_transform(std::shared_ptr<State> state) {

  stat_num_resets++;
  PRINT_WARNING(YELLOW "[GPS]: resetting E-to-G transform (reset #%d; %d accepted / %d rejected fixes so far this run)\n" RESET,
                stat_num_resets, stat_num_accepted, stat_num_rejected);

  if (state->_gps_yaw_EtoG->id() != -1) {
    StateHelper::marginalize(state, state->_gps_yaw_EtoG);
  }
  if (state->_gps_pos_EinG->id() != -1) {
    StateHelper::marginalize(state, state->_gps_pos_EinG);
  }

  // The lever arm was fit under the transform we just threw away: it chased that bad geometry while
  // its covariance shrank, so keeping it would chain the next init off an estimate that is both wrong
  // and too confident for later fixes to pull back.
  //
  // Marginalize-and-reinsert rather than set_initial_covariance(), which only rewrites blocks among
  // `order` and would leave stale cross-terms next to an inflated diagonal.
  if (state->_options.do_calib_gps_leverarm && state->_calib_GPStoIMU->id() != -1) {
    StateHelper::marginalize(state, state->_calib_GPStoIMU);
    state->_calib_GPStoIMU->set_value(_options.leverarm_prior);
    state->_calib_GPStoIMU->set_fej(_options.leverarm_prior);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd H_R(3, 0);
    Eigen::MatrixXd R = std::pow(_options.leverarm_prior_std, 2) * Eigen::Matrix3d::Identity();
    StateHelper::initialize_invertible(state, state->_calib_GPStoIMU, {}, H_R, H_L, R, res);
    PRINT_WARNING(YELLOW "[GPS]: re-seeded antenna lever arm to prior [%.3f, %.3f, %.3f] (std %.3f)\n" RESET, _options.leverarm_prior(0),
                  _options.leverarm_prior(1), _options.leverarm_prior(2), _options.leverarm_prior_std);
  }

  init_pairs.clear();
  _initialized = false;
  consecutive_rigorous_failures = 0;
  consecutive_update_rejections = 0;
  first_rejection_time = -1;

  // Re-accumulating geometry takes far longer than gap_threshold_secs, so without clearing this the
  // first fix after re-init reads as an outage and gets inflated for the re-initialization itself.
  last_fix_time = -1;
  recovery_sigma = -1;
  recovery_added_var = 0.0;
  recovery_attempts = 0;
  recovery_gap = 0.0;
  recovery_capped = false;
}

void UpdaterGPS::horn_align(const std::vector<Eigen::Vector3d> &p_G, const std::vector<Eigen::Vector3d> &z_E, double sigma_gps,
                            double &psi, Eigen::Vector3d &p_EinG, std::vector<bool> &inliers, double &resid_rms) {

  // Closed-form yaw + translation for p_G_i ~= Rz(psi) * z_E_i + p_EinG.
  // The SOURCE (E) is crossed into the TARGET (G); verified by calculus and by a 90-degree worked
  // example in test_gps_jacobians.
  auto solve_once = [&](const std::vector<int> &idx, double &psi_out, Eigen::Vector3d &p_out) {
    Eigen::Vector3d mean_G = Eigen::Vector3d::Zero(), mean_E = Eigen::Vector3d::Zero();
    for (int i : idx) {
      mean_G += p_G[i];
      mean_E += z_E[i];
    }
    mean_G /= (double)idx.size();
    mean_E /= (double)idx.size();
    double cross_sum = 0.0, dot_sum = 0.0;
    for (int i : idx) {
      Eigen::Vector3d xg = p_G[i] - mean_G;
      Eigen::Vector3d xe = z_E[i] - mean_E;
      cross_sum += xe(0) * xg(1) - xe(1) * xg(0);
      dot_sum += xg(0) * xe(0) + xg(1) * xe(1);
    }
    psi_out = std::atan2(cross_sum, dot_sum);
    p_out = mean_G - rot_z(psi_out) * mean_E;
  };

  std::vector<int> idx_all((int)p_G.size());
  for (size_t i = 0; i < p_G.size(); i++)
    idx_all[i] = (int)i;
  solve_once(idx_all, psi, p_EinG);

  // Light outlier rejection: drop large-residual points, re-solve, repeat.
  //
  // The threshold scales off the *median* residual rather than a fixed k*sigma_gps, because one bad
  // fix can pull the initial all-points fit far enough that every residual -- inliers included --
  // exceeds k*sigma_gps, rejecting everything. The median has breakdown point 0.5, so while fewer than
  // half the points are bad it tracks the inliers no matter how contaminated the current fit is.
  std::vector<int> idx_inliers = idx_all;
  inliers.assign(p_G.size(), true);
  for (int iter = 0; iter < 5; iter++) {
    std::vector<double> resid(p_G.size());
    for (size_t i = 0; i < p_G.size(); i++) {
      resid[i] = (p_G[i] - (rot_z(psi) * z_E[i] + p_EinG)).norm();
    }
    std::vector<double> resid_sorted = resid;
    std::sort(resid_sorted.begin(), resid_sorted.end());
    size_t mid = resid_sorted.size() / 2;
    double median = (resid_sorted.size() % 2 == 1) ? resid_sorted[mid] : 0.5 * (resid_sorted[mid - 1] + resid_sorted[mid]);
    double thresh = std::max(3.0 * std::max(sigma_gps, 1e-3), 5.0 * median);

    std::vector<int> idx_next;
    std::vector<bool> inliers_next(p_G.size(), true);
    for (size_t i = 0; i < p_G.size(); i++) {
      if (resid[i] <= thresh) {
        idx_next.push_back((int)i);
      } else {
        inliers_next[i] = false;
      }
    }
    if (idx_next.size() < 3 || idx_next == idx_inliers) {
      inliers = inliers_next;
      break;
    }
    idx_inliers = idx_next;
    inliers = inliers_next;
    solve_once(idx_inliers, psi, p_EinG);
  }

  // Post-fit residual RMS over the surviving inliers -- what the caller must size the transform
  // covariance from, not sigma_gps.
  double sum_sq = 0.0;
  int cnt = 0;
  for (size_t i = 0; i < p_G.size(); i++) {
    if (inliers[i]) {
      sum_sq += (p_G[i] - (rot_z(psi) * z_E[i] + p_EinG)).squaredNorm();
      cnt++;
    }
  }
  resid_rms = (cnt > 0) ? std::sqrt(sum_sq / cnt) : 0.0;
}

bool UpdaterGPS::try_initialize(std::shared_ptr<State> state) {

  if ((int)init_pairs.size() < _options.init_min_meas) {
    return false;
  }

  // Max pairwise XY distance among the accumulated VIO positions. XY-only because a purely vertical
  // excursion carries no yaw information yet would satisfy a 3D gate on its own.
  double baseline = 0.0;
  for (size_t i = 0; i < init_pairs.size(); i++) {
    for (size_t j = i + 1; j < init_pairs.size(); j++) {
      baseline = std::max(baseline, (init_pairs[i].p_ANTinG.head<2>() - init_pairs[j].p_ANTinG.head<2>()).norm());
    }
  }
  if (baseline < _options.init_min_baseline) {
    return false;
  }

  // The model has no scale factor, and monocular-inertial VIO is metric, so both baselines must
  // measure the same physical excursion. A mismatch means VIO has already drifted, and freezing a
  // transform on that geometry locks in a bad yaw permanently.
  double baseline_gps = 0.0;
  for (size_t i = 0; i < init_pairs.size(); i++) {
    for (size_t j = i + 1; j < init_pairs.size(); j++) {
      baseline_gps = std::max(baseline_gps, (init_pairs[i].meas_ENU.head<2>() - init_pairs[j].meas_ENU.head<2>()).norm());
    }
  }
  constexpr double kBaselineConsistencyTol = 0.15;
  if (std::abs(baseline - baseline_gps) > kBaselineConsistencyTol * std::max(baseline, baseline_gps)) {
    PRINT_WARNING(YELLOW
                  "[GPS]: delayed init geometry inconsistent (VIO baseline %.1fm vs GPS baseline %.1fm), refusing to "
                  "initialize -- likely VIO drift, keep accumulating\n" RESET,
                  baseline, baseline_gps);
    return false;
  }

  // Excursion: min eigenvalue of the 2D (XY) covariance of the accumulated VIO positions (so yaw is observable)
  Eigen::Vector2d mean_xy = Eigen::Vector2d::Zero();
  for (const auto &p : init_pairs) {
    mean_xy += p.p_ANTinG.head<2>();
  }
  mean_xy /= (double)init_pairs.size();
  Eigen::Matrix2d cov_xy = Eigen::Matrix2d::Zero();
  for (const auto &p : init_pairs) {
    Eigen::Vector2d d = p.p_ANTinG.head<2>() - mean_xy;
    cov_xy += d * d.transpose();
  }
  cov_xy /= (double)init_pairs.size();
  double tr = cov_xy.trace();
  double det = cov_xy.determinant();
  double min_eig = tr / 2.0 - std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
  if (min_eig < _options.init_min_excursion) {
    return false;
  }

  // Pairs sitting on the XY centroid vote in horn_align's cross/dot sums with an essentially arbitrary
  // sign (pure noise), diluting the yaw estimate, so exclude them from the fit.
  //
  // The threshold is geometric with a noise floor, never noise_floor alone: "far enough to carry yaw
  // information" is a property of the trajectory, not the receiver, and tying the two would let a
  // lower noise_floor silently disable this filter.
  double xy_excl_thresh = std::max(3.0 * _options.noise_floor, 0.1 * baseline);
  std::vector<size_t> idx_yaw;
  for (size_t i = 0; i < init_pairs.size(); i++) {
    if ((init_pairs[i].p_ANTinG.head<2>() - mean_xy).norm() > xy_excl_thresh) {
      idx_yaw.push_back(i);
    }
  }
  if (idx_yaw.size() < 3) {
    PRINT_DEBUG("[GPS]: delayed init not ready yet (only %d of %d pairs are XY-informative), keep accumulating\n", (int)idx_yaw.size(),
                (int)init_pairs.size());
    return false;
  }

  // Enough geometry accumulated: compute the closed-form 4-DOF guess from the XY-informative pairs only
  std::vector<Eigen::Vector3d> p_G, z_E;
  double sigma_gps = 0.0;
  for (size_t i : idx_yaw) {
    p_G.push_back(init_pairs[i].p_ANTinG);
    z_E.push_back(init_pairs[i].meas_ENU);
    sigma_gps += std::sqrt(init_pairs[i].cov.trace() / 3.0);
  }
  sigma_gps /= (double)idx_yaw.size();

  double psi;
  Eigen::Vector3d p_EinG;
  std::vector<bool> inliers_sub;
  double resid_rms;
  horn_align(p_G, z_E, sigma_gps, psi, p_EinG, inliers_sub, resid_rms);

  // Matching baselines alone can still hide a wrong yaw that happens to preserve pairwise distances.
  // A large residual relative to the baseline means the fit anchors neither yaw nor translation.
  constexpr double kResidRmsToBaselineTol = 0.25;
  if (resid_rms > kResidRmsToBaselineTol * baseline) {
    PRINT_WARNING(YELLOW
                  "[GPS]: delayed init fit residual too large relative to baseline (resid_rms=%.2fm, baseline=%.1fm), "
                  "keep accumulating\n" RESET,
                  resid_rms, baseline);
    return false;
  }

  // Expand back to full init_pairs indexing. Pairs the XY filter excluded are treated as outliers:
  // they played no part in this fit, so live_pairs below must not credit them either.
  std::vector<bool> inliers(init_pairs.size(), false);
  for (size_t k = 0; k < idx_yaw.size(); k++) {
    inliers[idx_yaw[k]] = inliers_sub[k];
  }

  // Set the linearization point before either insertion path. Fine to do while the variables are
  // still out of the covariance: value()/fej() work regardless of whether id() == -1.
  Eigen::VectorXd psi_vec(1);
  psi_vec << psi;
  state->_gps_yaw_EtoG->set_value(psi_vec);
  state->_gps_yaw_EtoG->set_fej(psi_vec);
  state->_gps_pos_EinG->set_value(p_EinG);
  state->_gps_pos_EinG->set_fej(p_EinG);

  // Only pairs whose clone is still live can build a jacobian-consistent system for
  // StateHelper::initialize(); a marginalized clone's information is already absorbed, so reusing it
  // would double-count. In practice always empty (feed_init() marginalizes immediately), so the
  // rigorous path never runs -- enabling it needs a GPS-aware window trim.
  std::vector<InitPair> live_pairs;
  for (size_t i = 0; i < init_pairs.size(); i++) {
    if (i < inliers.size() && !inliers[i])
      continue;
    if (state->_clones_IMU.count(init_pairs[i].timestamp)) {
      live_pairs.push_back(init_pairs[i]);
    }
  }

  bool success = false;
  if (!live_pairs.empty()) {
    success = try_initialize_rigorous(state, live_pairs);
  }
  if (success) {
    consecutive_rigorous_failures = 0;
  } else {
    consecutive_rigorous_failures++;
  }

  // Keep accumulating until the rigorous path has failed a few times, then take the simple insertion.
  // The delay avoids reaching for the weaker method immediately; it must never be able to refuse it
  // outright, since the rigorous path never runs and a veto would silently disable GPS fusion.
  if (!success) {
    if (consecutive_rigorous_failures < kSimpleFallbackAfterFailures) {
      PRINT_DEBUG("[GPS]: delayed init not ready yet (live_pairs=%d, rigorous failures=%d), keep accumulating\n", (int)live_pairs.size(),
                  consecutive_rigorous_failures);
      return false;
    }
    double d_centroid = 0.0;
    for (const auto &p : init_pairs) {
      d_centroid += (p.p_ANTinG - p_EinG).norm();
    }
    d_centroid /= (double)init_pairs.size();
    // Size the transform covariance from the Horn post-fit residual RMS, which captures VIO drift over
    // the init window, and inflate for the uncertainty of a single finite-sample fit. sigma_gps
    // underestimates this by 1-2 orders of magnitude on real data, starving the transform of any
    // later correction -- a primary cause of divergence.
    double sigma_eff = std::max(resid_rms, _options.noise_floor);
    const double kInflate = 3.0;
    double sigma_psi2 = kInflate * std::pow(sigma_eff / std::max(baseline, 1e-3), 2);
    double sigma_p2 = kInflate * (std::pow(sigma_eff, 2) / (double)idx_yaw.size() + sigma_psi2 * d_centroid * d_centroid);
    try_initialize_simple_yaw(state, sigma_psi2);
    try_initialize_simple_pos(state, sigma_p2);
    PRINT_WARNING(YELLOW "[GPS]: used the SIMPLE (documented sub-optimal, uncorrelated-with-state) delayed-init fallback\n" RESET);
    success = true;
  }

  _initialized = true;
  consecutive_rigorous_failures = 0;
  PRINT_INFO(GREEN "[GPS]: E-to-G transform initialized! yaw = %.2f deg, p_EinG = %.2f, %.2f, %.2f (from %d pairs, baseline %.1fm)\n" RESET,
             psi * 180.0 / M_PI, p_EinG(0), p_EinG(1), p_EinG(2), (int)init_pairs.size(), baseline);
  init_pairs.clear();
  return success;
}

bool UpdaterGPS::try_initialize_rigorous(std::shared_ptr<State> state, const std::vector<InitPair> &live_pairs) {

  // Collect the distinct clones referenced by the live pairs, and their column offsets
  std::vector<std::shared_ptr<Type>> clones_order;
  std::map<double, int> clone_col;
  int col = 0;
  for (const auto &p : live_pairs) {
    if (clone_col.count(p.timestamp))
      continue;
    clone_col[p.timestamp] = col;
    clones_order.push_back(state->_clones_IMU.at(p.timestamp));
    col += 6;
  }
  int clones_width = col;
  int n = (int)live_pairs.size();

  auto floor_inflate = [&](Eigen::Matrix3d cov) {
    for (int d = 0; d < 3; d++) {
      cov(d, d) = std::max(cov(d, d), std::pow(_options.noise_floor, 2));
    }
    return _options.noise_multiplier * cov;
  };

  // Step 1: yaw (1 dof). H_order = clones only, so p_EinG's error is implicitly fixed at the Horn
  // guess for this step.
  {
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(3 * n, clones_width);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Zero(3 * n, 1);
    Eigen::VectorXd res(3 * n);
    for (int i = 0; i < n; i++) {
      auto clone = state->_clones_IMU.at(live_pairs[i].timestamp);
      std::vector<std::shared_ptr<Type>> H_order_i;
      Eigen::MatrixXd H_i;
      Eigen::VectorXd res_i;
      get_measurement_jacobian(state, clone, live_pairs[i].meas_ENU, H_order_i, H_i, res_i);
      int c0 = clone_col.at(live_pairs[i].timestamp);
      H_R.block(3 * i, c0, 3, 6) = H_i.block(0, 0, 3, 6);
      H_L.block(3 * i, 0, 3, 1) = H_i.block(0, 6, 3, 1);
      res.segment(3 * i, 3) = res_i;
      // Whiten so the stacked noise is isotropic, as StateHelper::initialize() requires
      Eigen::Matrix3d Ri = floor_inflate(live_pairs[i].cov);
      for (int d = 0; d < 3; d++) {
        double s = std::sqrt(Ri(d, d));
        H_R.row(3 * i + d) /= s;
        H_L.row(3 * i + d) /= s;
        res(3 * i + d) /= s;
      }
    }
    Eigen::MatrixXd R_white = Eigen::MatrixXd::Identity(3 * n, 3 * n);
    bool ok = StateHelper::initialize(state, state->_gps_yaw_EtoG, clones_order, H_R, H_L, R_white, res, _options.chi2_multipler);
    if (!ok) {
      return false;
    }
    // initialize_invertible() moves the value via update() but never the fej, which would otherwise
    // stay pinned at the Horn guess. Freeze it at the estimate as of insertion.
    state->_gps_yaw_EtoG->set_fej(state->_gps_yaw_EtoG->value());
  }

  // Step 2: p_EinG (3 dof). H_order = clones + the now-live yaw, so the resulting covariance captures
  // the yaw/pos_EinG cross-correlation.
  {
    std::vector<std::shared_ptr<Type>> H_order_base = clones_order;
    H_order_base.push_back(state->_gps_yaw_EtoG);
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(3 * n, clones_width + 1);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Zero(3 * n, 3);
    Eigen::VectorXd res(3 * n);
    for (int i = 0; i < n; i++) {
      auto clone = state->_clones_IMU.at(live_pairs[i].timestamp);
      std::vector<std::shared_ptr<Type>> H_order_i;
      Eigen::MatrixXd H_i;
      Eigen::VectorXd res_i;
      get_measurement_jacobian(state, clone, live_pairs[i].meas_ENU, H_order_i, H_i, res_i);
      int c0 = clone_col.at(live_pairs[i].timestamp);
      H_R.block(3 * i, c0, 3, 6) = H_i.block(0, 0, 3, 6);
      H_R.block(3 * i, clones_width, 3, 1) = H_i.block(0, 6, 3, 1);
      H_L.block(3 * i, 0, 3, 3) = H_i.block(0, 7, 3, 3);
      res.segment(3 * i, 3) = res_i;
      Eigen::Matrix3d Ri = floor_inflate(live_pairs[i].cov);
      for (int d = 0; d < 3; d++) {
        double s = std::sqrt(Ri(d, d));
        H_R.row(3 * i + d) /= s;
        H_L.row(3 * i + d) /= s;
        res(3 * i + d) /= s;
      }
    }
    Eigen::MatrixXd R_white = Eigen::MatrixXd::Identity(3 * n, 3 * n);
    bool ok = StateHelper::initialize(state, state->_gps_pos_EinG, H_order_base, H_R, H_L, R_white, res, _options.chi2_multipler);
    if (!ok) {
      // Yaw is already in the state, so fall back for pos only rather than leave it half-initialized.
      PRINT_WARNING(YELLOW "[GPS]: pos_EinG rigorous init failed its chi2 gate after yaw succeeded, using simple fallback for pos\n" RESET);
      double sigma_gps = 0.0;
      for (const auto &p : live_pairs) {
        sigma_gps += std::sqrt(p.cov.trace() / 3.0);
      }
      sigma_gps /= (double)live_pairs.size();
      try_initialize_simple_pos(state, std::pow(sigma_gps, 2));
    } else {
      state->_gps_pos_EinG->set_fej(state->_gps_pos_EinG->value());
    }
  }

  return true;
}

void UpdaterGPS::try_initialize_simple_yaw(std::shared_ptr<State> state, double sigma_psi2) {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(1);
  Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd H_R(1, 0);
  Eigen::MatrixXd R(1, 1);
  R << sigma_psi2;
  StateHelper::initialize_invertible(state, state->_gps_yaw_EtoG, {}, H_R, H_L, R, res);
}

void UpdaterGPS::try_initialize_simple_pos(std::shared_ptr<State> state, double sigma_p2) {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd H_R(3, 0);
  Eigen::MatrixXd R = sigma_p2 * Eigen::Matrix3d::Identity();
  StateHelper::initialize_invertible(state, state->_gps_pos_EinG, {}, H_R, H_L, R, res);
}
