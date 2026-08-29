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

#ifndef OV_MSCKF_UPDATER_GPS_H
#define OV_MSCKF_UPDATER_GPS_H

#include <Eigen/Eigen>
#include <deque>
#include <map>
#include <memory>
#include <vector>

#include "update/UpdaterGPSOptions.h"
#include "utils/sensor_data.h"

namespace ov_type {
class Type;
class PoseJPL;
} // namespace ov_type

namespace ov_msckf {

class State;
class Propagator;

/**
 * @brief Single-filter, position-domain GPS updater (loosely coupled).
 *
 * GPS updates the primary EKF directly rather than a downstream pose graph, so a fix corrects the
 * states that generate drift (velocity, orientation, IMU biases) and not just the reported pose. No
 * pose is ever interpolated: the filter propagates to the fix time, clones, updates, marginalizes.
 * The ENU-to-global transform (1-dof yaw + 3-dof translation) and the antenna lever arm are state
 * variables, delayed-initialized once GPS geometry makes them observable.
 *
 * This class only touches the state through StateHelper and owns its delayed-init buffer. It does
 * not own the GPS queue or drive the clone/marginalize cycle -- that is VioManager::drain_gps_queue().
 *
 * Full design rationale, measured numbers and tuning guidance: docs/gps-fusion.md
 */
class UpdaterGPS {

public:
  /**
   * @brief Default constructor
   * @param options GPS updater options (chi2 multiplier, noise floor, init thresholds, ...)
   * @param prop Propagator, used by feed_init() to clone at the fix time during delayed init
   */
  UpdaterGPS(GPSOptions options, std::shared_ptr<Propagator> prop);

  /// True once the E-to-G transform has been inserted into the state/covariance
  bool transform_initialized() const { return _initialized; }

  /**
   * @brief Process a fix while the E-to-G transform is not yet initialized.
   *
   * Records the exact (p_ANTinG, meas_ENU) pair, marginalizes any clone it created itself, and
   * attempts delayed initialization once enough geometry has accumulated. See docs/gps-fusion.md §5.
   *
   * @param state Pointer to state
   * @param meas GPS measurement (already in the local ENU frame)
   */
  void feed_init(std::shared_ptr<State> state, const ov_core::GpsData &meas);

  /**
   * @brief Inflate the IMU position covariance to account for VIO drift across a GPS outage.
   *
   * Must be called with each fix's timestamp BEFORE the clone is created (so the inflation reaches
   * the clone the update runs against) and only once the transform is initialized. Inflating P
   * rather than widening the chi2 gate is deliberate: P drives both admission and the Kalman gain.
   * A detected gap opens a recovery that keeps inflating until a fix is accepted or the
   * GPSOptions::gap_max_sigma ceiling is hit. See docs/gps-fusion.md §7.
   *
   * @param state Pointer to state
   * @param meas_timestamp Timestamp of the fix about to be processed
   */
  void handle_measurement_gap(std::shared_ptr<State> state, double meas_timestamp);

  /**
   * @brief Chi2-gated EKF update of the state against the given (already-cloned) pose.
   *
   * Does NOT clone or marginalize the passed clone -- that is the caller's responsibility.
   * A sustained streak of rejections tears the transform down via reset_transform().
   *
   * @param state Pointer to state
   * @param meas GPS measurement (already in the local ENU frame)
   * @param clone The IMU pose clone at meas.timestamp we should update against
   * @return true if the fix passed the chi2 gate and was used to update the state
   */
  bool try_update(std::shared_ptr<State> state, const ov_core::GpsData &meas, std::shared_ptr<ov_type::PoseJPL> clone);

  /// Cumulative run statistics. reset_transform() deliberately does not clear the accept/reject
  /// tallies, so a final count stays interpretable across resets; stat_num_resets flags that they
  /// happened.
  int stat_num_accepted = 0;
  int stat_num_rejected = 0;
  int stat_num_resets = 0;

  /**
   * @brief Measurement model + analytical Jacobians, shared by try_update() and delayed init.
   *
   * h(x) = Rz(psi)^T * (p_IinG + R_GtoI^T*p_ANTinI - p_EinG), r = z - h(x)
   *
   * H has the fixed column layout [clone_theta(3), clone_pos(3), yaw(1), pos_EinG(3),
   * (leverarm(3) if StateOptions::do_calib_gps_leverarm)], i.e. 10 or 13 columns, with H_order in
   * the matching variable order so callers can slice/stack. Public so test_gps_jacobians can
   * exercise it against finite differences.
   */
  void get_measurement_jacobian(std::shared_ptr<State> state, std::shared_ptr<ov_type::PoseJPL> clone, const Eigen::Vector3d &meas_ENU,
                                 std::vector<std::shared_ptr<ov_type::Type>> &H_order, Eigen::MatrixXd &H, Eigen::VectorXd &res) const;

  /**
   * @brief Closed-form 4-DOF (yaw + xyz) Horn/Umeyama alignment with a light outlier rejection pass.
   * Model: p_G_i ~= Rz(psi) * z_E_i + p_EinG
   *
   * @param[out] resid_rms Post-fit residual RMS over the surviving inliers (meters). Callers must
   * size the resulting transform covariance from this, not from sigma_gps -- it captures the actual
   * scatter of the fit (dominated by VIO drift), which sigma_gps underestimates by 1-2 orders of
   * magnitude. Public so test_gps_jacobians can check it against a synthetic known transform.
   */
  static void horn_align(const std::vector<Eigen::Vector3d> &p_G, const std::vector<Eigen::Vector3d> &z_E, double sigma_gps, double &psi,
                          Eigen::Vector3d &p_EinG, std::vector<bool> &inliers, double &resid_rms);

protected:
  /// One accumulated (VIO position, GPS measurement) pair used for delayed E-to-G initialization
  struct InitPair {
    double timestamp;
    Eigen::Vector3d p_ANTinG;
    Eigen::Vector3d meas_ENU;
    Eigen::Matrix3d cov;
  };

  /// Try to trigger delayed initialization of the E-to-G transform given the accumulated pairs; returns true if it fired
  bool try_initialize(std::shared_ptr<State> state);

  /// Rigorous path: insert yaw then pos_EinG via StateHelper::initialize() using only pairs whose
  /// clone is still live. Currently unreachable -- feed_init() marginalizes clones immediately, so
  /// live_pairs is always empty. Kept because a GPS-aware window trim would activate it for free.
  bool try_initialize_rigorous(std::shared_ptr<State> state, const std::vector<InitPair> &live_pairs);

  /// Simple (sub-optimal) insertion: diagonal, uncorrelated-with-state covariance. The only path
  /// that actually runs today, so it must never be possible to disable it.
  void try_initialize_simple_yaw(std::shared_ptr<State> state, double sigma_psi2);
  void try_initialize_simple_pos(std::shared_ptr<State> state, double sigma_p2);

  /// Record a chi2 gate outcome and warn if the recent rejection rate is sustained high
  void track_gate_outcome(bool accepted, double timestamp);

  /// Marginalize the E-to-G transform back out of the state, re-seed the lever arm to its prior,
  /// and reset all delayed-init bookkeeping so try_initialize() can fit it again.
  void reset_transform(std::shared_ptr<State> state);

  GPSOptions _options;
  std::shared_ptr<Propagator> _prop;
  std::map<int, double> chi_squared_table;
  bool _initialized = false;

  /// Accumulated (VIO, GPS) pairs for delayed init, bounded to GPSOptions::init_max_pairs (oldest
  /// dropped first). The cap bounds both the O(n^2) baseline loops in try_initialize() and how much
  /// VIO drift the oldest retained pair can carry into the baseline-consistency check there.
  std::deque<InitPair> init_pairs;

  /// Consecutive rigorous-path failures; gates when the simple fallback may kick in
  int consecutive_rigorous_failures = 0;
  static constexpr int kSimpleFallbackAfterFailures = 3;

  /// Consecutive try_update() rejections since init/reset, and when the streak started (-1 if none).
  /// The reset is gated on the streak's *duration*, not its length: a count is meaningless without
  /// the receiver rate, and at high rates is short enough for a single turn to trigger it. The count
  /// below is only a generous backstop.
  int consecutive_update_rejections = 0;
  double first_rejection_time = -1;
  static constexpr int kMaxConsecutiveUpdateRejections = 500;

  /// Timestamp of the last fix that actually reached try_update() (-1 if none since init/reset).
  /// Advanced there and nowhere else: a fix dropped before being judged must not consume the gap.
  double last_fix_time = -1;

  /// In-progress post-outage recovery. recovery_sigma is the per-fix inflation (1-sigma, meters),
  /// recovery_added_var the cumulative variance so far (bounded by gap_max_sigma^2). Cleared by
  /// try_update() on the first acceptance and by reset_transform().
  double recovery_sigma = -1;
  double recovery_added_var = 0.0;
  int recovery_attempts = 0;
  double recovery_gap = 0.0;
  bool recovery_capped = false;

  /// Rolling window of recent gate outcomes. A sustained high rejection rate is the signature of a
  /// mis-set noise_floor and is otherwise a silent failure -- the filter looks healthy while simply
  /// not using GPS.
  std::deque<bool> recent_outcomes;
  double last_rate_warning_time = -1;
  static constexpr size_t kRateWindow = 40;
  static constexpr double kRateWarnFraction = 0.5;
  static constexpr double kRateWarnIntervalSecs = 20.0;

  /// Accepted-update counter, so the lever-arm estimate is logged periodically rather than only on
  /// rejection (i.e. never in the healthy runs whose calibration you actually want to read off).
  int accepted_since_leverarm_log = 0;
  static constexpr int kLeverarmLogEvery = 100;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_GPS_H
