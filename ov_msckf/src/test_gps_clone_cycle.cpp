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

// Unit test for the GPS clone-update-marginalize cycle (see UpdaterGPS / VioManager::drain_gps_queue).
// Drives State + Propagator + UpdaterGPS directly (the same public APIs VioManager's drain loop calls)
// rather than a full VioManager+image pipeline, since the thing under test is the state bookkeeping,
// not feature tracking.
//
// The synthetic IMU stream never rotates (wm=0 always, so R_GtoI=I throughout) and only ever applies a
// piecewise-*constant* acceleration, so its true position is an exact closed-form function of time
// (true_pos() below) that we can independently evaluate to build self-consistent GPS measurements --
// without ever needing to clone at the same timestamp twice (Propagator::propagate_and_clone() asserts
// against that). Two acceleration segments (first along +x, then along +y) give genuine non-collinear
// 2D excursion so the delayed-init geometry gates can actually pass. The assumed truth uses an identity
// E-to-G transform and a zero lever arm, so meas_ENU == true_pos(t) exactly.

#include <memory>

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "state/StateOptions.h"
#include "types/PoseJPL.h"
#include "update/UpdaterGPS.h"
#include "utils/NoiseManager.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// Segment 1 (t < T_SWITCH): am=(A1,0,g) from rest -> p = (0.5*A1*t^2, 0, 0)
// Segment 2 (t >= T_SWITCH): am=(0,A2,g), x-velocity (A1*T_SWITCH) carries over -> p_x linear, p_y quadratic
static constexpr double A1 = 2.0;
static constexpr double A2 = 2.0;
static constexpr double T_SWITCH = 4.0;

static Eigen::Vector3d true_pos(double t) {
  if (t < T_SWITCH) {
    return Eigen::Vector3d(0.5 * A1 * t * t, 0.0, 0.0);
  }
  double s = t - T_SWITCH;
  double x_at_switch = 0.5 * A1 * T_SWITCH * T_SWITCH;
  double vx_at_switch = A1 * T_SWITCH;
  return Eigen::Vector3d(x_at_switch + vx_at_switch * s, 0.5 * A2 * s * s, 0.0);
}

static bool check_cov_symmetric_psd(const Eigen::MatrixXd &Cov) {
  double sym_err = (Cov - Cov.transpose()).norm();
  if (sym_err > 1e-8) {
    PRINT_ERROR(RED "[TEST]: covariance not symmetric, ||Cov-Cov^T|| = %.3e\n" RESET, sym_err);
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cov);
  double min_eig = es.eigenvalues().minCoeff();
  if (min_eig < -1e-9) {
    PRINT_ERROR(RED "[TEST]: covariance not PSD, min eigenvalue = %.3e\n" RESET, min_eig);
    return false;
  }
  return true;
}

int main() {

  bool all_passed = true;

  // ---- Setup: state + propagator fed with the piecewise-constant-acceleration IMU stream above ----
  StateOptions options;
  options.do_fej = true;
  auto state = std::make_shared<State>(options);
  state->_timestamp = 0.0;

  NoiseManager noises;
  auto propagator = std::make_shared<Propagator>(noises, 9.81);
  for (int i = 0; i <= 1600; i++) {
    double t = -1.0 + 0.01 * i; // covers [-1, 15]
    ImuData m;
    m.timestamp = t;
    m.wm = Eigen::Vector3d::Zero();
    m.am = (t < T_SWITCH) ? Eigen::Vector3d(A1, 0.0, 9.81) : Eigen::Vector3d(0.0, A2, 9.81);
    propagator->feed_imu(m);
  }

  GPSOptions gps_opts;
  gps_opts.init_min_meas = 5;
  gps_opts.init_min_baseline = 2.0;
  gps_opts.init_min_excursion = 0.1;
  gps_opts.chi2_multipler = 1000.0; // this test is about bookkeeping, not statistical gating precision
  gps_opts.noise_floor = 0.05;
  auto updaterGPS = std::make_shared<UpdaterGPS>(gps_opts, propagator);
  state->_calib_GPStoIMU->set_value(Eigen::Vector3d::Zero());
  state->_calib_GPStoIMU->set_fej(Eigen::Vector3d::Zero());

  Eigen::Matrix3d R_gps = std::pow(gps_opts.noise_floor, 2) * Eigen::Matrix3d::Identity();
  auto make_fix = [&](double t) {
    GpsData g;
    g.timestamp = t;
    g.meas_ENU = true_pos(t); // identity E->G transform, zero lever arm -> meas == true VIO position
    g.cov = R_gps;
    return g;
  };

  // ---- Phase A: delayed init, fed fixes that never coincide with a camera clone (own_clone path) ----
  // Spread across both acceleration segments so the accumulated positions are non-collinear.
  for (int i = 0; i < 10; i++) {
    updaterGPS->feed_init(state, make_fix(0.5 + 1.0 * i)); // t = 0.5, 1.5, ..., 9.5
  }
  if (!updaterGPS->transform_initialized()) {
    PRINT_ERROR(RED "[TEST]: delayed init did not trigger with enough accumulated non-collinear pairs\n" RESET);
    all_passed = false;
  }
  if (!state->_clones_IMU.empty()) {
    PRINT_ERROR(RED "[TEST]: leftover clone(s) after delayed init, _clones_IMU.size()=%d (expected 0)\n" RESET,
                (int)state->_clones_IMU.size());
    all_passed = false;
  }
  if (!check_cov_symmetric_psd(StateHelper::get_full_covariance(state))) {
    all_passed = false;
  }

  // ---- Phase B: the actual clone/update/marginalize cycle, N GPS fixes between two camera images ----
  propagator->propagate_and_clone(state, 10.0); // camera clone #1
  if (state->_clones_IMU.size() != 1 || state->_clones_IMU.count(10.0) == 0) {
    PRINT_ERROR(RED "[TEST]: expected exactly the camera clone at t=10.0 after cloning it\n" RESET);
    all_passed = false;
  }

  int n_between = 4;
  for (int i = 0; i < n_between; i++) {
    double t = 10.2 + 0.15 * i;
    GpsData g = make_fix(t);

    // Mirrors VioManager::drain_gps_queue's main-path branch exactly: propagate+clone, update, marginalize
    propagator->propagate_and_clone(state, t);
    auto clone = state->_clones_IMU.at(t);
    bool accepted = updaterGPS->try_update(state, g, clone);
    if (!accepted) {
      PRINT_ERROR(RED "[TEST]: GPS update at t=%.3f was rejected unexpectedly\n" RESET, t);
      all_passed = false;
    }
    StateHelper::marginalize(state, clone);
    state->_clones_IMU.erase(t);

    if (state->_clones_IMU.size() != 1 || state->_clones_IMU.count(10.0) == 0) {
      PRINT_ERROR(RED "[TEST]: _clones_IMU corrupted after GPS cycle at t=%.3f (size=%d, expected {10.0})\n" RESET, t,
                  (int)state->_clones_IMU.size());
      all_passed = false;
    }
    if (!check_cov_symmetric_psd(StateHelper::get_full_covariance(state))) {
      all_passed = false;
    }
  }

  propagator->propagate_and_clone(state, 11.0); // camera clone #2
  if (state->_clones_IMU.size() != 2 || state->_clones_IMU.count(10.0) == 0 || state->_clones_IMU.count(11.0) == 0) {
    PRINT_ERROR(RED "[TEST]: expected exactly the two camera clones {10.0, 11.0} after draining %d GPS fixes between them, got %d\n" RESET,
                n_between, (int)state->_clones_IMU.size());
    all_passed = false;
  }

  // ---- Phase C: a fix landing exactly on an existing camera clone (PPS case) must NOT marginalize it ----
  {
    double t = 11.0;
    GpsData g = make_fix(t);
    auto clone = state->_clones_IMU.at(t);
    updaterGPS->try_update(state, g, clone);
    if (state->_clones_IMU.count(t) == 0) {
      PRINT_ERROR(RED "[TEST]: camera clone at t=%.3f was incorrectly removed by a coincident-timestamp GPS update\n" RESET, t);
      all_passed = false;
    }
    if (state->_clones_IMU.size() != 2) {
      PRINT_ERROR(RED "[TEST]: expected clone count to stay at 2 after the coincident-timestamp GPS update, got %d\n" RESET,
                  (int)state->_clones_IMU.size());
      all_passed = false;
    }
  }

  // ---- Phase D: a sustained rejection streak resets the transform AND re-seeds the antenna lever arm ----
  // The lever arm is fit *under* the transform being discarded, so leaving it in place (drifted, with a
  // covariance that has already shrunk) would chain the next delayed init off a poisoned estimate that
  // later fixes are too confident to pull back -- see UpdaterGPS::reset_transform(). Needs its own state
  // since it requires do_calib_gps_leverarm (the phases above deliberately run with it off).
  {
    StateOptions options_d;
    options_d.do_fej = true;
    options_d.do_calib_gps_leverarm = true;
    auto state_d = std::make_shared<State>(options_d);
    state_d->_timestamp = 0.0;

    // Own propagator so Propagator::last_prop_time_offset can't be shared across two states
    auto prop_d = std::make_shared<Propagator>(noises, 9.81);
    for (int i = 0; i <= 1600; i++) {
      double t = -1.0 + 0.01 * i;
      ImuData m;
      m.timestamp = t;
      m.wm = Eigen::Vector3d::Zero();
      m.am = (t < T_SWITCH) ? Eigen::Vector3d(A1, 0.0, 9.81) : Eigen::Vector3d(0.0, A2, 9.81);
      prop_d->feed_imu(m);
    }

    Eigen::Vector3d leverarm_prior(0.10, -0.20, 0.05);
    double leverarm_prior_std = 0.3;
    GPSOptions opts_d = gps_opts;
    opts_d.leverarm_prior = leverarm_prior;
    opts_d.leverarm_prior_std = leverarm_prior_std;
    opts_d.do_calib_leverarm = true;
    opts_d.reset_after_rejection_secs = 2.0; // keeps the test timeline inside the synthetic IMU buffer
    auto upd_d = std::make_shared<UpdaterGPS>(opts_d, prop_d);

    // Mirror what VioManager does at construction when calibrating the lever arm online
    state_d->_calib_GPStoIMU->set_value(leverarm_prior);
    state_d->_calib_GPStoIMU->set_fej(leverarm_prior);
    StateHelper::set_initial_covariance(state_d, std::pow(leverarm_prior_std, 2) * Eigen::Matrix3d::Identity(),
                                        {state_d->_calib_GPStoIMU});

    // R_GtoI = I throughout this stream, so p_ANTinG = p_IinG + p_ANTinI exactly
    auto make_fix_d = [&](double t, const Eigen::Vector3d &offset) {
      GpsData g;
      g.timestamp = t;
      g.meas_ENU = true_pos(t) + leverarm_prior + offset;
      g.cov = R_gps;
      return g;
    };

    for (int i = 0; i < 10; i++) {
      upd_d->feed_init(state_d, make_fix_d(0.5 + 1.0 * i, Eigen::Vector3d::Zero()));
    }
    if (!upd_d->transform_initialized()) {
      PRINT_ERROR(RED "[TEST]: phase D delayed init did not trigger\n" RESET);
      all_passed = false;
    }

    // Accepted updates first, so the lever arm's covariance has actually moved off its prior by the time
    // we force the reset -- otherwise the restoration assertion below would pass vacuously.
    for (int i = 0; i < 4; i++) {
      double t = 10.0 + 0.2 * i;
      prop_d->propagate_and_clone(state_d, t);
      auto clone = state_d->_clones_IMU.at(t);
      upd_d->try_update(state_d, make_fix_d(t, Eigen::Vector3d::Zero()), clone);
      StateHelper::marginalize(state_d, clone);
      state_d->_clones_IMU.erase(t);
    }
    int lid_before = state_d->_calib_GPStoIMU->id();
    double cov_before = StateHelper::get_full_covariance(state_d)(lid_before, lid_before);
    if (std::abs(cov_before - std::pow(leverarm_prior_std, 2)) < 1e-12) {
      PRINT_ERROR(RED "[TEST]: phase D lever-arm covariance never moved off its prior, reset assertion would be vacuous\n" RESET);
      all_passed = false;
    }

    // Now a sustained streak of grossly-wrong fixes. The offset is large enough to blow the chi2 gate
    // even at this test's deliberately permissive multiplier, and the streak spans more than
    // reset_after_rejection_secs of measurement time.
    Eigen::Vector3d bad_offset(500.0, -500.0, 250.0);
    for (int i = 0; i < 8 && upd_d->transform_initialized(); i++) {
      double t = 11.0 + 0.5 * i;
      prop_d->propagate_and_clone(state_d, t);
      auto clone = state_d->_clones_IMU.at(t);
      if (upd_d->try_update(state_d, make_fix_d(t, bad_offset), clone)) {
        PRINT_ERROR(RED "[TEST]: phase D grossly-wrong fix at t=%.3f was accepted\n" RESET, t);
        all_passed = false;
      }
      StateHelper::marginalize(state_d, clone);
      state_d->_clones_IMU.erase(t);
    }

    if (upd_d->transform_initialized()) {
      PRINT_ERROR(RED "[TEST]: phase D transform survived a rejection streak longer than %.1fs\n" RESET,
                  opts_d.reset_after_rejection_secs);
      all_passed = false;
    }
    if (upd_d->stat_num_resets != 1) {
      PRINT_ERROR(RED "[TEST]: phase D expected exactly 1 transform reset, got %d\n" RESET, upd_d->stat_num_resets);
      all_passed = false;
    }
    if (state_d->_gps_yaw_EtoG->id() != -1 || state_d->_gps_pos_EinG->id() != -1) {
      PRINT_ERROR(RED "[TEST]: phase D E-to-G blocks still in the covariance after reset (yaw id=%d, pos id=%d)\n" RESET,
                  state_d->_gps_yaw_EtoG->id(), state_d->_gps_pos_EinG->id());
      all_passed = false;
    }

    // The actual fix: value back at the configured prior, covariance back at prior_std^2, and no stale
    // cross-correlation left to the rest of the state.
    Eigen::Vector3d leverarm_after = state_d->_calib_GPStoIMU->value();
    double val_err = (leverarm_after - leverarm_prior).norm();
    if (val_err > 1e-9) {
      PRINT_ERROR(RED "[TEST]: phase D lever arm not re-seeded to prior after reset (||err||=%.3e)\n" RESET, val_err);
      all_passed = false;
    }
    int lid = state_d->_calib_GPStoIMU->id();
    if (lid == -1) {
      PRINT_ERROR(RED "[TEST]: phase D lever arm was marginalized out and never re-inserted\n" RESET);
      all_passed = false;
    } else {
      Eigen::MatrixXd Cov = StateHelper::get_full_covariance(state_d);
      Eigen::Matrix3d P_ll = Cov.block(lid, lid, 3, 3);
      double cov_err = (P_ll - std::pow(leverarm_prior_std, 2) * Eigen::Matrix3d::Identity()).norm();
      if (cov_err > 1e-9) {
        PRINT_ERROR(RED "[TEST]: phase D lever-arm covariance not restored to prior_std^2 after reset (||err||=%.3e)\n" RESET, cov_err);
        all_passed = false;
      }
      Eigen::MatrixXd cross = Cov.block(0, lid, lid, 3);
      if (cross.norm() > 1e-9) {
        PRINT_ERROR(RED "[TEST]: phase D lever arm re-inserted with stale cross-correlation (||P_lx||=%.3e)\n" RESET, cross.norm());
        all_passed = false;
      }
      if (!check_cov_symmetric_psd(Cov)) {
        all_passed = false;
      }
    }
  }

  // ---- Phase E: post-outage covariance inflation -- escalation, ceiling, and gap accounting ----
  // None of this is reachable from the real dataset: its drift model is sized correctly, so the very
  // first inflation always admits the returning fix and the escalation/ceiling paths never execute.
  // The failure they guard against is silent (GPS quietly stops contributing while the filter looks
  // healthy), so a synthetic case that forces sustained rejection is the only way to cover them.
  // See UpdaterGPS::handle_measurement_gap().
  {
    StateOptions options_e;
    options_e.do_fej = true;
    auto state_e = std::make_shared<State>(options_e);
    state_e->_timestamp = 0.0;

    // Own propagator so Propagator::last_prop_time_offset can't be shared across two states
    auto prop_e = std::make_shared<Propagator>(noises, 9.81);
    for (int i = 0; i <= 1600; i++) {
      double t = -1.0 + 0.01 * i;
      ImuData m;
      m.timestamp = t;
      m.wm = Eigen::Vector3d::Zero();
      m.am = (t < T_SWITCH) ? Eigen::Vector3d(A1, 0.0, 9.81) : Eigen::Vector3d(0.0, A2, 9.81);
      prop_e->feed_imu(m);
    }

    // Short gaps and a fast drift rate keep the whole phase inside the synthetic IMU buffer ([-1, 15]);
    // a literal 60s outage would run off the end of it. The ceiling is 2x the per-fix sigma, so the
    // cumulative variance budget (4 * GAP_SIGMA^2) is spent in exactly four attempts.
    const double NOMINAL_DT = 0.2;
    const double GAP = 1.5;
    const double DRIFT_RATE = 2.0;
    const double GAP_SIGMA = DRIFT_RATE * GAP; // 3.0 m per attempt -> 9 m^2
    GPSOptions opts_e = gps_opts;
    opts_e.chi2_multipler = 5.0;             // unlike the phases above, phase E needs fixes to be rejectable
    opts_e.gap_threshold_secs = 0.5;
    opts_e.gap_drift_rate = DRIFT_RATE;
    opts_e.gap_max_sigma = 2.0 * GAP_SIGMA;  // 6.0 m -> 36 m^2 budget
    opts_e.reset_after_rejection_secs = 1e6; // isolate the gap logic from the rejection-streak reset
    auto upd_e = std::make_shared<UpdaterGPS>(opts_e, prop_e);
    state_e->_calib_GPStoIMU->set_value(Eigen::Vector3d::Zero());
    state_e->_calib_GPStoIMU->set_fej(Eigen::Vector3d::Zero());

    auto pos_block = [&]() -> Eigen::Matrix3d {
      int pid = state_e->_imu->pose()->p()->id();
      return StateHelper::get_full_covariance(state_e).block(pid, pid, 3, 3);
    };
    // The clone/update/marginalize half of what VioManager::drain_gps_queue does. handle_measurement_gap()
    // is deliberately NOT called here -- each site below calls it explicitly, because when it is called
    // (and when it is not) is precisely what this phase tests.
    auto update_fix = [&](double t, const Eigen::Vector3d &offset) {
      prop_e->propagate_and_clone(state_e, t);
      auto clone = state_e->_clones_IMU.at(t);
      GpsData g;
      g.timestamp = t;
      g.meas_ENU = true_pos(t) + offset;
      g.cov = R_gps;
      bool ok = upd_e->try_update(state_e, g, clone);
      StateHelper::marginalize(state_e, clone);
      state_e->_clones_IMU.erase(t);
      return ok;
    };

    for (int i = 0; i < 10; i++) {
      upd_e->feed_init(state_e, make_fix(0.5 + 1.0 * i));
    }
    if (!upd_e->transform_initialized()) {
      PRINT_ERROR(RED "[TEST]: phase E delayed init did not trigger\n" RESET);
      all_passed = false;
    }

    // Every assertion below measures the position covariance immediately either side of a single
    // handle_measurement_gap() call. Comparing across an intervening propagate_and_clone() instead
    // would fold in IMU process noise and make these exact-equality checks meaningless.
    auto gap_delta = [&](double t) -> Eigen::Matrix3d {
      Eigen::Matrix3d before = pos_block();
      upd_e->handle_measurement_gap(state_e, t);
      return Eigen::Matrix3d(pos_block() - before);
    };
    const Eigen::Matrix3d one_inflation = std::pow(GAP_SIGMA, 2) * Eigen::Matrix3d::Identity();

    // E1: fixes at the nominal interval must not inflate anything. The first fix after init also
    // exercises the "no previous fix" path -- feed_init() never advances last_fix_time, so the long
    // accumulation window must not be mistaken for an outage.
    double t_first = 10.0;
    if (gap_delta(t_first).norm() > 1e-12) {
      PRINT_ERROR(RED "[TEST]: phase E inflated on the very first fix after init (no previous fix to bridge from)\n" RESET);
      all_passed = false;
    }
    update_fix(t_first, Eigen::Vector3d::Zero());
    double t_nominal = t_first + NOMINAL_DT;
    if (gap_delta(t_nominal).norm() > 1e-12) {
      PRINT_ERROR(RED "[TEST]: phase E inflated on a nominal %.2fs interval (threshold %.2fs)\n" RESET, NOMINAL_DT,
                  opts_e.gap_threshold_secs);
      all_passed = false;
    }
    update_fix(t_nominal, Eigen::Vector3d::Zero());

    // E2/E3/E4: an outage opens a recovery. Every fix stays grossly wrong so nothing is ever accepted,
    // which is the case the real data cannot produce -- the recovery must keep escalating on its own
    // until the ceiling stops it. The offset blows the chi2 gate even at full inflation.
    const Eigen::Vector3d bad_offset(500.0, -500.0, 250.0);
    int pid = state_e->_imu->pose()->p()->id();

    // E2: the first inflation is exactly (gap_drift_rate*gap)^2 on the position block, and leaves every
    // other block -- including the cross-covariances to the rest of the state -- untouched.
    double t_return = t_nominal + GAP;
    Eigen::MatrixXd cov_before_gap = StateHelper::get_full_covariance(state_e);
    upd_e->handle_measurement_gap(state_e, t_return);
    Eigen::MatrixXd delta = StateHelper::get_full_covariance(state_e) - cov_before_gap;
    if ((delta.block(pid, pid, 3, 3) - one_inflation).norm() > 1e-9) {
      PRINT_ERROR(RED "[TEST]: phase E first inflation was %.4f m^2 on the position block, expected %.4f\n" RESET, delta(pid, pid),
                  std::pow(GAP_SIGMA, 2));
      all_passed = false;
    }
    delta.block(pid, pid, 3, 3).setZero();
    if (delta.norm() > 1e-9) {
      PRINT_ERROR(RED "[TEST]: phase E inflation disturbed the covariance outside the position block (||d||=%.3e)\n" RESET, delta.norm());
      all_passed = false;
    }
    Eigen::Matrix3d cumulative = one_inflation;
    if (update_fix(t_return, bad_offset)) {
      PRINT_ERROR(RED "[TEST]: phase E grossly-wrong returning fix was accepted\n" RESET);
      all_passed = false;
    }

    // E3: the escalation. This fix arrives at the nominal interval, so it has no gap of its own -- a
    // single-shot implementation stops inflating here, which is exactly how the original rejection
    // cascade survived the one fix that was supposed to end it.
    double t_second = t_return + NOMINAL_DT;
    Eigen::Matrix3d d_second = gap_delta(t_second);
    if ((d_second - one_inflation).norm() > 1e-9) {
      PRINT_ERROR(RED "[TEST]: phase E did not escalate on the second post-gap fix (added %.4f m^2, expected %.4f) "
                      "-- inflation is single-shot\n" RESET,
                  d_second(0, 0), std::pow(GAP_SIGMA, 2));
      all_passed = false;
    }
    cumulative += d_second;
    if (update_fix(t_second, bad_offset)) {
      PRINT_ERROR(RED "[TEST]: phase E grossly-wrong fix accepted on escalation attempt 2\n" RESET);
      all_passed = false;
    }

    // Attempts 3 and 4 exhaust the 36 m^2 budget...
    for (int i = 0; i < 2; i++) {
      double t = t_second + NOMINAL_DT * (i + 1);
      cumulative += gap_delta(t);
      update_fix(t, bad_offset);
    }
    if ((cumulative - std::pow(opts_e.gap_max_sigma, 2) * Eigen::Matrix3d::Identity()).norm() > 1e-9) {
      PRINT_ERROR(RED "[TEST]: phase E cumulative inflation was %.4f m^2, expected the %.4f m^2 ceiling\n" RESET, cumulative(0, 0),
                  std::pow(opts_e.gap_max_sigma, 2));
      all_passed = false;
    }

    // ...and E4: once the ceiling is reached, further fixes must add nothing at all. Escalating past it
    // would leave the gate so wide it no longer rejects anything, multipath included.
    for (int i = 0; i < 2; i++) {
      double t = t_second + NOMINAL_DT * (i + 3);
      if (gap_delta(t).norm() > 1e-12) {
        PRINT_ERROR(RED "[TEST]: phase E kept inflating past the %.2f m ceiling\n" RESET, opts_e.gap_max_sigma);
        all_passed = false;
      }
      update_fix(t, bad_offset);
    }

    // E5: an accepted fix closes the recovery, so the next nominal-interval fix inflates nothing.
    double t_good = t_second + NOMINAL_DT * 5;
    upd_e->handle_measurement_gap(state_e, t_good);
    if (!update_fix(t_good, Eigen::Vector3d::Zero())) {
      PRINT_ERROR(RED "[TEST]: phase E a consistent fix was rejected at t=%.3f, cannot test recovery closure\n" RESET, t_good);
      all_passed = false;
    }
    double t_after = t_good + NOMINAL_DT;
    if (gap_delta(t_after).norm() > 1e-12) {
      PRINT_ERROR(RED "[TEST]: phase E still inflating after an accepted fix -- the recovery never closed\n" RESET);
      all_passed = false;
    }
    update_fix(t_after, Eigen::Vector3d::Zero());

    // E6: gap accounting. A fix that never reaches try_update() must not consume the gap. The first call
    // below is sub-threshold and does nothing; the second is only above threshold if that first call left
    // last_fix_time alone. If handle_measurement_gap() advanced it itself, a real outage would be
    // silently swallowed by a fix that was never even judged.
    if (gap_delta(t_after + 0.3).norm() > 1e-12) { // no update_fix(): simulates a dropped/late fix
      PRINT_ERROR(RED "[TEST]: phase E inflated on a sub-threshold 0.30s interval\n" RESET);
      all_passed = false;
    }
    double t_real = t_after + 0.6;
    Eigen::Matrix3d d_real = gap_delta(t_real);
    double expect_var = std::pow(DRIFT_RATE * (t_real - t_after), 2);
    if ((d_real - expect_var * Eigen::Matrix3d::Identity()).norm() > 1e-9) {
      PRINT_ERROR(RED "[TEST]: phase E gap was consumed by a fix that never reached try_update() "
                      "(inflated %.4f m^2, expected %.4f)\n" RESET,
                  d_real(0, 0), expect_var);
      all_passed = false;
    }

    if (!check_cov_symmetric_psd(StateHelper::get_full_covariance(state_e))) {
      all_passed = false;
    }
  }

  if (!all_passed) {
    PRINT_ERROR(RED "[TEST]: test_gps_clone_cycle FAILED\n" RESET);
    return EXIT_FAILURE;
  }
  PRINT_INFO(GREEN "[TEST]: test_gps_clone_cycle PASSED (accepted=%d, rejected=%d)\n" RESET, updaterGPS->stat_num_accepted,
             updaterGPS->stat_num_rejected);
  return EXIT_SUCCESS;
}
