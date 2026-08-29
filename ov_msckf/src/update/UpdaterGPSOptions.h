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

#ifndef OV_MSCKF_UPDATER_GPS_OPTIONS_H
#define OV_MSCKF_UPDATER_GPS_OPTIONS_H

#include <Eigen/Eigen>
#include <memory>

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace ov_msckf {

/**
 * @brief Options for GPS fusion (see UpdaterGPS). Tuning guidance: docs/gps-fusion.md §13.
 */
struct GPSOptions {

  /// If GPS fusion should be enabled at all
  bool enabled = false;

  /// Chi-squared multiplier applied to both the per-fix update gate and the delayed-init gate
  double chi2_multipler = 5.0;

  /// Minimum std (meters) each axis of the reported fix covariance is floored to.
  ///
  /// SET THIS TO THE FULL ERROR BUDGET, NOT THE RECEIVER SPEC: it must cover VIO drift between fixes,
  /// lever-arm and transform error, and time-sync error, typically 0.5-2 m. Setting it too low is
  /// catastrophic rather than suboptimal -- rejected fixes carry no information, so a slightly wrong
  /// transform can never be corrected. See docs/gps-fusion.md §6 for the measured failure.
  double noise_floor = 0.5;

  /// Multiplier applied to the (floored) GPS covariance from the message
  double noise_multiplier = 1.0;

  /// Fixed GPS-to-IMU time offset (seconds): t_imu = t_gps + toff. NOT calibrated online (out of scope, see design notes).
  double toff = 0.0;

  /// Prior value for the antenna lever arm p_ANTinI (meters), used as the fixed value if not calibrating online
  Eigen::Vector3d leverarm_prior = Eigen::Vector3d::Zero();

  /// Prior standard deviation (meters, per axis) for the antenna lever arm if calibrating online
  double leverarm_prior_std = 0.05;

  /// If we should estimate the antenna lever arm online
  bool do_calib_leverarm = false;

  /// Minimum number of accumulated GPS/VIO pairs before we attempt delayed E-to-G initialization
  int init_min_meas = 30;

  /// Minimum baseline (meters, max pairwise distance) of the accumulated VIO positions before we attempt initialization
  double init_min_baseline = 5.0;

  /// Minimum eigenvalue (meters^2) of the 2D (XY) covariance of the accumulated VIO positions, so that yaw is observable
  double init_min_excursion = 1.0;

  /// Maximum accumulated GPS/VIO pairs kept for delayed init (oldest dropped first). Bounds the
  /// O(n^2) baseline loops and how much VIO drift the oldest retained pair can carry.
  int init_max_pairs = 200;

  /// How long (seconds) a consecutive-rejection streak must last before the E-to-G transform is torn
  /// down. A duration, not a fix count: a count is meaningless without the receiver rate, and at high
  /// rates is short enough that a single turn triggers a reset.
  double reset_after_rejection_secs = 30.0;

  /// A GPS gap longer than this (seconds) counts as an outage. Should sit comfortably above the
  /// normal fix interval so a couple of dropped fixes do not trigger it.
  double gap_threshold_secs = 2.0;

  /// Assumed VIO position drift (meters/second) sizing the post-outage covariance inflation. Measured
  /// ~0.13 m/s on a 3 m/s ground vehicle; the default is deliberately higher, since under-inflating
  /// reintroduces the rejection cascade this exists to prevent.
  double gap_drift_rate = 0.15;

  /// Ceiling (meters, 1-sigma) on the cumulative inflation one outage may apply. Without it a long
  /// gap widens the gate until it stops discriminating and the state snaps onto any fix, multipath
  /// included; past the ceiling the rejection-streak reset takes over instead.
  double gap_max_sigma = 25.0;

  /// Fixes older than (state timestamp - max_late_dt) are dropped with a warning instead of being processed
  double max_late_dt = 0.5;

  /// Simulated GPS outage window, seconds relative to the first fix received (negative = disabled).
  /// Fixes inside it are discarded at the VioManager input, before the queue or the updater, so the
  /// filter is never told they existed. Evaluation-only.
  double dropout_start_secs = -1.0;
  double dropout_end_secs = -1.0;

  /// Nice print function of what parameters we have loaded
  void print(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("gps_enabled", enabled);
      parser->parse_config("gps_chi2_multipler", chi2_multipler);
      parser->parse_config("gps_noise_floor", noise_floor);
      parser->parse_config("gps_noise_multiplier", noise_multiplier);
      parser->parse_config("gps_toff", toff);
      std::vector<double> leverarm = {leverarm_prior(0), leverarm_prior(1), leverarm_prior(2)};
      parser->parse_config("gps_leverarm_prior", leverarm);
      leverarm_prior << leverarm.at(0), leverarm.at(1), leverarm.at(2);
      parser->parse_config("gps_leverarm_prior_std", leverarm_prior_std);
      parser->parse_config("gps_do_calib_leverarm", do_calib_leverarm);
      parser->parse_config("gps_init_min_meas", init_min_meas);
      parser->parse_config("gps_init_min_baseline", init_min_baseline);
      parser->parse_config("gps_init_min_excursion", init_min_excursion);
      parser->parse_config("gps_init_max_pairs", init_max_pairs, false);
      parser->parse_config("gps_reset_after_rejection_secs", reset_after_rejection_secs, false);
      parser->parse_config("gps_gap_threshold_secs", gap_threshold_secs, false);
      parser->parse_config("gps_gap_drift_rate", gap_drift_rate, false);
      parser->parse_config("gps_gap_max_sigma", gap_max_sigma, false);
      parser->parse_config("gps_max_late_dt", max_late_dt);
      parser->parse_config("gps_dropout_start_secs", dropout_start_secs, false);
      parser->parse_config("gps_dropout_end_secs", dropout_end_secs, false);
    }
    PRINT_DEBUG("GPS PARAMETERS:\n");
    PRINT_DEBUG("  - gps_enabled: %d\n", enabled);
    PRINT_DEBUG("  - gps_chi2_multipler: %.2f\n", chi2_multipler);
    PRINT_DEBUG("  - gps_noise_floor: %.3f\n", noise_floor);
    PRINT_DEBUG("  - gps_noise_multiplier: %.2f\n", noise_multiplier);
    PRINT_DEBUG("  - gps_toff: %.4f\n", toff);
    PRINT_DEBUG("  - gps_leverarm_prior: %.3f, %.3f, %.3f\n", leverarm_prior(0), leverarm_prior(1), leverarm_prior(2));
    PRINT_DEBUG("  - gps_leverarm_prior_std: %.3f\n", leverarm_prior_std);
    PRINT_DEBUG("  - gps_do_calib_leverarm: %d\n", do_calib_leverarm);
    PRINT_DEBUG("  - gps_init_min_meas: %d\n", init_min_meas);
    PRINT_DEBUG("  - gps_init_min_baseline: %.2f\n", init_min_baseline);
    PRINT_DEBUG("  - gps_init_min_excursion: %.2f\n", init_min_excursion);
    PRINT_DEBUG("  - gps_init_max_pairs: %d\n", init_max_pairs);
    PRINT_DEBUG("  - gps_reset_after_rejection_secs: %.2f\n", reset_after_rejection_secs);
    PRINT_DEBUG("  - gps_gap_threshold_secs: %.2f\n", gap_threshold_secs);
    PRINT_DEBUG("  - gps_gap_drift_rate: %.3f\n", gap_drift_rate);
    PRINT_DEBUG("  - gps_gap_max_sigma: %.2f\n", gap_max_sigma);
    PRINT_DEBUG("  - gps_max_late_dt: %.2f\n", max_late_dt);
    if (dropout_start_secs >= 0.0 && dropout_end_secs > dropout_start_secs) {
      PRINT_WARNING(YELLOW "  - gps_dropout: SIMULATED OUTAGE ENABLED from t+%.1fs to t+%.1fs after the first fix\n" RESET,
                    dropout_start_secs, dropout_end_secs);
    }
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_GPS_OPTIONS_H
