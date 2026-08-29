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

// Standalone numerical vs. analytical Jacobian test for UpdaterGPS.
// Also sanity checks the closed-form Horn yaw solve against a synthetic known transform.
// This must pass before anything else in the GPS design is trusted (see design doc).

#include <random>
#include <sstream>

#include "state/State.h"
#include "state/StateOptions.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "update/UpdaterGPS.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// Independent re-implementation of the measurement model, used only to finite-difference against.
static Eigen::Vector3d compute_h(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG, double psi, const Eigen::Vector3d &p_EinG,
                                 const Eigen::Vector3d &p_ANTinI) {
  Eigen::Matrix3d C = rot_z(psi).transpose();
  Eigen::Vector3d u = p_IinG + R_GtoI.transpose() * p_ANTinI - p_EinG;
  return C * u;
}

int main() {

  bool all_passed = true;
  const double eps = 1e-6;
  const double tol = 1e-5;

  // Fixed (non-trivial, non-axis-aligned) test values
  Eigen::Matrix3d R_GtoI = rot_x(0.3) * rot_y(-0.2) * rot_z(0.7);
  Eigen::Vector3d p_IinG(1.5, -2.3, 0.4);
  double psi = 0.35;
  Eigen::Vector3d p_EinG(10.0, -5.0, 0.7);
  Eigen::Vector3d p_ANTinI(0.10, 0.05, -0.20);
  Eigen::Vector3d meas_ENU(3.0, -1.0, 0.2);

  for (int trial = 0; trial < 2; trial++) {
    bool do_leverarm = (trial == 1);

    StateOptions options;
    options.do_calib_gps_leverarm = do_leverarm;
    auto state = std::make_shared<State>(options);

    // Wire up a stand-alone clone (does not need to be inserted into the covariance for this test)
    auto clone = std::make_shared<PoseJPL>();
    Eigen::Matrix<double, 7, 1> pose0;
    pose0.block(0, 0, 4, 1) = rot_2_quat(R_GtoI);
    pose0.block(4, 0, 3, 1) = p_IinG;
    clone->set_value(pose0);
    clone->set_fej(pose0);

    Eigen::VectorXd psi_vec(1);
    psi_vec << psi;
    state->_gps_yaw_EtoG->set_value(psi_vec);
    state->_gps_yaw_EtoG->set_fej(psi_vec);
    state->_gps_pos_EinG->set_value(p_EinG);
    state->_gps_pos_EinG->set_fej(p_EinG);
    state->_calib_GPStoIMU->set_value(p_ANTinI);
    state->_calib_GPStoIMU->set_fej(p_ANTinI);

    GPSOptions gps_opts;
    UpdaterGPS updater(gps_opts, nullptr);

    std::vector<std::shared_ptr<Type>> H_order;
    Eigen::MatrixXd H;
    Eigen::VectorXd res;
    updater.get_measurement_jacobian(state, clone, meas_ENU, H_order, H, res);

    int expected_cols = do_leverarm ? 13 : 10;
    if (H.rows() != 3 || H.cols() != expected_cols) {
      PRINT_ERROR(RED "[TEST]: H is %dx%d, expected 3x%d\n" RESET, (int)H.rows(), (int)H.cols(), expected_cols);
      all_passed = false;
      continue;
    }

    Eigen::Vector3d h0 = compute_h(R_GtoI, p_IinG, psi, p_EinG, p_ANTinI);
    if ((res - (meas_ENU - h0)).norm() > 1e-10) {
      PRINT_ERROR(RED "[TEST]: residual does not match an independent computation of z - h(x)\n" RESET);
      all_passed = false;
    }

    Eigen::MatrixXd H_numerical = Eigen::MatrixXd::Zero(3, expected_cols);

    // d/d(delta_theta_I): perturb the clone orientation using the JPL boxplus (matches PoseJPL::update)
    for (int i = 0; i < 3; i++) {
      Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
      dtheta(i) = eps;
      Eigen::Vector4d dq;
      dq << 0.5 * dtheta, 1.0;
      dq = quatnorm(dq);
      Eigen::Vector4d q_pert = quat_multiply(dq, rot_2_quat(R_GtoI));
      Eigen::Matrix3d R_pert = quat_2_Rot(q_pert);
      H_numerical.col(i) = (compute_h(R_pert, p_IinG, psi, p_EinG, p_ANTinI) - h0) / eps;
    }

    // d/d(p_IinG)
    for (int i = 0; i < 3; i++) {
      Eigen::Vector3d p_pert = p_IinG;
      p_pert(i) += eps;
      H_numerical.col(3 + i) = (compute_h(R_GtoI, p_pert, psi, p_EinG, p_ANTinI) - h0) / eps;
    }

    // d/d(psi)
    H_numerical.col(6) = (compute_h(R_GtoI, p_IinG, psi + eps, p_EinG, p_ANTinI) - h0) / eps;

    // d/d(p_EinG)
    for (int i = 0; i < 3; i++) {
      Eigen::Vector3d p_pert = p_EinG;
      p_pert(i) += eps;
      H_numerical.col(7 + i) = (compute_h(R_GtoI, p_IinG, psi, p_pert, p_ANTinI) - h0) / eps;
    }

    // d/d(p_ANTinI), only if calibrating online
    if (do_leverarm) {
      for (int i = 0; i < 3; i++) {
        Eigen::Vector3d p_pert = p_ANTinI;
        p_pert(i) += eps;
        H_numerical.col(10 + i) = (compute_h(R_GtoI, p_IinG, psi, p_EinG, p_pert) - h0) / eps;
      }
    }

    double err = (H - H_numerical).norm();
    PRINT_INFO("[TEST]: trial %d (leverarm=%d): ||H_analytical - H_numerical|| = %.3e\n", trial, (int)do_leverarm, err);
    if (err > tol) {
      all_passed = false;
      std::stringstream ss;
      ss << "H_analytical:\n" << H << "\nH_numerical:\n" << H_numerical << std::endl;
      PRINT_ERROR(RED "[TEST]: jacobian mismatch!\n%s" RESET, ss.str().c_str());
    }
  }

  // Sanity check the Horn closed-form yaw/translation solve against a synthetic known transform
  {
    double true_psi = 0.6;
    Eigen::Vector3d true_pEinG(3.0, -1.0, 0.0);
    std::vector<Eigen::Vector3d> p_G, z_E;
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < 20; i++) {
      Eigen::Vector3d z(dist(gen), dist(gen), 0.0);
      Eigen::Vector3d p = rot_z(true_psi) * z + true_pEinG;
      p_G.push_back(p);
      z_E.push_back(z);
    }
    double psi_est;
    Eigen::Vector3d p_est;
    std::vector<bool> inliers;
    double resid_rms;
    UpdaterGPS::horn_align(p_G, z_E, 0.01, psi_est, p_est, inliers, resid_rms);
    double err_psi = std::abs(psi_est - true_psi);
    double err_p = (p_est - true_pEinG).norm();
    PRINT_INFO("[TEST]: horn_align err_psi=%.3e err_p=%.3e resid_rms=%.3e\n", err_psi, err_p, resid_rms);
    if (err_psi > 1e-8 || err_p > 1e-6) {
      all_passed = false;
      PRINT_ERROR(RED "[TEST]: horn_align did not recover the true synthetic transform!\n" RESET);
    }

    // Also check that a single gross outlier gets rejected by the light RANSAC pass
    p_G.push_back(Eigen::Vector3d(500.0, -500.0, 0.0));
    z_E.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
    UpdaterGPS::horn_align(p_G, z_E, 0.01, psi_est, p_est, inliers, resid_rms);
    if (inliers.back()) {
      all_passed = false;
      PRINT_ERROR(RED "[TEST]: horn_align failed to flag the injected gross outlier\n" RESET);
    }
    if (std::abs(psi_est - true_psi) > 1e-6 || (p_est - true_pEinG).norm() > 1e-4) {
      all_passed = false;
      PRINT_ERROR(RED "[TEST]: horn_align result was corrupted by the injected outlier\n" RESET);
    }
  }

  if (!all_passed) {
    PRINT_ERROR(RED "[TEST]: test_gps_jacobians FAILED\n" RESET);
    return EXIT_FAILURE;
  }
  PRINT_INFO(GREEN "[TEST]: test_gps_jacobians PASSED\n" RESET);
  return EXIT_SUCCESS;
}
