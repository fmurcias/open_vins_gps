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

#ifndef OV_CORE_GPS_CONV_H
#define OV_CORE_GPS_CONV_H

#include <Eigen/Eigen>
#include <cmath>

namespace ov_core {

/**
 * @brief Minimal, dependency-free WGS84 LLA -> ECEF -> local ENU conversion.
 *
 * Deliberately no GeographicLib dependency. The local ENU frame ("E" elsewhere) is defined by a
 * fixed datum, usually the first valid fix. ENU is tangent-plane and locally gravity-aligned by
 * construction, which is why the filter estimates only a 1-DOF yaw for R_EtoG (see UpdaterGPS).
 */
namespace gps {

/// WGS84 ellipsoid constants
struct WGS84 {
  static constexpr double A = 6378137.0;             ///< semi-major axis (m)
  static constexpr double F = 1.0 / 298.257223563;   ///< flattening
  static constexpr double B = A * (1.0 - F);         ///< semi-minor axis (m)
  static constexpr double E2 = F * (2.0 - F);        ///< first eccentricity squared
};

/**
 * @brief Convert geodetic LLA (WGS84) to ECEF
 * @param lat_deg Latitude in degrees
 * @param lon_deg Longitude in degrees
 * @param alt_m Altitude above the WGS84 ellipsoid in meters
 * @return ECEF position (meters)
 */
inline Eigen::Vector3d lla2ecef(double lat_deg, double lon_deg, double alt_m) {
  double lat = lat_deg * M_PI / 180.0;
  double lon = lon_deg * M_PI / 180.0;
  double sinlat = std::sin(lat), coslat = std::cos(lat);
  double sinlon = std::sin(lon), coslon = std::cos(lon);
  double N = WGS84::A / std::sqrt(1.0 - WGS84::E2 * sinlat * sinlat);
  Eigen::Vector3d ecef;
  ecef(0) = (N + alt_m) * coslat * coslon;
  ecef(1) = (N + alt_m) * coslat * sinlon;
  ecef(2) = (N * (1.0 - WGS84::E2) + alt_m) * sinlat;
  return ecef;
}

/**
 * @brief Rotation that maps a ECEF delta vector into the local ENU frame at the given datum
 * @param lat0_deg Datum latitude in degrees
 * @param lon0_deg Datum longitude in degrees
 * @return 3x3 rotation matrix (ECEF -> ENU)
 */
inline Eigen::Matrix3d ecef2enu_rot(double lat0_deg, double lon0_deg) {
  double lat0 = lat0_deg * M_PI / 180.0;
  double lon0 = lon0_deg * M_PI / 180.0;
  double sinlat0 = std::sin(lat0), coslat0 = std::cos(lat0);
  double sinlon0 = std::sin(lon0), coslon0 = std::cos(lon0);
  Eigen::Matrix3d R;
  // clang-format off
  R << -sinlon0,            coslon0,             0.0,
       -sinlat0 * coslon0, -sinlat0 * sinlon0,   coslat0,
        coslat0 * coslon0,  coslat0 * sinlon0,   sinlat0;
  // clang-format on
  return R;
}

/**
 * @brief Convert an ECEF position into the local ENU frame defined by the given ECEF datum
 * @param ecef ECEF position to convert (meters)
 * @param ecef0 ECEF position of the datum origin (meters)
 * @param lat0_deg Datum latitude in degrees
 * @param lon0_deg Datum longitude in degrees
 * @return ENU position relative to the datum (meters)
 */
inline Eigen::Vector3d ecef2enu(const Eigen::Vector3d &ecef, const Eigen::Vector3d &ecef0, double lat0_deg, double lon0_deg) {
  return ecef2enu_rot(lat0_deg, lon0_deg) * (ecef - ecef0);
}

/**
 * @brief Convert geodetic LLA directly into the local ENU frame defined by a LLA datum
 * @param lat_deg Latitude in degrees
 * @param lon_deg Longitude in degrees
 * @param alt_m Altitude above the WGS84 ellipsoid in meters
 * @param lat0_deg Datum latitude in degrees
 * @param lon0_deg Datum longitude in degrees
 * @param alt0_deg Datum altitude in meters
 * @return ENU position relative to the datum (meters)
 */
inline Eigen::Vector3d lla2enu(double lat_deg, double lon_deg, double alt_m, double lat0_deg, double lon0_deg, double alt0_deg) {
  Eigen::Vector3d ecef = lla2ecef(lat_deg, lon_deg, alt_m);
  Eigen::Vector3d ecef0 = lla2ecef(lat0_deg, lon0_deg, alt0_deg);
  return ecef2enu(ecef, ecef0, lat0_deg, lon0_deg);
}

} // namespace gps
} // namespace ov_core

#endif // OV_CORE_GPS_CONV_H
