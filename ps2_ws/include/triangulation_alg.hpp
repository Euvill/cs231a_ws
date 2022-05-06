#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>

bool estimate_initial_RT(const Eigen::Matrix3d& E, 
                         std::vector<Eigen::Matrix<double, 3, 4>>& estimated_RT);

bool linear_estimate_3d_point(const Eigen::MatrixXd& image_points, 
                              const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                              Eigen::Vector3d& estimated_3d_point);

bool reprojection_error(const Eigen::Vector3d& point_3d, 
                        const Eigen::MatrixXd& image_points, 
                        const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                              Eigen::MatrixXd& error);

bool jacobian(const Eigen::Vector3d& point_3d, 
              const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                    Eigen::MatrixXd& J);

bool nonlinear_estimate_3d_point(const Eigen::MatrixXd& image_points,
                                 const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                                       Eigen::Vector3d& estimated_3d_point);

bool estimate_RT_from_E(const Eigen::Matrix3d& E, 
                        const Eigen::MatrixXd& image_points,
                        const Eigen::Matrix3d& K, 
                              Eigen::Matrix<double, 3, 4>& estimated_RT);