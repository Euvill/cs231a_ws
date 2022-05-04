#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

bool readtxt2vec(const std::string &file, std::vector<Eigen::Vector2d> &vec);

bool lls_eight_point_alg(const std::vector<Eigen::Vector2d> &pts1,
                         const std::vector<Eigen::Vector2d> &pts2,
                         Eigen::Matrix3d &Fundamental_matrix, bool show);

bool normalized_eight_point_alg(const std::vector<Eigen::Vector2d> &pts1,
                                const std::vector<Eigen::Vector2d> &pts2,
                                Eigen::Matrix3d &Fundamental_matrix);

void plot_epipolar_lines_on_images(const std::vector<Eigen::Vector2d> &pts1,
                                   const std::vector<Eigen::Vector2d> &pts2,
                                   const std::string str,
                                   const cv::Mat &img1, const cv::Mat &img2,
                                   const Eigen::Matrix3d &Fundamental_matrix);

double compute_distance_to_epipolar_lines(const std::vector<Eigen::Vector2d> &pts1,
                                          const std::vector<Eigen::Vector2d> &pts2,
                                          const Eigen::Matrix3d &Fundamental_matrix);

bool compute_epipole(const std::vector<Eigen::Vector2d> &pts1,
                     const Eigen::Matrix3d &Fundamental_matrix,
                     Eigen::Vector3d &epipole);

bool compute_matching_homographies(const Eigen::Vector3d &epipole2,
                                   const Eigen::Matrix3d &Fundamental_matrix,
                                   const cv::Mat &image2,
                                   const std::vector<Eigen::Vector2d> &pts1,
                                   const std::vector<Eigen::Vector2d> &pts2,
                                   Eigen::Matrix3d &H1, Eigen::Matrix3d &H2);