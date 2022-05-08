#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include <Eigen/Dense> 
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "triangulation_alg.hpp"

bool triangulate(const std::vector<std::vector<int>>& match_idx, 
                 const std::vector<Eigen::Matrix<double, 3, 4>>& motion,
                 const Eigen::MatrixXd& match_points,
                 const Eigen::Matrix3d& K,
                       std::vector<Eigen::Vector3d>& structure);

class Frame{
public:
    Frame(const Eigen::MatrixXd& matches, 
          const double& focal_length, 
          const Eigen::Matrix3d& F, 
          const int& im_width, 
          const int& im_height) {
        
        focal_length_ = focal_length;
        matches_      = matches;
        im_width_     = im_width;
        im_height_    = im_height;

        int N = matches_.rows();

        std::vector<int> tmp;
        int index = 0;
        for (; index < matches_.cols(); ++index)
             tmp.push_back(index);
        match_idx_.push_back(tmp);
        tmp.clear();
        for (; index < 2 * matches_.cols(); ++index)
             tmp.push_back(index);
        match_idx_.push_back(tmp);

        match_points_.resize(N * matches_.cols() / 2, 2);

        int i = 0;
        for (; i < matches_.cols(); ++i) {
            match_points_.block<1, 2>(i, 0) = Eigen::Vector2d(matches_(0, i), matches_(1, i)).transpose();
        }
        for (int j = 0; j < matches_.cols(); ++j) {
            match_points_.block<1, 2>(i, 0) = Eigen::Vector2d(matches_(2, j), matches_(3, j)).transpose();
            ++i;
        }

        Eigen::MatrixXd match_reshape_ = Eigen::MatrixXd::Zero(N * matches_.cols() / 2, 2);
        for (int i = 0; i < matches_.cols(); ++i) {
            Eigen::Matrix2d tmp;
            tmp << matches_(0, i), matches_(1, i),
                   matches_(2, i), matches_(3, i);
            match_reshape_.block<2, 2>(2 * i, 0) = tmp;
        }

        K_ = Eigen::Matrix3d::Identity();
        K_(0, 0) = focal_length;
        K_(1, 1) = focal_length;

        E_ = K_.transpose() * F * K_;

        estimate_RT_from_E(E_, match_reshape_, K_, RT_);

        Eigen::Matrix<double, 3, 4> motion_0 = Eigen::Matrix<double, 3, 4>::Zero();
        motion_0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        motion_.push_back(motion_0);
        motion_.push_back(RT_);

        triangulate(match_idx_, motion_, match_points_, K_, structure_);

        /*for (int i = 0; i < match_points_.rows(); ++i) {
            for (int j = 0; j < match_points_.cols(); ++j) {
                std::cout << match_points_(i, j) << ", ";
            }
            std::cout << std::endl;
        }

        
        for (int i = 0; i < match_idx_.size(); ++i) {
            for (int j = 0; j < match_idx_[i].size(); ++j) {
                std::cout << match_idx_[i][j] << ", ";
            }
            std::cout << std::endl;
        }*/

    }

public:
    Eigen::MatrixXd matches_;
    double focal_length_;
    int im_width_;
    int im_height_;

    std::vector<std::vector<int>> match_idx_;
    Eigen::MatrixXd match_points_;
    Eigen::MatrixXd match_reshape_;

    Eigen::Matrix3d K_;
    Eigen::Matrix3d E_;

    Eigen::Matrix<double, 3, 4> RT_;    

    std::vector<Eigen::Matrix<double, 3, 4>> motion_;

    std::vector<Eigen::Vector3d> structure_;
};

bool bundle_adjustment(Frame& frame);


