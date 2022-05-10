#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <memory>

#include <Eigen/Dense> 
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "triangulation_alg.hpp"
#include "SnavelyReprojectionError.h"

const double non_pix = 100000; 

extern std::vector<std::shared_ptr<double>> camera_parameter;

bool triangulate(const std::vector<Eigen::Matrix4d>& motion,
                 const std::vector<std::vector<double>>& match_points,
                 const Eigen::Matrix3d& K,
                       std::vector<std::vector<double>>& structure);

bool RtMatrixToCam(const Eigen::Matrix4d& RT, 
                   const double focal_length, const double r1, const double r2);

bool CamToRtMatrix(std::shared_ptr<double> cam, Eigen::Matrix4d& RT); 

bool ResetCamVector(int index);

class Frame{
public:
    Frame(const Eigen::MatrixXd& matches, 
          const double& focal_length, 
          const Eigen::Matrix3d& F, 
          const int& im_width, 
          const int& im_height,
          const int& camera_seq) {
        
        focal_length_ = focal_length;
        matches_      = matches;
        im_width_     = im_width;
        im_height_    = im_height;

        for (int i = 0; i < matches_.cols(); ++i) {
            std::vector<double> tmp{matches_(0, i), matches_(1, i), matches_(2, i), matches_(3, i)};
            match_points_.push_back(tmp);
        }

        Eigen::MatrixXd match_reshape_ = Eigen::MatrixXd::Zero(matches_.rows() * matches_.cols() / 2, 2);
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

        Eigen::Matrix<double, 3, 4> RT_;  

        estimate_RT_from_E(E_, match_reshape_, K_, RT_); // slove from 1 to 0

        Eigen::Matrix4d motion_0 = Eigen::Matrix4d::Identity();

        motion_.push_back(motion_0);

        Eigen::Matrix4d tmp_0 = Eigen::Matrix4d::Identity();
        tmp_0.block<3, 3>(0, 0) = RT_.block<3, 3>(0, 0);
        tmp_0.block<3, 1>(0, 3) = RT_.block<3, 1>(0, 3);
        motion_.push_back(tmp_0);

        if (camera_seq == 0) {
            RtMatrixToCam(motion_0, focal_length_, 0, 0);
            RtMatrixToCam(tmp_0, focal_length_, 0, 0);
        } else {
            ResetCamVector(camera_seq);
            RtMatrixToCam(tmp_0, focal_length_, 0, 0);
        }
            
        triangulate(motion_, match_points_, K_, structure_);

        camera_index_.push_back(camera_seq);
        camera_index_.push_back(camera_seq + 1);
    }

public:
    Eigen::MatrixXd matches_;
    double focal_length_;
    int im_width_;
    int im_height_;

    std::vector<std::vector<double>> match_points_;
    Eigen::MatrixXd match_reshape_;

    Eigen::Matrix3d K_;
    Eigen::Matrix3d E_;  

    std::vector<Eigen::Matrix4d> motion_;

    std::vector<int> camera_index_;

    std::vector<std::vector<double>> structure_;
};

bool bundle_adjustment(Frame& frame, bool show);
bool merged_frames(std::vector<Frame>& frames, const std::vector<Eigen::Matrix4d>& cam_RTs);


