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

extern std::deque<std::vector<Eigen::Matrix<double, 3, 4>>> frames_motion;
extern std::vector<std::shared_ptr<double>> camera_parameter;

bool triangulate(const std::vector<Eigen::Matrix<double, 3, 4>>& motion,
                 const std::vector<Eigen::Vector4d>& match_points,
                 const Eigen::Matrix3d& K,
                       std::vector<std::vector<double>>& structure);

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
            Eigen::Vector4d tmp(matches_(0, i), matches_(1, i), matches_(2, i), matches_(3, i));
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

        estimate_RT_from_E(E_, match_reshape_, K_, RT_);

        Eigen::Matrix<double, 3, 4> motion_0 = Eigen::Matrix<double, 3, 4>::Zero();

        if (camera_seq == 0) {
            motion_0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        }
        else {
            motion_0 = frames_motion.front()[1];
            RT_.block<3, 3>(0, 0) = motion_0.block<3, 3>(0, 0) * RT_.block<3, 3>(0, 0);
            RT_.block<3, 1>(0, 3) = motion_0.block<3, 3>(0, 0) * RT_.block<3, 1>(0, 3) + motion_0.block<3, 1>(0, 3);
            frames_motion.pop_front();
        }

        motion_.push_back(motion_0);
        motion_.push_back(RT_);

        frames_motion.push_back(motion_);

        if (camera_seq == 0) {

            std::shared_ptr<double> cam(new double[9], std::default_delete<double[]>());

            for (int i = 0; i < 6; ++i){
                cam.get()[i] = 0; 
            }
            cam.get()[6] = focal_length_; 
            cam.get()[7] = 0; 
            cam.get()[8] = 0;

            camera_parameter.push_back(cam);
        }
        else
        {
            Eigen::AngleAxisd rotation_vector;
            rotation_vector.fromRotationMatrix(motion_0.block<3, 3>(0, 0));
            Eigen::Vector3d tmpr = rotation_vector.axis() * rotation_vector.angle();
            Eigen::Vector3d tmpt = motion_0.block<3, 1>(0, 3);

            std::shared_ptr<double> cam(new double[9], std::default_delete<double[]>());

            cam.get()[0] = tmpr(0); 
            cam.get()[1] = tmpr(1); 
            cam.get()[2] = tmpr(2); 
            cam.get()[3] = tmpt(0); 
            cam.get()[4] = tmpt(1); 
            cam.get()[5] = tmpt(2); 
            cam.get()[6] = focal_length_; 
            cam.get()[7] = 0; 
            cam.get()[8] = 0; 

            camera_parameter.push_back(cam);

            if (camera_seq == 3) {
                Eigen::AngleAxisd rotation_vector;
                rotation_vector.fromRotationMatrix(RT_.block<3, 3>(0, 0));
                Eigen::Vector3d tmpr = rotation_vector.axis() * rotation_vector.angle();
                Eigen::Vector3d tmpt = RT_.block<3, 1>(0, 3);

                std::shared_ptr<double> cam(new double[9], std::default_delete<double[]>());

                cam.get()[0] = tmpr(0); 
                cam.get()[1] = tmpr(1); 
                cam.get()[2] = tmpr(2); 
                cam.get()[3] = tmpt(0); 
                cam.get()[4] = tmpt(1); 
                cam.get()[5] = tmpt(2); 
                cam.get()[6] = focal_length_; 
                cam.get()[7] = 0; 
                cam.get()[8] = 0; 

                camera_parameter.push_back(cam);
                frames_motion.pop_front();
            }
        }

        triangulate(motion_, match_points_, K_, structure_);

        /*for (int i = 0; i < match_points_.size(); ++i) {
            std::cout << match_points_[i](0) << ", " 
                      << match_points_[i](1) << ", " 
                      << match_points_[i](2) << ", " 
                      << match_points_[i](3) << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;

        for (int k = 0; k < motion_.size(); ++k) {
            for (int i = 0; i < motion_[i].rows(); ++i) {
                for (int j = 0; j < motion_[i].cols(); ++j) {
                    std::cout << motion_[k](i, j) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;*/

    }

public:
    Eigen::MatrixXd matches_;
    double focal_length_;
    int im_width_;
    int im_height_;

    std::vector<Eigen::Vector4d> match_points_;
    Eigen::MatrixXd match_reshape_;

    Eigen::Matrix3d K_;
    Eigen::Matrix3d E_;

    Eigen::Matrix<double, 3, 4> RT_;    

    std::vector<Eigen::Matrix<double, 3, 4>> motion_;

    std::vector<std::vector<double>> structure_;
};

bool bundle_adjustment(Frame& frame);


