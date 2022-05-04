#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>

bool readtxt2vec2d(const std::string &file, std::vector<Eigen::Vector2d> &vec);

bool readtxt2vec3d(const std::string &file, std::vector<Eigen::Vector3d> &vec);

bool writevec3d2txt(const std::string &file, const Eigen::MatrixXd structure); 

bool factorization_method(const std::vector<Eigen::Vector2d>& pts1,
                          const std::vector<Eigen::Vector2d>& pts2,
                          Eigen::MatrixXd& M, Eigen::MatrixXd& S);
