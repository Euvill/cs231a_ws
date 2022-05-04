#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>

void compute_vanishing_point(const std::vector<Eigen::Vector2d>& points,
                             Eigen::Vector2d& vanishing_point) {
    double x1 = points[0](0); double y1 = points[0](1);
    double x2 = points[1](0); double y2 = points[1](1);
    double x3 = points[2](0); double y3 = points[2](1);
    double x4 = points[3](0); double y4 = points[3](1);

    double k1 = (y2 - y1) / (x2 - x1);
    double k2 = (y4 - y3) / (x4 - x3);

    double b1 = y2 - k1 * x2;
    double b2 = y4 - k2 * x4;

    // vanishing point coordinates
    double x = (b2 - b1) / (k1 - k2);
    double y = k1 * ((b2 - b1) / (k1 - k2)) + b1;

    vanishing_point(0) = x;
    vanishing_point(1) = y;

    std::cout << "vanishing point: " 
              << vanishing_point(0) << ", " 
              << vanishing_point(1) << std::endl;
}

void compute_K_from_vanishing_points(const std::vector<Eigen::Vector2d>& vanishing_points,
                                     Eigen::MatrixXd& K) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 4);
    Eigen::Vector4d tmp = Eigen::Vector4d(vanishing_points[0](0) * vanishing_points[1](0) + vanishing_points[1](1) * vanishing_points[0](1),
                                          vanishing_points[1](0) + vanishing_points[0](0), vanishing_points[1](1) + vanishing_points[0](1), 1.0);
    A.block<1, 4>(0, 0) = tmp.transpose();
    tmp = Eigen::Vector4d(vanishing_points[0](0) * vanishing_points[2](0) + vanishing_points[2](1) * vanishing_points[0](1),
                          vanishing_points[2](0) + vanishing_points[0](0), vanishing_points[2](1) + vanishing_points[0](1), 1.0);
    A.block<1, 4>(1, 0) = tmp.transpose();
    tmp = Eigen::Vector4d(vanishing_points[1](0) * vanishing_points[2](0) + vanishing_points[2](1) * vanishing_points[1](1),
                          vanishing_points[2](0) + vanishing_points[1](0), vanishing_points[2](1) + vanishing_points[1](1), 1.0);
    A.block<1, 4>(2, 0) = tmp.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV(); // row:4, col:4

    Eigen::Vector4d v_omega = V.block<4, 1>(0, 3);

    Eigen::Matrix3d omega;
    omega << v_omega(0),          0, v_omega(1),
                      0, v_omega(0), v_omega(2),
             v_omega(1), v_omega(2), v_omega(3);
    
    Eigen::Matrix3d inv_K = omega.llt().matrixU();
    K = inv_K.inverse();
    K = K / K(2, 2);

    std::cout << "K matrix: " << std::endl;

    for (int i = 0; i < K.rows(); ++i) {
        for (int j = 0; j < K.cols(); ++j) {
            std::cout << K(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

double compute_angle_between_planes(const std::vector<Eigen::Vector3d>& floor_vanishing, 
                                    const std::vector<Eigen::Vector3d>& box_vanishing, 
                                    const Eigen::Matrix3d& K_actual) {
    Eigen::Vector3d l_floor = floor_vanishing[0].cross(floor_vanishing[1]);
    Eigen::Vector3d l_box   =   box_vanishing[0].cross(  box_vanishing[1]);

    Eigen::MatrixXd omega_inv = K_actual * K_actual.transpose();

    double tmp1 = l_floor.transpose() * omega_inv * l_box;
    double tmp2 = sqrt(l_floor.transpose() * omega_inv * l_floor) * sqrt(l_box.transpose() * omega_inv * l_box);
  
    double theta = std::acos(tmp1 / tmp2);

    std::cout << "theta between two planar: " << theta * (180 / 3.141592654) << std::endl;

    return 1.0;
}

void compute_rotation_matrix_between_cameras(const std::vector<Eigen::Vector3d>& vanishing_pts1, 
                                             const std::vector<Eigen::Vector3d>& vanishing_pts2,
                                             const Eigen::Matrix3d& K_actual, 
                                             Eigen::Matrix3d& Rotation) {
    std::vector<Eigen::Vector3d> d1;
    for (int i = 0; i < vanishing_pts1.size(); ++i) {
        Eigen::Vector3d tmp = K_actual.inverse() * vanishing_pts1[i];
        d1.push_back(tmp / tmp.norm());
    }
    std::vector<Eigen::Vector3d> d2;
    for (int i = 0; i < vanishing_pts2.size(); ++i) {
        Eigen::Vector3d tmp = K_actual.inverse() * vanishing_pts2[i];
        d2.push_back(tmp / tmp.norm());
    }
    Eigen::Matrix3d Dir1;
    Dir1.block<3, 1>(0, 0) = d1[0];
    Dir1.block<3, 1>(0, 1) = d1[1];
    Dir1.block<3, 1>(0, 2) = d1[2];
    Eigen::Matrix3d Dir2;
    Dir2.block<3, 1>(0, 0) = d2[0];
    Dir2.block<3, 1>(0, 1) = d2[1];
    Dir2.block<3, 1>(0, 2) = d2[2];

    Rotation = Dir2 * Dir1.inverse();

    std::cout << "rotation matrix is: " << std::endl;
    for (int i = 0; i < Rotation.rows(); ++i) {
        for (int j = 0; j < Rotation.cols(); ++j) {
            std::cout << Rotation(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

int main(void) {

    std::vector<Eigen::Vector2d> points_v1;
    points_v1.push_back(Eigen::Vector2d(674, 1826));
    points_v1.push_back(Eigen::Vector2d(2456, 1060));
    points_v1.push_back(Eigen::Vector2d(1094, 1340));
    points_v1.push_back(Eigen::Vector2d(1774, 1086));
    Eigen::Vector2d v1;
    compute_vanishing_point(points_v1, v1);

    std::vector<Eigen::Vector2d> points_v2;
    points_v2.push_back(Eigen::Vector2d(674, 1826));
    points_v2.push_back(Eigen::Vector2d(126, 1056));
    points_v2.push_back(Eigen::Vector2d(2456, 1060));
    points_v2.push_back(Eigen::Vector2d(1940, 866));
    Eigen::Vector2d v2;
    compute_vanishing_point(points_v2, v2);

    std::vector<Eigen::Vector2d> points_v3;
    points_v3.push_back(Eigen::Vector2d(1094, 1340));
    points_v3.push_back(Eigen::Vector2d(1080, 598));
    points_v3.push_back(Eigen::Vector2d(1774, 1086));
    points_v3.push_back(Eigen::Vector2d(1840, 478));
    Eigen::Vector2d v3;
    compute_vanishing_point(points_v3, v3);

    std::vector<Eigen::Vector2d> points_v4;
    points_v4.push_back(Eigen::Vector2d(1094, 1340));
    points_v4.push_back(Eigen::Vector2d(1774, 1086));
    points_v4.push_back(Eigen::Vector2d(1080,  598));
    points_v4.push_back(Eigen::Vector2d(1840,  478));
    Eigen::Vector2d v4;
    compute_vanishing_point(points_v4, v4);

    std::vector<Eigen::Vector2d> points_v1b;
    points_v1b.push_back(Eigen::Vector2d(314, 1912));
    points_v1b.push_back(Eigen::Vector2d(2060, 1040));
    points_v1b.push_back(Eigen::Vector2d(750, 1378));
    points_v1b.push_back(Eigen::Vector2d(1438, 1094));
    Eigen::Vector2d v1b;
    compute_vanishing_point(points_v1b, v1b);

    std::vector<Eigen::Vector2d> points_v2b;
    points_v2b.push_back(Eigen::Vector2d(314, 1912));
    points_v2b.push_back(Eigen::Vector2d(36, 1578));
    points_v2b.push_back(Eigen::Vector2d(2060, 1040));
    points_v2b.push_back(Eigen::Vector2d(1598, 882));
    Eigen::Vector2d v2b;
    compute_vanishing_point(points_v2b, v2b);

    std::vector<Eigen::Vector2d> points_v3b;
    points_v3b.push_back(Eigen::Vector2d(750, 1378));
    points_v3b.push_back(Eigen::Vector2d(714, 614));
    points_v3b.push_back(Eigen::Vector2d(1438, 1094));
    points_v3b.push_back(Eigen::Vector2d(1474, 494));
    Eigen::Vector2d v3b;
    compute_vanishing_point(points_v3b, v3b);

    std::vector<Eigen::Vector2d> vanishing_points;
    vanishing_points.push_back(v1);
    vanishing_points.push_back(v2);
    vanishing_points.push_back(v3);
    Eigen::MatrixXd K;
    compute_K_from_vanishing_points(vanishing_points, K);
    Eigen::Matrix3d K_actual;
    K_actual << 2448.0,    0.0, 1253.0,
                   0.0, 2438.0,  986.0,
                   0.0,    0.0,    1.0;
    std::vector<Eigen::Vector3d> floor_vanishing;
    floor_vanishing.push_back(Eigen::Vector3d(v1(0), v1(1), 1.0));
    floor_vanishing.push_back(Eigen::Vector3d(v2(0), v2(1), 1.0));
    std::vector<Eigen::Vector3d> box_vanishing;
    box_vanishing.push_back(Eigen::Vector3d(v3(0), v3(1), 1.0));
    box_vanishing.push_back(Eigen::Vector3d(v4(0), v4(1), 1.0));
    double angle = compute_angle_between_planes(floor_vanishing, box_vanishing, K_actual);


    std::vector<Eigen::Vector3d> vanishing_pts1;
    vanishing_pts1.push_back(Eigen::Vector3d(v1(0), v1(1), 1.0));
    vanishing_pts1.push_back(Eigen::Vector3d(v2(0), v2(1), 1.0));
    vanishing_pts1.push_back(Eigen::Vector3d(v3(0), v3(1), 1.0));
    std::vector<Eigen::Vector3d> vanishing_pts2;
    vanishing_pts2.push_back(Eigen::Vector3d(v1b(0), v1b(1), 1.0));
    vanishing_pts2.push_back(Eigen::Vector3d(v2b(0), v2b(1), 1.0));
    vanishing_pts2.push_back(Eigen::Vector3d(v3b(0), v3b(1), 1.0));
    Eigen::Matrix3d Rotation;
    compute_rotation_matrix_between_cameras(vanishing_pts1, vanishing_pts2, K_actual, Rotation);

    return 0;
}