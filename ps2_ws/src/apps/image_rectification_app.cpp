#include <iostream>

#include "epipolar_alg.hpp"

int main(void) {
    
    std::string data_path = "/home/euvill/Desktop/cs231a/ps2_ws/data";

    std::string set1_points1_file = data_path + "/set1/pt_2D_1.txt";
    std::string set1_points2_file = data_path + "/set1/pt_2D_2.txt";

    cv::Mat set1_image1 = cv::imread(data_path + "/set1/image1.jpg", cv::IMREAD_COLOR);
    cv::Mat set1_image2 = cv::imread(data_path + "/set1/image2.jpg", cv::IMREAD_COLOR);

    std::vector<Eigen::Vector2d> set1_pts1;
    std::vector<Eigen::Vector2d> set1_pts2;

    readtxt2vec(set1_points1_file, set1_pts1);
    readtxt2vec(set1_points2_file, set1_pts2);

    std::cout << std::endl;

    Eigen::Matrix3d set1_Fundamental_matrix_N;
    normalized_eight_point_alg(set1_pts1, set1_pts2, set1_Fundamental_matrix_N);

    std::cout << std::endl;

    std::cout << "epipole in image2: ";
    Eigen::Vector3d epipole2;
    compute_epipole(set1_pts1, set1_Fundamental_matrix_N, epipole2);

    std::cout << "epipole in image1: ";
    Eigen::Vector3d epipole1;
    compute_epipole(set1_pts2, set1_Fundamental_matrix_N.transpose(), epipole1);

    Eigen::Matrix3d H1;
    Eigen::Matrix3d H2;
    compute_matching_homographies(epipole2, 
                                  (-1)*set1_Fundamental_matrix_N.transpose(), 
                                  set1_image2, 
                                  set1_pts1, 
                                  set1_pts2, 
                                  H1, 
                                  H2);

    return 0;
}