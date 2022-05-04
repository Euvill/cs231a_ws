#include <iostream>

#include "epipolar_alg.hpp"

int main(void) {
    
    std::string data_path = "../data";

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

    std::cout << "epipole1 in image1: ";
    Eigen::Vector3d epipole1;
    compute_epipole(set1_pts2, set1_Fundamental_matrix_N, epipole1);

    std::cout << "epipole2 in image2: ";
    Eigen::Vector3d epipole2;
    compute_epipole(set1_pts1, set1_Fundamental_matrix_N.transpose(), epipole2);

    std::cout << std::endl;

    Eigen::Matrix3d H1;
    Eigen::Matrix3d H2;
    compute_matching_homographies(epipole2, 
                                  set1_Fundamental_matrix_N, 
                                  set1_image2, 
                                  set1_pts1, 
                                  set1_pts2, 
                                  H1, 
                                  H2);

    std::cout << std::endl;

    std::vector<Eigen::Vector2d> new_points1;
    std::vector<Eigen::Vector2d> new_points2;

    for (int i = 0; i < set1_pts1.size(); ++i) {
        Eigen::Vector3d new_point1 = H1 * Eigen::Vector3d(set1_pts1[i](0), set1_pts1[i](1), 1.0);
        Eigen::Vector3d new_point2 = H2 * Eigen::Vector3d(set1_pts2[i](0), set1_pts2[i](1), 1.0);

        new_point1 = new_point1 / new_point1(2);
        new_point2 = new_point2 / new_point2(2);

        new_points1.push_back(Eigen::Vector2d(new_point1(0), new_point1(1)));
        new_points2.push_back(Eigen::Vector2d(new_point2(0), new_point2(1)));
    }

    cv::Mat newImage1;
    Eigen::Vector2d offset1;
    compute_rectified_image(set1_image1, H1, newImage1, offset1);
    
    cv::Mat newImage2;
    Eigen::Vector2d offset2;
    compute_rectified_image(set1_image2, H2, newImage2, offset2);
    
    for (int i = 0; i < set1_pts1.size(); ++i) {
        new_points1[i] = new_points1[i] - offset1;
        new_points2[i] = new_points2[i] - offset2;
    }

    Eigen::Matrix3d F_New;
    std::cout << "New ";
    normalized_eight_point_alg(new_points1, new_points2, F_New);

    plot_epipolar_lines_on_images(new_points1, new_points2, " ", newImage1, newImage2, F_New);
    
    return 0;
}