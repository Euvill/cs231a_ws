#include <iostream>

#include "epipolar_alg.hpp"

int main(void) {

    /* 
     * pt_2D data format :
     * the size of set : 37
     * 2D point coordinate(y, x): 473.000000 395.000000
     */

    std::string data_path = "/home/euvill/Desktop/cs231a/ps2_ws/data";

    std::cout << "--------------------------------------------------------------- " << std::endl;
    std::cout << "-------------------------set1's solved------------------------- " << std::endl;
    std::cout << "--------------------------------------------------------------- " << std::endl;

    std::string set1_points1_file = data_path + "/set1/pt_2D_1.txt";
    std::string set1_points2_file = data_path + "/set1/pt_2D_2.txt";

    cv::Mat set1_image1 = cv::imread(data_path + "/set1/image1.jpg", cv::IMREAD_COLOR);
    cv::Mat set1_image2 = cv::imread(data_path + "/set1/image2.jpg", cv::IMREAD_COLOR);

    std::vector<Eigen::Vector2d> set1_pts1;
    std::vector<Eigen::Vector2d> set1_pts2;

    readtxt2vec(set1_points1_file, set1_pts1);
    readtxt2vec(set1_points2_file, set1_pts2);

    Eigen::Matrix3d set1_Fundamental_matrix;
    lls_eight_point_alg(set1_pts1, set1_pts2, set1_Fundamental_matrix, true);

    std::cout << std::endl;

    Eigen::Matrix3d set1_Fundamental_matrix_N;
    normalized_eight_point_alg(set1_pts1, set1_pts2, set1_Fundamental_matrix_N);

    std::cout << std::endl;

    std::cout << "Fundamental Matrix epiolar distance (image1) : ";
    std::cout << compute_distance_to_epipolar_lines(set1_pts1, set1_pts2, set1_Fundamental_matrix) << std::endl;
    std::cout << std::endl;

    std::cout << "Fundamental Matrix epiolar distance (image2) : ";
    std::cout << compute_distance_to_epipolar_lines(set1_pts2, set1_pts1, set1_Fundamental_matrix.transpose()) << std::endl;
    std::cout << std::endl;

    std::cout << "Normalized Fundamental Matrix epiolar distance (image1) : ";
    std::cout << compute_distance_to_epipolar_lines(set1_pts1, set1_pts2, set1_Fundamental_matrix_N) << std::endl;
    std::cout << std::endl;

    std::cout << "Normalized Fundamental Matrix epiolar distance (image2) : ";
    std::cout << compute_distance_to_epipolar_lines(set1_pts2, set1_pts1, set1_Fundamental_matrix_N.transpose()) << std::endl;
    std::cout << std::endl;

    plot_epipolar_lines_on_images(set1_pts1, set1_pts2, "Fundamental Matrix ", 
                                  set1_image1, set1_image2, set1_Fundamental_matrix);

    plot_epipolar_lines_on_images(set1_pts1, set1_pts2, "Normalized Fundamental Matrix ", 
                                  set1_image1, set1_image2, set1_Fundamental_matrix_N);

    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------- " << std::endl;
    std::cout << "-------------------------set2's solved------------------------- " << std::endl;
    std::cout << "--------------------------------------------------------------- " << std::endl;

    std::string set2_points1_file = data_path + "/set2/pt_2D_1.txt";
    std::string set2_points2_file = data_path + "/set2/pt_2D_2.txt";

    cv::Mat set2_image1 = cv::imread(data_path + "/set2/image1.jpg", cv::IMREAD_COLOR);
    cv::Mat set2_image2 = cv::imread(data_path + "/set2/image2.jpg", cv::IMREAD_COLOR);

    std::vector<Eigen::Vector2d> set2_pts1;
    std::vector<Eigen::Vector2d> set2_pts2;

    readtxt2vec(set2_points1_file, set2_pts1);
    readtxt2vec(set2_points2_file, set2_pts2);

    Eigen::Matrix3d set2_Fundamental_matrix;
    lls_eight_point_alg(set2_pts1, set2_pts2, set2_Fundamental_matrix, true);

    std::cout << std::endl;

    Eigen::Matrix3d set2_Fundamental_matrix_N;
    normalized_eight_point_alg(set2_pts1, set2_pts2, set2_Fundamental_matrix_N);

    std::cout << std::endl;

    std::cout << "Fundamental Matrix epiolar distance (image1) : ";
    std::cout << compute_distance_to_epipolar_lines(set2_pts1, set2_pts2, set2_Fundamental_matrix) << std::endl;
    std::cout << std::endl;

    std::cout << "Fundamental Matrix epiolar distance (image2) : ";
    std::cout << compute_distance_to_epipolar_lines(set2_pts2, set2_pts1, set2_Fundamental_matrix.transpose()) << std::endl;
    std::cout << std::endl;

    std::cout << "Normalized Fundamental Matrix epiolar distance (image1) : ";
    std::cout << compute_distance_to_epipolar_lines(set2_pts1, set2_pts2, set2_Fundamental_matrix_N) << std::endl;
    std::cout << std::endl;

    std::cout << "Normalized Fundamental Matrix epiolar distance (image2) : ";
    std::cout << compute_distance_to_epipolar_lines(set2_pts2, set2_pts1, set2_Fundamental_matrix_N.transpose()) << std::endl;
    std::cout << std::endl;

    plot_epipolar_lines_on_images(set2_pts1, set2_pts2, "Fundamental Matrix ", 
                                  set2_image1, set2_image2, set2_Fundamental_matrix);

    plot_epipolar_lines_on_images(set2_pts1, set2_pts2, "Normalized Fundamental Matrix ", 
                                  set2_image1, set2_image2, set2_Fundamental_matrix_N);

    return 0;
}