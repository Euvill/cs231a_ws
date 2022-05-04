#include <iostream>

#include "reconstruction_alg.hpp"

int main(void) {

    std::string data_path = "../data";

    std::cout << "--------------------------------------------------------------- " << std::endl;
    std::cout << "-------------------------set1's solved------------------------- " << std::endl;
    std::cout << "--------------------------------------------------------------- " << std::endl;

    std::string set1_points1_file = data_path + "/set1/pt_2D_1.txt";
    std::string set1_points2_file = data_path + "/set1/pt_2D_2.txt";
    std::string set1_points3d_file = data_path + "/set1/pt_3D.txt";

    cv::Mat set1_image1 = cv::imread(data_path + "/set1/image1.jpg", cv::IMREAD_COLOR);
    cv::Mat set1_image2 = cv::imread(data_path + "/set1/image2.jpg", cv::IMREAD_COLOR);

    std::vector<Eigen::Vector2d> set1_pts1;
    std::vector<Eigen::Vector2d> set1_pts2;
    std::vector<Eigen::Vector3d> set1_pts3d;

    readtxt2vec2d(set1_points1_file, set1_pts1);
    readtxt2vec2d(set1_points2_file, set1_pts2);
    readtxt2vec3d(set1_points3d_file, set1_pts3d);

    Eigen::MatrixXd set1_Motion;
    Eigen::MatrixXd set1_Structure;

    factorization_method(set1_pts1, set1_pts2, set1_Motion, set1_Structure);

    writevec3d2txt("../bin/set1_structure_esti.txt", set1_Structure);

    std::cout << "--------------------------------------------------------------- " << std::endl;
    std::cout << "-------------------------set1_sub's solved------------------------- " << std::endl;
    std::cout << "--------------------------------------------------------------- " << std::endl;

    std::string set1sub_points1_file = data_path + "/set1_subset/pt_2D_1.txt";
    std::string set1sub_points2_file = data_path + "/set1_subset/pt_2D_2.txt";
    std::string set1sub_points3d_file = data_path + "/set1_subset/pt_3D.txt";

    cv::Mat set1sub_image1 = cv::imread(data_path + "/set1_subset/image1.jpg", cv::IMREAD_COLOR);
    cv::Mat set1sub_image2 = cv::imread(data_path + "/set1_subset/image2.jpg", cv::IMREAD_COLOR);

    std::vector<Eigen::Vector2d> set1sub_pts1;
    std::vector<Eigen::Vector2d> set1sub_pts2;
    std::vector<Eigen::Vector3d> set1sub_pts3d;

    readtxt2vec2d(set1sub_points1_file, set1sub_pts1);
    readtxt2vec2d(set1sub_points2_file, set1sub_pts2);
    readtxt2vec3d(set1sub_points3d_file, set1sub_pts3d);

    Eigen::MatrixXd set1sub_Motion;
    Eigen::MatrixXd set1sub_Structure;

    factorization_method(set1sub_pts1, set1sub_pts2, set1sub_Motion, set1sub_Structure);

    writevec3d2txt("../bin/set1sub_structure_esti.txt", set1sub_Structure);

}