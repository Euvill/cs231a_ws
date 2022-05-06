#include <iostream>

#include "triangulation_alg.hpp"

std::vector<cv::Mat> image_vec;
std::vector<Eigen::Matrix3d> fundamental_matrices;
std::vector<Eigen::Matrix<double, 3, 4>>  unit_test_camera_matrix;
std::vector<Eigen::MatrixXd> matches_subset;

Eigen::Matrix<double, 4, 2> unit_test_image_matches;

double focal_length = 719.5459;

void data_loaded(void) {

    Eigen::Matrix<double, 3, 4> cam_mat1;
    cam_mat1 << 702.070357461425146539, -64.627080867035203937, -143.759715495692091736, 682.561730276152616170, 
                 68.548784347160477637, 716.158532359180185267,   12.818849421125436905, 679.299226799677171584,
                  0.197251630013212720, -0.036416074533596320,     0.979676305711582485,   9.994723980891825121;       
    Eigen::Matrix<double, 3, 4> cam_mat2;
    cam_mat2 << 629.394773250173216184, -131.878411432753239296, -322.825968921563685399, 2089.221578721471814788,
                140.749638027130146156,  705.510765137339035391,  -13.798618676093367341,  764.681273646807881050,
                  0.443415887805456432,   -0.070986233299994900,    0.893500590443888631,   10.871240366762190632;
    Eigen::Matrix<double, 3, 4> cam_mat3;
    cam_mat3 << 496.886393622648370183, -205.012480127700854382, -478.351436740162398564, 3080.176835818995186855, 
                208.220784208786653835,  684.436997232453336437,  -77.048063244383783399,  923.684932964671020272,
                  0.662867574439857665,   -0.118433635823113129,    0.739310525193533241,   12.739032953604789711; 
    Eigen::Matrix<double, 3, 4> cam_mat4;
    cam_mat4 << 348.929410988396057292, -248.184407655376105595, -578.272486074558969449, 1387.403004978845729056,
                263.460483126038695900,  658.095556501567330088, -123.471108161773713618, 1317.516294737142970916, 
                  0.794215536879320450,   -0.211047856931056821,    0.569807408748167155,   13.318417960657356502;
    unit_test_camera_matrix.push_back(cam_mat1);
    unit_test_camera_matrix.push_back(cam_mat2);
    unit_test_camera_matrix.push_back(cam_mat3);
    unit_test_camera_matrix.push_back(cam_mat4);

    unit_test_image_matches << -2.369201660156250000, -175.634399414062500000,
                               22.803573608398437500, -174.544494628906250000,
                               35.098800659179687500, -176.100097656250000000,
                               47.869995117187500000, -167.592590332031250000;

    Eigen::Matrix3d f_tmp1;
    f_tmp1 << -4.04723691e-08, -2.55896526e-07, -4.93926104e-05,
              -1.34663426e-06,  1.96045499e-07, -5.56818858e-03,
               6.19417340e-04,  5.59528844e-03, -5.23679608e-04;
    Eigen::Matrix3d f_tmp2;
    f_tmp2 << 9.70161499e-08,  1.26743601e-06, -1.54506132e-04,
              4.83945527e-07,  3.25985186e-09,  5.60754818e-03,
             -4.74956177e-04, -5.47792282e-03,  2.26098145e-02;
    Eigen::Matrix3d f_tmp3;
    f_tmp3 << 6.75705582e-08, -1.09021840e-06,  5.93224732e-04,
              3.23134307e-06,  6.38757150e-08,  5.80151966e-03,
             -1.27135584e-03, -5.48838112e-03,  7.93359941e-02;
    Eigen::Matrix3d f_tmp4;
    f_tmp3 << -2.66322904e-08,  3.22928035e-07, -1.79473476e-04,
               1.98701325e-06, -4.02694866e-07,  6.93923582e-03,
              -5.76424460e-04, -7.13155017e-03, -3.92654749e-02;
    fundamental_matrices.push_back(f_tmp1);
    fundamental_matrices.push_back(f_tmp2);
    fundamental_matrices.push_back(f_tmp3);
    fundamental_matrices.push_back(f_tmp4);

    
    cv::Mat image1 = cv::imread("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/images/B21.jpg");
    cv::Mat image2 = cv::imread("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/images/B22.jpg");
    cv::Mat image3 = cv::imread("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/images/B23.jpg");
    cv::Mat image4 = cv::imread("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/images/B24.jpg");
    cv::Mat image5 = cv::imread("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/images/B25.jpg");
    image_vec.push_back(image1);
    image_vec.push_back(image2);
    image_vec.push_back(image3);
    image_vec.push_back(image4);
    image_vec.push_back(image5);

    std::ifstream file_;
    file_.open("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/matches_subset.txt");
    std::string tmp;
    int index = 0;

    while (std::getline(file_, tmp)) {
        static bool flag_find_one_row = false;
        static std::vector<double> number;
        static int row_index = 0;
        static bool flag_resize = false;
        if (tmp == "begin") {
            row_index = 0;
            number.clear();
            flag_find_one_row = false;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tmp_zero;
            matches_subset.push_back(tmp_zero);

            flag_resize = true;
        }
        else if (tmp == "end")
        {
            index = index + 1;
        }
        else
        {
            std::string str;
            for (int i = 0; i < tmp.size(); ++i) {
                if (tmp[i] == '[') {
                    flag_find_one_row = false;
                }else if (tmp[i] == ']') {
                    flag_find_one_row = true;
                    if(str.size() != 0) {
                        number.push_back(std::stod(str));
                        str.clear();
                    }
                    if(flag_resize) {
                        matches_subset[index].resize(4, number.size());
                        flag_resize = false;
                    }
                }else {
                    if(tmp[i] == ' ') {
                        if(str.size() != 0) {
                            number.push_back(std::stod(str));
                            str.clear();
                        }
                    }else {
                        str.push_back(tmp[i]);
                    }
                }
            }
    
            if (flag_find_one_row) {
                for (int i = 0; i < matches_subset[index].cols(); ++i) {
                    matches_subset[index](row_index ,i) = number[i];
                }
                number.clear();
                row_index = row_index + 1;
                flag_find_one_row = false;
            }
        }
    }

    file_.close();
}

int main(void) {

    data_loaded();

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = focal_length;
    K(1, 1) = focal_length;
    Eigen::Matrix3d E = K.transpose() * fundamental_matrices[0] * K;

    Eigen::Matrix<double, 3, 4> example_RT; 
    example_RT << 0.9736, -0.0988, -0.2056,  0.9994,
                  0.1019,  0.9948,  0.0045, -0.0089,
                  0.2041, -0.0254,  0.9786,  0.0331;
    
    std::vector<Eigen::Matrix<double, 3, 4>> estimated_RT;

    estimate_initial_RT(E, estimated_RT);

    std::cout << "estimated RT: " << std::endl << std::endl;
    for (int index = 0; index < 4; ++index) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << estimated_RT[index](i, j) << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }    

    Eigen::Matrix2d unit_test_matches;
    unit_test_matches << matches_subset[0](0, 0), matches_subset[0](1, 0), 
                         matches_subset[0](2, 0), matches_subset[0](3, 0);
    
    std::vector<Eigen::Matrix<double, 3, 4>> camera_matrices;
    Eigen::Matrix<double, 3, 4> tmp_camera_m;

    tmp_camera_m.block<3, 3>(0, 0) = K;
    tmp_camera_m.block<3, 1>(0, 3) = Eigen::Vector3d(0, 0, 0);
    camera_matrices.push_back(tmp_camera_m);

    tmp_camera_m = K * example_RT;
    camera_matrices.push_back(tmp_camera_m);

    Eigen::Vector3d estimated_3d_point;

    linear_estimate_3d_point(unit_test_matches, camera_matrices, estimated_3d_point);

    std::cout << "linear estimated 3d point: " ;
    for (int i = 0; i < 3; ++i) {
        std::cout << estimated_3d_point(i) << ", ";
    }
    std::cout << std::endl;

    Eigen::Vector3d expected_3d_point(0.6774, -1.1029, 4.6621);

    std::cout << "Difference: " << fabs(expected_3d_point(0) - estimated_3d_point(0)) + 
                                   fabs(expected_3d_point(1) - estimated_3d_point(1)) + 
                                   fabs(expected_3d_point(2) - estimated_3d_point(2)) << std::endl;
    std::cout << std::endl;

    Eigen::MatrixXd estimated_error;
    reprojection_error(expected_3d_point, unit_test_matches, camera_matrices, estimated_error);
    
    std::cout << "estimated reprojection error: " ;
    for (int i = 0; i < estimated_error.rows(); ++i) {
        for (int j = 0; j < estimated_error.cols(); ++j) {
            std::cout << estimated_error(i, j) << ", ";
        }
    }
    std::cout << std::endl;

    Eigen::Matrix<double, 4, 1> expected_error;
    expected_error << -0.0095458, 
                      -0.5171407,
                       0.0059307,  
                       0.5016310;
    Eigen::Matrix<double, 4, 3> expected_jacobian; 
    expected_jacobian << 154.33943931,           0., -22.42541691,
                                   0., 154.33943931,  36.51165089,
                         141.87950588, -14.27738422, -56.20341644,
                           21.9792766, 149.50628901,  32.23425643;
    Eigen::MatrixXd estimated_jacobian;
    jacobian(expected_3d_point, camera_matrices, estimated_jacobian);

    std::cout << "estimated jacobian: " << std::endl;
    for (int i = 0; i < estimated_jacobian.rows(); ++i) {
        for (int j = 0; j < estimated_jacobian.cols(); ++j) {
            std::cout << estimated_jacobian(i, j) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "reprojection error difference: " 
              << fabs((estimated_error - expected_error).sum()) << std::endl;
    std::cout << "jacobian difference: " 
              << fabs((estimated_jacobian - expected_jacobian).sum()) << std::endl;
    
    Eigen::Vector3d estimated_3d_point_linear;
    Eigen::Vector3d estimated_3d_point_nonlinear;
    Eigen::MatrixXd error_linear;
    Eigen::MatrixXd error_nonlinear;

    linear_estimate_3d_point(unit_test_image_matches, 
                             unit_test_camera_matrix, 
                             estimated_3d_point_linear);

    std::cout << "linear_estimate_3d_point: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << estimated_3d_point_linear(i) << ", ";
    }
    std::cout << std::endl;

    nonlinear_estimate_3d_point(unit_test_image_matches, 
                                unit_test_camera_matrix, 
                                estimated_3d_point_nonlinear);

    std::cout << "nonlinear_estimate_3d_point: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << estimated_3d_point_nonlinear(i) << ", ";
    }
    std::cout << std::endl;

    reprojection_error(estimated_3d_point_linear,
                       unit_test_image_matches, 
                       unit_test_camera_matrix, 
                       error_linear);

    reprojection_error(estimated_3d_point_nonlinear, 
                       unit_test_image_matches,
                       unit_test_camera_matrix,
                       error_nonlinear);
    
    std::cout << "Linear method error: "    << error_linear.norm() << std::endl;
    std::cout << "Nonlinear method error: " << error_nonlinear.norm() << std::endl;
    std::cout << std::endl;

    Eigen::Matrix<double, 3, 4> estimated_RT_2;
    Eigen::Matrix2d image_points;
    image_points << unit_test_image_matches(0, 0), unit_test_image_matches(0, 1),
                    unit_test_image_matches(1, 0), unit_test_image_matches(1, 1);

    estimate_RT_from_E(E, image_points, K, estimated_RT_2); 

    std::cout << "example RT: " << std::endl;
    for (int i = 0; i < example_RT.rows(); ++i) {
        for (int j = 0; j < example_RT.cols(); ++j) {
            std::cout << example_RT(i, j) << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "final estimated RT: " << std::endl;
    for (int i = 0; i < estimated_RT_2.rows(); ++i) {
        for (int j = 0; j < estimated_RT_2.cols(); ++j) {
            std::cout << estimated_RT_2(i, j) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}