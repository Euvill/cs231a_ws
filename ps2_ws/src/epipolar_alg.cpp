#include "epipolar_alg.hpp"

bool readtxt2vec(const std::string& file, std::vector<Eigen::Vector2d>& vec){
    std::ifstream file_;
    file_.open(file);

    std::string tmp;
    std::getline(file_, tmp);

    while (std::getline(file_, tmp)) {
        std::vector<std::string> number;
        int index = 0;
        while(true) {
            std::string tmpp;
            for (; index < tmp.size(); ++index) {
                if(tmp[index] != ' ') {
                    tmpp.push_back(tmp[index]);
                }
                else
                    break;
            }
            if(number.size() >= 2)
                break;
            number.push_back(tmpp);
            ++index;
        }
        Eigen::Vector2d point(std::stod(number[1]), std::stod(number[0]));
        vec.push_back(point);
    }

    file_.close();
}

bool lls_eight_point_alg(const std::vector<Eigen::Vector2d>& pts1, 
                         const std::vector<Eigen::Vector2d>& pts2,
                         Eigen::Matrix3d& Fundamental_matrix, bool show) {
    
    if (pts1.size() != pts2.size()) {
        std::cout << "lls_eight_point_alg error, the size of two points set must be equal." << std::endl;
        return false;
    }

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(pts1.size(), 9);
    
    for (int i = 0; i < pts1.size(); ++i) {
        Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(1, 9);

        double u1 = pts1[i](0);
        double v1 = pts1[i](1);
        double u2 = pts2[i](0);
        double v2 = pts2[i](1);

        tmp(0, 0) = u1 * u2;
        tmp(0, 1) = u1 * v2;
        tmp(0, 2) = u1;
        tmp(0, 3) = v1 * u2;
        tmp(0, 4) = v1 * v2;
        tmp(0, 5) = v1;
        tmp(0, 6) = u2;
        tmp(0, 7) = v2;
        tmp(0, 8) = 1.0;

        W.block<1, 9>(i, 0) = tmp;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V1 = svd1.matrixV(); 
    Eigen::MatrixXd v_F_hat = V1.block<9, 1>(0, 8);

    Eigen::Matrix3d F_hat;
    F_hat << v_F_hat(0), v_F_hat(3), v_F_hat(6),
             v_F_hat(1), v_F_hat(4), v_F_hat(7),
             v_F_hat(2), v_F_hat(5), v_F_hat(8);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(F_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singularValue = svd2.singularValues();
    Eigen::Matrix3d Sigma = Eigen::Matrix3d::Zero(3, 3);
    Sigma(0, 0) = singularValue(0);
    Sigma(1, 1) = singularValue(1);
    Fundamental_matrix = svd2.matrixU() * Sigma * svd2.matrixV().transpose();

    if (show) {
        std::cout << "Fundamental Matrix: " << std::endl;
        for (int i = 0; i < Fundamental_matrix.rows(); ++i) {
            for (int j = 0; j < Fundamental_matrix.cols(); ++j) {
                std::cout << Fundamental_matrix(i, j) << ",";
            }
            std::cout << std::endl;
        }
    }

    return true;
}

bool normalized_eight_point_alg(const std::vector<Eigen::Vector2d>& pts1, 
                                const std::vector<Eigen::Vector2d>& pts2,
                                Eigen::Matrix3d& Fundamental_matrix) {

    if (pts1.size() != pts2.size()) {
        std::cout << "lls_eight_point_alg error, the size of two points set must be equal." << std::endl;
        return false;
    }

    Eigen::Vector2d mean1(0, 0);
    Eigen::Vector2d mean2(0, 0);

    for (int i = 0; i < pts1.size(); ++i) {
        mean1(0) = mean1(0) + pts1[i](0);
        mean1(1) = mean1(1) + pts1[i](1);
        mean2(0) = mean2(0) + pts2[i](0);
        mean2(1) = mean2(1) + pts2[i](1);
    }

    mean1(0) = mean1(0) / pts1.size();
    mean1(1) = mean1(1) / pts1.size();
    mean2(0) = mean2(0) / pts2.size();
    mean2(1) = mean2(1) / pts2.size();

    std::vector<Eigen::Vector2d> pts1_uncentered;
    std::vector<Eigen::Vector2d> pts2_uncentered;

    for (int i = 0; i < pts1.size(); ++i) {
        Eigen::Vector2d point1 = pts1[i] - mean1;
        Eigen::Vector2d point2 = pts2[i] - mean2;
        pts1_uncentered.push_back(point1);
        pts2_uncentered.push_back(point2);
    }

    double d1_sum = 0;
    double d2_sum = 0;
    for (int i = 0; i < pts1.size(); ++i) {
        d1_sum = d1_sum + pow(pts1_uncentered[i](0), 2) + pow(pts1_uncentered[i](1), 2);
        d2_sum = d2_sum + pow(pts2_uncentered[i](0), 2) + pow(pts2_uncentered[i](1), 2);
    }
    d1_sum = d1_sum / pts1.size();
    d2_sum = d2_sum / pts1.size();

    double scale1 = sqrt(2 / d1_sum);
    double scale2 = sqrt(2 / d2_sum);

    Eigen::Matrix3d T1;
    T1 << scale1,      0, -mean1(0) * scale1,
               0, scale1, -mean1(1) * scale1,
               0,      0,                1.0;
    Eigen::Matrix3d T2;
    T2 << scale2,      0, -mean2(0) * scale2,
               0, scale2, -mean2(1) * scale2,
               0,      0,                1.0;

    std::vector<Eigen::Vector2d> pts1_normalized;
    std::vector<Eigen::Vector2d> pts2_normalized;
    for (int i = 0; i < pts1.size(); ++i) {
        Eigen::Vector3d point1_normalized = T1 * Eigen::Vector3d(pts1[i](0), pts1[i](1), 1.0);
        Eigen::Vector3d point2_normalized = T2 * Eigen::Vector3d(pts2[i](0), pts2[i](1), 1.0);
        pts1_normalized.push_back(Eigen::Vector2d(point1_normalized(0), point1_normalized(1)));
        pts2_normalized.push_back(Eigen::Vector2d(point2_normalized(0), point2_normalized(1)));
    }

    lls_eight_point_alg(pts1_normalized, pts2_normalized, Fundamental_matrix, false);
    
    // don't know why there is a negative factor
    Fundamental_matrix = (-1) * T2.transpose() * Fundamental_matrix * T1; 

    std::cout << "Normalized Fundamental Matrix: " << std::endl;
    for (int i = 0; i < Fundamental_matrix.rows(); ++i) {
        for (int j = 0; j < Fundamental_matrix.cols(); ++j) {
            std::cout << Fundamental_matrix(i, j) << ",";
        }
        std::cout << std::endl;
    }    
}

void plot_epipolar_lines_on_images(const std::vector<Eigen::Vector2d>& pts1, 
                                   const std::vector<Eigen::Vector2d>& pts2,
                                   const std::string str,
                                   const cv::Mat& img1, const cv::Mat& img2, 
                                   const Eigen::Matrix3d& Fundamental_matrix) {

    std::vector<cv::Vec<double, 3>> line1;
    std::vector<cv::Vec<double, 3>> line2;

    for (size_t i = 0; i < pts1.size(); ++i) {
        Eigen::Vector3d point1(pts1[i](0), pts1[i](1), 1.0);
        Eigen::Vector3d point2(pts2[i](0), pts2[i](1), 1.0);
        Eigen::Vector3d line1_ = Fundamental_matrix.transpose() * point2;
        Eigen::Vector3d line2_ = Fundamental_matrix * point1;
        line1.push_back(cv::Vec<double, 3>(line1_(0), line1_(1), line1_(2)));
        line2.push_back(cv::Vec<double, 3>(line2_(0), line2_(1), line2_(2)));
    }

    cv::Mat image1; img1.copyTo(image1);
    cv::Mat image2; img2.copyTo(image2);

    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::line(image1, 
                 cv::Point(0, -line1[i][2] * 1.0 / line1[i][1]),
                 cv::Point(image1.cols, -(line1[i][2] + line1[i][0] * image1.cols) * 1.0 / line1[i][1]), 
                 cv::Scalar(0, 0, 255));
        cv::line(image2, 
                 cv::Point(0, -line2[i][2] * 1.0 / line2[i][1]),
                 cv::Point(image2.cols, -(line2[i][2] + line2[i][0] * image2.cols) * 1.0 / line2[i][1]), 
                 cv::Scalar(0, 0, 255));
        cv::circle(image1, cv::Point2f(pts1[i](0), pts1[i](1)), 3, cv::Scalar(255, 0, 0), 3);
        cv::circle(image2, cv::Point2f(pts2[i](0), pts2[i](1)), 3, cv::Scalar(255, 0, 0), 3);
    }
    
    cv::imshow(str + "line1", image1);
    cv::imshow(str + "line2", image2);
    cv::waitKey(0);
}

double compute_distance_to_epipolar_lines(const std::vector<Eigen::Vector2d>& pts1, 
                                          const std::vector<Eigen::Vector2d>& pts2,
                                          const Eigen::Matrix3d& Fundamental_matrix) {
    
    std::vector<cv::Vec<double, 3>> line;
    
    for (size_t i = 0; i < pts2.size(); ++i) {

        Eigen::Vector3d point2(pts2[i](0), pts2[i](1), 1.0);

        Eigen::Vector3d line_ = Fundamental_matrix.transpose() * point2;
        
        line.push_back(cv::Vec<double, 3>(line_(0), line_(1), line_(2)));
    }

    double dis_sum = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {

        dis_sum += fabs(line[i][0] * pts1[i](0) + line[i][1] * pts1[i](1) + line[i][2]) 
                    / sqrt(pow(line[i][0], 2.0) + pow(line[i][1], 2.0));

    }

    return dis_sum / pts1.size();
}

bool compute_epipole(const std::vector<Eigen::Vector2d>& pts1, 
                     const Eigen::Matrix3d& Fundamental_matrix,
                     Eigen::Vector3d& epipole) {

    Eigen::MatrixXd line = Eigen::MatrixXd::Zero(pts1.size(), 3);

    for (size_t i = 0; i < pts1.size(); ++i) {

        Eigen::Vector3d point1(pts1[i](0), pts1[i](1), 1.0);

        Eigen::Vector3d line_ = Fundamental_matrix.transpose() * point1;
        
        line.block<1, 3>(i, 0) = line_.transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(line, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    
    if (V.block<3, 1>(0, 2)(2) == 0)
        return false;
    else {
        epipole = V.block<3, 1>(0, 2);
        epipole = epipole / epipole(2);
    }

    std::cout << epipole(0) << "," << epipole(1) << "," << epipole(2) << std::endl;

    return true;
}

bool compute_matching_homographies(const Eigen::Vector3d& epipole2, 
                                   const Eigen::Matrix3d& Fundamental_matrix, 
                                   const cv::Mat& image2, 
                                   const std::vector<Eigen::Vector2d>& pts1, 
                                   const std::vector<Eigen::Vector2d>& pts2,
                                   Eigen::Matrix3d& H1, Eigen::Matrix3d& H2) {
    double image_width  = image2.cols;
    double image_height = image2.rows;

    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    T(0, 2) = -image_width / 2;
    T(1, 2) = -image_height / 2;

    Eigen::Vector3d vec_e = T * epipole2;
    double alpha = vec_e(0) >= 0 ? 1.0 : -1.0;
    double tmp_e3 = sqrt(pow(vec_e(0), 2) + pow(vec_e(1), 2));

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(0, 0) = alpha * vec_e(0) / tmp_e3;
    R(0, 1) = alpha * vec_e(1) / tmp_e3;
    R(1, 0) = -R(0, 1);
    R(1, 1) =  R(0, 0);

    double f = (R * vec_e)(0);
    
    Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
    G(2, 0) = -1.0 / f;

    H2 = T.inverse() * G * R * T;

    Eigen::Matrix3d vec_e_wedge = Eigen::Matrix3d::Zero();
    vec_e_wedge(0, 1) =-epipole2(2);
    vec_e_wedge(0, 2) = epipole2(1);
    vec_e_wedge(1, 0) = epipole2(2); 
    vec_e_wedge(1, 2) =-epipole2(0);
    vec_e_wedge(2, 0) =-epipole2(1);
    vec_e_wedge(2, 1) = epipole2(0);

    Eigen::Matrix3d tmp = epipole2.replicate<1, 3>();

    Eigen::Matrix3d M = vec_e_wedge * Fundamental_matrix + tmp;

    Eigen::MatrixXd pts1_Mat = Eigen::MatrixXd::Zero(pts1.size(), 3);
    for (int i = 0; i < pts1.size(); ++i) {
        pts1_Mat.block<1, 3>(i, 0) = Eigen::Vector3d(pts1[i](0), pts1[i](1), 1.0).transpose();
    }
    Eigen::MatrixXd pts2_Mat = Eigen::MatrixXd::Zero(pts2.size(), 3);
    for (int i = 0; i < pts2.size(); ++i) {
        pts2_Mat.block<1, 3>(i, 0) = Eigen::Vector3d(pts2[i](0), pts2[i](1), 1.0).transpose();
    }

    Eigen::MatrixXd pts1_hat = (H2 * M * pts1_Mat.transpose()).transpose();
    Eigen::MatrixXd pts2_hat = (H2 * pts2_Mat.transpose()).transpose();

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(pts1_hat.rows(), pts1_hat.cols());
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(pts1_hat.rows(), 1);

    for (int i = 0; i < W.rows(); ++i) {
        W.block<1, 3>(i, 0) = pts1_hat.block<1, 3>(i, 0) / pts1_hat(i, 2);
        b.block<1, 1>(i, 0) = pts2_hat.block<1, 1>(i, 0) / pts2_hat(i, 2);
    }

    Eigen::Vector3d vec_solved = W.colPivHouseholderQr().solve(b);
    Eigen::Matrix3d HA = Eigen::Matrix3d::Identity();
    HA.block<1, 3>(0, 0) = vec_solved.transpose();

    H1 = HA * H2 * M;

    std::cout << "H1: " << std::endl;
    for (int i = 0; i < H1.rows(); ++i) {
        for (int j = 0; j < H1.cols(); ++j) {
            std::cout << H1(i, j) << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "H2: " << std::endl;
    for (int i = 0; i < H2.rows(); ++i) {
        for (int j = 0; j < H2.cols(); ++j) {
            std::cout << H2(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}