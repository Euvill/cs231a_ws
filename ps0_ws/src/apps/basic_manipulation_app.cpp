#include <iostream>
#include <memory>
#include <deque>
#include <string>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

void setImageLeftBlack(cv::Mat& image) {

    int cols = image.cols;
    int rows = image.rows;
  
    int imageArray_count=0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0;j < cols; j++) {
            if(j >= cols / 2) {
                image.at<cv::Vec3b>(i, j)[0] = 0;
                image.at<cv::Vec3b>(i, j)[1] = 0;
                image.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
}

void setImageRightBlack(cv::Mat& image) {

    int cols = image.cols;
    int rows = image.rows;
  
    int imageArray_count=0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0;j < cols; j++) {
            if(j < cols / 2) {
                image.at<cv::Vec3b>(i, j)[0] = 0;
                image.at<cv::Vec3b>(i, j)[1] = 0;
                image.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
}

void display_d(const Eigen::MatrixXd& d) {
    for (int i = 0; i < d.rows(); ++i) {
        for (int j = 0; j < d.cols(); ++j) {
            std::cout << d(i, j) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void display_i(const Eigen::MatrixXi& d) {
    for (int i = 0; i < d.rows(); ++i) {
        for (int j = 0; j < d.cols(); ++j) {
            std::cout << d(i, j) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(void) {

    Eigen::MatrixXd M(4, 3);
    M << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         0, 2, 2;
    Eigen::Vector3d a;
    a << 1, 1, 0;
    Eigen::Vector3d b;
    b << -1, 2, 5;
    Eigen::Vector4d c;
    c << 0, 2, 3, 2;

    auto aDotb = a.transpose() * b;
    display_d(aDotb);

    Eigen::Vector3d a_b = a.cwiseProduct(b);
    display_d(a_b);

    auto aDotbMa = a.transpose() * b * M * a;
    display_d(aDotbMa);

    auto aTMat = a.transpose().replicate<4, 1>();
    auto newM = M.cwiseProduct(aTMat);
    display_d(newM);

    cv::Mat image1 = cv::imread("/home/euvill/Desktop/ps0_ws/image1.jpeg", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread("/home/euvill/Desktop/ps0_ws/image2.jpeg", cv::IMREAD_COLOR);
    //cv::imshow("image1", image1);
    //cv::imshow("image2", image2);
    //cv::waitKey(0);

    cv::Mat image1_float;
    image1.convertTo(image1_float, CV_32FC3, 1 / 255.0);
    cv::Mat image2_float;
    image2.convertTo(image2_float, CV_32FC3, 1 / 255.0);
    //cv::imshow("image1_float", image1_float);
    //cv::imshow("image2_float", image2_float);
    //cv::waitKey(0);

    cv::Mat image_together = (image1 + image2) / 2;
    cv::Mat image_together_float;
    image_together.convertTo(image_together_float, CV_32FC3, 1 / 255.0);
    //cv::imshow("image_together_float", image_together_float);
    //cv::waitKey(0);

    cv::Mat image1_copy;
    image1.copyTo(image1_copy);
    setImageLeftBlack(image1_copy);
    cv::Mat image2_copy;
    image2.copyTo(image2_copy);
    setImageRightBlack(image2_copy);
    cv::Mat image_add = image1_copy + image2_copy;
    //cv::imshow("image_add", image_add);
    //cv::waitKey(0);

    cv::Mat m2e = cv::Mat::zeros(image1.rows, image1.cols, CV_8UC3);
    for (int i = 0; i < m2e.rows; ++i) {
        for (int j = 0; j < m2e.cols; ++j) {
            if(i % 2 == 0)
                m2e.at<cv::Vec3b>(i, j) = image1.at<cv::Vec3b>(i, j);
            else
                m2e.at<cv::Vec3b>(i, j) = image2.at<cv::Vec3b>(i, j);
        }
    }
    //cv::imshow("m2e", m2e);
    //cv::waitKey(0);

    cv::Mat image1_gray;
    cv::cvtColor(image1, image1_gray, cv::COLOR_BGR2GRAY);
    cv::Mat image2_gray;
    cv::cvtColor(image2, image2_gray, cv::COLOR_BGR2GRAY);
    Eigen::MatrixXi image1_gray_;
    cv2eigen(image1_gray, image1_gray_);
    Eigen::MatrixXi image2_gray_;
    cv2eigen(image2_gray, image2_gray_);
    Eigen::Vector2i odd_tmp;
    odd_tmp << 1, 
               0;
    Eigen::MatrixXi odd = odd_tmp.replicate<150, 300>();
    Eigen::Vector2i even_tmp;
    even_tmp << 0, 
                1;
    Eigen::MatrixXi even = even_tmp.replicate<150, 300>();
    Eigen::MatrixXi result = image1_gray_.cwiseProduct(odd) + 
                             image2_gray_.cwiseProduct(even);
    cv::Mat result_image;
    eigen2cv(result, result_image);
    result_image.convertTo(result_image, CV_8UC1);
    cv::imshow("result_image", result_image);
    cv::waitKey(0);

    image1_gray.convertTo(image1_gray, CV_32FC1);
    cv::Mat U, W, V;
    cv::SVD::compute(image1_gray,
                     W,  // singular value vector, CV_32F
                     U,  // left vector, CV_32F
                     V); // right vector's transpose, CV_32F
    std::cout << "W >> <" << W.rows << "," << W.cols << ">" << std::endl;
	std::cout << "U >> <" << U.rows << "," << U.cols << ">" << std::endl;
	std::cout << "V >> <" << V.rows << "," << V.cols << ">" << std::endl;
    cv::Mat U_1st_col = U.colRange(0, 1).clone();
    cv::Mat V_1st_row = V.rowRange(0, 1).clone();
    std::cout << "U_1st_col >> <" << U_1st_col.rows << "," << U_1st_col.cols << ">" << std::endl;
    std::cout << "V_1st_row >> <" << V_1st_row.rows << "," << V_1st_row.cols << ">" << std::endl;
    cv::Mat rank_1st_image = U_1st_col * W.at<float>(0, 0) * V_1st_row;
    rank_1st_image.convertTo(rank_1st_image, CV_8UC1);
    //cv::imshow("rank_1st_image", rank_1st_image);
    //cv::waitKey(0);
    
    cv::Mat U_20th_col = U.colRange(0, 20).clone();
    cv::Mat V_20th_row = V.rowRange(0, 20).clone();
    cv::Mat W_20th = cv::Mat::zeros(cv::Size(20, 20), CV_32FC1);
    for (int i = 0; i < 20; ++i) {
        W_20th.at<float>(i, i) = W.at<float>(i, i);
    }
    cv::Mat rank_20th_image = U_20th_col * W_20th * V_20th_row;
    rank_20th_image.convertTo(rank_20th_image, CV_8UC1);
    //cv::imshow("rank_20th_image", rank_20th_image);
    //cv::waitKey(0);

    return 0;
}