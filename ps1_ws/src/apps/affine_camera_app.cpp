#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>

bool readtxt2vec(const std::string& file, std::vector<Eigen::Vector2d>& vec){
    std::ifstream file_;
    file_.open(file);

    std::string tmp;
    while(std::getline(file_, tmp)) {
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
        Eigen::Vector2d point(std::stod(number[0]), std::stod(number[1]));
        vec.push_back(point);
    }

    file_.close();
}

void TwoDimpointsNormalize(const std::vector<Eigen::Vector2d>& vKeys, 
                                 std::vector<Eigen::Vector3d>& vNormalizedPoints, 
                                              Eigen::MatrixXd& T) {
    // the total number of points
    const int N = vKeys.size(); 
    vNormalizedPoints.resize(N);

    double meanX = 0;
    double meanY = 0;
    for(int i = 0; i < N; ++i) {
        meanX += vKeys[i](0);
        meanY += vKeys[i](1);
    }
    meanX = meanX / N; // the mean of x-axis
    meanY = meanY / N; // the mean of y-axis

    float meanDevX = 0;
    float meanDevY = 0;
    for(int i = 0; i < N; ++i) {
        vNormalizedPoints[i](0) = vKeys[i](0) - meanX;
        vNormalizedPoints[i](1) = vKeys[i](1) - meanY;
        meanDevX += fabs(vNormalizedPoints[i](0));
        meanDevY += fabs(vNormalizedPoints[i](1));
    }
    meanDevX = meanDevX / N; 
    meanDevY = meanDevY / N;

    float sX = 1.0 / meanDevX;
    float sY = 1.0 / meanDevY;

    for(int i = 0; i < N; ++i) {
        vNormalizedPoints[i](0) = vNormalizedPoints[i](0) * sX; 
        vNormalizedPoints[i](1) = vNormalizedPoints[i](1) * sY;
        vNormalizedPoints[i](2) = 1.0;
    }

    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0       1   |
    T.setIdentity(3, 3);
    T(0,0) = sX;
    T(1,1) = sY;
    T(0,2) = -meanX * sX;
    T(1,2) = -meanY * sY;
}


void ThreeDimpointsNormalize(const std::vector<Eigen::Vector3d>& vKeys, 
                                   std::vector<Eigen::Vector4d>& vNormalizedPoints, 
                                                Eigen::MatrixXd& T) {
    // the total number of points
    const int N = vKeys.size(); 
    vNormalizedPoints.resize(N);

    double meanX = 0;
    double meanY = 0;
    double meanZ = 0;
    for(int i = 0; i < N; ++i) {
        meanX += vKeys[i](0);
        meanY += vKeys[i](1);
        meanZ += vKeys[i](2);
    }
    meanX = meanX / N; // the mean of x-axis
    meanY = meanY / N; // the mean of y-axis
    meanY = meanZ / N; // the mean of z-axis

    double meanDevX = 0;
    double meanDevY = 0;
    double meanDevZ = 0;
    for(int i = 0; i < N; ++i) {
        vNormalizedPoints[i](0) = vKeys[i](0) - meanX;
        vNormalizedPoints[i](1) = vKeys[i](1) - meanY;
        vNormalizedPoints[i](2) = vKeys[i](2) - meanZ;
        meanDevX += fabs(vNormalizedPoints[i](0));
        meanDevY += fabs(vNormalizedPoints[i](1));
        meanDevZ += fabs(vNormalizedPoints[i](2));
    }
    meanDevX = meanDevX / N; 
    meanDevY = meanDevY / N;
    meanDevZ = meanDevZ / N;

    float sX = 1.0 / meanDevX;
    float sY = 1.0 / meanDevY;
    float sZ = 1.0 / meanDevZ;

    for(int i = 0; i < N; ++i) {
        vNormalizedPoints[i](0) = vNormalizedPoints[i](0) * sX; 
        vNormalizedPoints[i](1) = vNormalizedPoints[i](1) * sY;
        vNormalizedPoints[i](2) = vNormalizedPoints[i](2) * sZ;
        vNormalizedPoints[i](3) = 1.0;
    }

    // |sX  0   0  -meanx*sX|
    // |0   sY  0  -meany*sY|
    // |0   0   sZ -meanz*sZ|
    // |0   0   0      1    |
    T.setIdentity(4, 4);
    T(0,0) = sX;
    T(1,1) = sY;
    T(2,2) = sZ;
    T(0,3) = -meanX * sX;
    T(1,3) = -meanY * sY;
    T(2,3) = -meanZ * sZ;
}

void solve_pinv(const Eigen::MatrixXd& A, Eigen::MatrixXd& pinv_A) {

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = row > col ? col : row;
    
    pinv_A = Eigen::MatrixXd::Zero(col,row);
    
    Eigen::MatrixXd singularValues_inv = svd.singularValues();
    Eigen::MatrixXd singularValues_inv_mat = Eigen::MatrixXd::Zero(col, row);
    
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else 
            singularValues_inv(i) = 0;
    }
    
    for (long i = 0; i < k; ++i) {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }

    pinv_A=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());
}

void compute_camera_matrix(const std::vector<Eigen::Vector2d>& front_image_points, 
                           const std::vector<Eigen::Vector2d>& back_image_points, 
                           const std::vector<Eigen::Vector3d>& front_real_points,
                           const std::vector<Eigen::Vector3d>& back_real_points, 
                                              Eigen::MatrixXd& camera_matrix) {

    std::vector<Eigen::Vector2d> TwoDim_vKeys;
    std::vector<Eigen::Vector3d> TwoDim_vNormalizedPoints;
    Eigen::MatrixXd TwoDim_Ts;
    for (int i = 0; i < front_image_points.size(); ++i) {
        TwoDim_vKeys.push_back(front_image_points[i]);
    }
    for (int i = 0; i < back_image_points.size(); ++i) {
        TwoDim_vKeys.push_back(back_image_points[i]);
    }
    TwoDimpointsNormalize(TwoDim_vKeys, TwoDim_vNormalizedPoints, TwoDim_Ts);

    std::vector<Eigen::Vector3d> ThreeDim_vKeys;
    std::vector<Eigen::Vector4d> ThreeDim_vNormalizedPoints;
    Eigen::MatrixXd ThreeDim_Ts;
    for (int i = 0; i < front_real_points.size(); ++i) {
        ThreeDim_vKeys.push_back(front_real_points[i]);
    }
    for (int i = 0; i < back_real_points.size(); ++i) {
        ThreeDim_vKeys.push_back(back_real_points[i]);
    }
    ThreeDimpointsNormalize(ThreeDim_vKeys, ThreeDim_vNormalizedPoints, ThreeDim_Ts);

    int index = 0;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(48, 8);
    for (int i = 0; i < 2 * ThreeDim_vNormalizedPoints.size(); i = i + 2) {
        A.block<1, 4>(i + 0, 0) = ThreeDim_vNormalizedPoints[index].transpose();
        A.block<1, 4>(i + 1, 4) = ThreeDim_vNormalizedPoints[index].transpose();
        index = index + 1;
    }
    index = 0;
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(48, 1);
    for (int i = 0; i < 48; i = i + 2) {
        b(i + 0, 0) = TwoDim_vNormalizedPoints[index](0);
        b(i + 1, 0) = TwoDim_vNormalizedPoints[index](1);
        index = index + 1;
    }

    Eigen::MatrixXd pinv_A;
    solve_pinv(A, pinv_A);
    Eigen::MatrixXd p8 = pinv_A * b;

    Eigen::MatrixXd Normalized_PA = Eigen::MatrixXd::Zero(3, 4);
    Eigen::Vector4d one_vector(0, 0, 0, 1);
    Normalized_PA.block<1, 4>(0, 0) = p8.block<4, 1>(0, 0).transpose();
    Normalized_PA.block<1, 4>(1, 0) = p8.block<4, 1>(4, 0).transpose();
    Normalized_PA.block<1, 4>(2, 0) = one_vector.transpose();

    camera_matrix = TwoDim_Ts.inverse() * Normalized_PA * ThreeDim_Ts;

    std::cout << "Camera Matrix is: " << std::endl;

    for (int i = 0; i < camera_matrix.rows(); ++i) {
        for (int j = 0; j < camera_matrix.cols(); ++j) {
            std::cout << camera_matrix(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

double rms_error(const std::vector<Eigen::Vector2d>& front_image_points, 
                 const std::vector<Eigen::Vector2d>& back_image_points, 
                 const std::vector<Eigen::Vector3d>& front_real_points,
                 const std::vector<Eigen::Vector3d>& back_real_points, 
                 const Eigen::MatrixXd& camera_matrix) {

    std::vector<Eigen::Vector4d> real_points;
    for (int i = 0; i < front_real_points.size(); ++i) {
        Eigen::Vector4d real_point;
        real_point(0) = front_real_points[i](0);
        real_point(1) = front_real_points[i](1);
        real_point(2) = front_real_points[i](2);
        real_point(3) = 1.0;
        real_points.push_back(real_point);
    }
    for (int i = 0; i < back_real_points.size(); ++i) {
        Eigen::Vector4d real_point;
        real_point(0) = back_real_points[i](0);
        real_point(1) = back_real_points[i](1);
        real_point(2) = back_real_points[i](2);
        real_point(3) = 1.0;
        real_points.push_back(real_point);
    }

    std::vector<Eigen::Vector3d> image_points;
    for (int i = 0; i < front_image_points.size(); ++i) {
        Eigen::Vector3d image_point;
        image_point(0) = front_image_points[i](0);
        image_point(1) = front_image_points[i](1);
        image_point(2) = 1.0;
        image_points.push_back(image_point);
    }
    for (int i = 0; i < back_image_points.size(); ++i) {
        Eigen::Vector3d image_point;
        image_point(0) = back_image_points[i](0);
        image_point(1) = back_image_points[i](1);
        image_point(2) = 1.0;
        image_points.push_back(image_point);
    }

    double rmse = 0;

    for (int i = 0; i < image_points.size(); ++i) {
        Eigen::Vector3d pred_x;
        pred_x = camera_matrix * real_points[i];
        rmse = rmse + sqrt(pow(pred_x(0) - image_points[i](0), 2) + pow(pred_x(1) - image_points[i](1), 2));
    }

    return rmse / image_points.size();
}

int main(void) {

    std::string front_image_file = "../data/front_image.txt";
    std::string back_image_file  = "../data/back_image.txt";
    std::string real_xy_file     = "../data/real_XY.txt";

    std::vector<Eigen::Vector2d> front_image_points;
    std::vector<Eigen::Vector2d> back_image_points;
    std::vector<Eigen::Vector2d> real_xy_points;

    readtxt2vec(front_image_file, front_image_points);
    readtxt2vec(back_image_file, back_image_points);
    readtxt2vec(real_xy_file, real_xy_points);

    Eigen::MatrixXd camera_matrix(3, 4);

    std::vector<Eigen::Vector3d> front_real_points;
    std::vector<Eigen::Vector3d> back_real_points;

    for (int i = 0; i < real_xy_points.size(); ++i) {
        Eigen::Vector3d front_point(real_xy_points[i](0), real_xy_points[i](1), 0.0);
        Eigen::Vector3d  back_point(real_xy_points[i](0), real_xy_points[i](1), 150.0);
        front_real_points.push_back(front_point);
        back_real_points.push_back(back_point);
    }

    compute_camera_matrix(front_image_points, 
                          back_image_points, 
                          front_real_points, 
                          back_real_points, 
                          camera_matrix);

    std::cout << "rmse error: "
              << rms_error(front_image_points,
                           back_image_points,
                           front_real_points,
                           back_real_points,
                           camera_matrix)
              << std::endl;

    return 0;
}