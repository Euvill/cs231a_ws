#include "reconstruction_alg.hpp"

bool readtxt2vec2d(const std::string &file, std::vector<Eigen::Vector2d> &vec) {
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

bool readtxt2vec3d(const std::string &file, std::vector<Eigen::Vector3d>& vec) {
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
            if(number.size() >= 3) 
                break;
            number.push_back(tmpp);
            ++index;
        }
        Eigen::Vector3d point(std::stod(number[0]), std::stod(number[1]), std::stod(number[2]));
        vec.push_back(point);
    }

    file_.close();
}


bool factorization_method(const std::vector<Eigen::Vector2d>& pts1,
                          const std::vector<Eigen::Vector2d>& pts2,
                          Eigen::MatrixXd& M, Eigen::MatrixXd& S) {
    
    if (pts1.size() != pts2.size()) {
        std::cout << "factorization_method error, the size of two points set must be equal." << std::endl;
        return false;
    }

    int N = pts1.size();

    Eigen::Vector2d sum_pts1 = Eigen::Vector2d::Zero();
    Eigen::Vector2d sum_pts2 = Eigen::Vector2d::Zero();
    for (int i = 0; i < N; ++i) {
        sum_pts1 += pts1[i];
        sum_pts2 += pts2[i];
    }
    Eigen::Vector2d centroid_1 = (1.0 / N) * sum_pts1;
    Eigen::Vector2d centroid_2 = (1.0 / N) * sum_pts2;
    
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(4, N);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d point1 = pts1[i] - centroid_1;
        Eigen::Vector2d point2 = pts2[i] - centroid_2;

        D.block<2, 1>(0, i) = point1;
        D.block<2, 1>(2, i) = point2;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU(); 
    Eigen::MatrixXd V = svd.matrixV(); // no transpose
    Eigen::MatrixXd singular = svd.singularValues();

    std::cout << "U shape: row = " << U.rows() << ", col = " << U.cols() << std::endl;
    std::cout << "V shape: row = " << V.rows() << ", col = " << V.cols() << std::endl;
    std::cout << "singular shape: row = " << singular.rows() << ", col = " << singular.cols() << std::endl;

    std::cout << "singular value: " << std::endl;
    for (int i = 0; i < singular.rows(); ++i) {
        for (int j = 0; j < singular.cols(); ++j) {
            std::cout << singular(i, j) << ",";
        }
        std::cout << std::endl;
    }

    M = U.block<4, 3>(0, 0);
    Eigen::Matrix3d singularMatrix = Eigen::Matrix3d::Identity();
    singularMatrix(0, 0) = singular(0, 0);
    singularMatrix(1, 1) = singular(1, 0);
    singularMatrix(2, 2) = singular(2, 0);

    Eigen::MatrixXd V_transpose = V.transpose();

    S = singularMatrix * V_transpose.block(0, 0, 3, N);

    std::cout << "Motion Matrix (include two camera's motion): " << std::endl;
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            std::cout << M(i, j) << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "Structure Matrix: " << std::endl;
    for (int i = 0; i < S.rows(); ++i) {
        for (int j = 0; j < S.cols(); ++j) {
            std::cout << S(i, j) << ",";
        }
        std::cout << std::endl;
    }
}

bool writevec3d2txt(const std::string &file, const Eigen::MatrixXd structure) {
    std::ofstream file_;
    file_.open(file);

    for(int i = 0; i < structure.cols(); ++i) {
        Eigen::Vector3d point = structure.block<3, 1>(0, i);
        file_ << point(0) << " " << point(1) << " " << point(2) << std::endl;
    }

    file_.close();
}