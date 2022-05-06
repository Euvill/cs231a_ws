#include "triangulation_alg.hpp"

bool estimate_initial_RT(const Eigen::Matrix3d& E, 
                         std::vector<Eigen::Matrix<double, 3, 4>>& estimated_RT) {
    Eigen::Matrix3d Z;
    Z << 0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0,
         0.0, 0.0, 0.0;
    Eigen::Matrix3d W;
    W << 0.0,-1.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 0.0, 1.0;
    

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d M  = U * Z * U.transpose();
    Eigen::Matrix3d Q1 = U * W * V.transpose();
    Eigen::Matrix3d R1 = Q1.determinant() * 1.0 * Q1;

    Eigen::Matrix3d Q2 = U * W.transpose() * V.transpose();
    Eigen::Matrix3d R2 = Q2.determinant() * 1.0 * Q2;

    Eigen::Vector3d T1 = U.block<3, 1>(0, 2);
    Eigen::Vector3d T2 =-T1;    

    Eigen::Matrix<double, 3, 4> tmp1;
    tmp1.block<3, 3>(0, 0) = R1;
    tmp1.block<3, 1>(0, 3) = T1;

    Eigen::Matrix<double, 3, 4> tmp2;
    tmp2.block<3, 3>(0, 0) = R1;
    tmp2.block<3, 1>(0, 3) = T2;

    Eigen::Matrix<double, 3, 4> tmp3;
    tmp3.block<3, 3>(0, 0) = R2;
    tmp3.block<3, 1>(0, 3) = T1;

    Eigen::Matrix<double, 3, 4> tmp4;
    tmp4.block<3, 3>(0, 0) = R2;
    tmp4.block<3, 1>(0, 3) = T2;

    estimated_RT.push_back(tmp1);
    estimated_RT.push_back(tmp2);
    estimated_RT.push_back(tmp3);
    estimated_RT.push_back(tmp4);

    return true;
}

bool linear_estimate_3d_point(const Eigen::MatrixXd& image_points, 
                              const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                              Eigen::Vector3d& estimated_3d_point) {
    int N = image_points.rows();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * N, 4);

    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<double, 1, 2> pi = image_points.block<1, 2>(i, 0);
        Eigen::Matrix<double, 3, 4> Mi = camera_matrices[i];

        Eigen::Matrix<double, 1, 4> Aix = pi(0, 0) * Mi.block<1, 4>(2, 0) - Mi.block<1, 4>(0, 0);
        Eigen::Matrix<double, 1, 4> Aiy = pi(0, 1) * Mi.block<1, 4>(2, 0) - Mi.block<1, 4>(1, 0);
        
        A.block<1, 4>(2 * i, 0)     = Aix;
        A.block<1, 4>(2 * i + 1, 0) = Aiy;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXd V = svd.matrixV();

    Eigen::Vector4d P_homo = V.block<4, 1>(0, V.cols() - 1);
    P_homo = P_homo / P_homo(3);

    estimated_3d_point = Eigen::Vector3d(P_homo(0), P_homo(1), P_homo(2));

    return true;
}

bool reprojection_error(const Eigen::Vector3d& point_3d, 
                        const Eigen::MatrixXd& image_points, 
                        const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                              Eigen::MatrixXd& error) {
    int N = image_points.rows();
    Eigen::Vector4d point_3d_homo(point_3d(0), point_3d(1), point_3d(2), 1.0);

    std::vector<Eigen::Vector2d> error_set;

    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<double, 1, 2> pi = image_points.block<1, 2>(i, 0);
        Eigen::Matrix<double, 3, 4> Mi = camera_matrices[i];
        Eigen::Vector3d Yi = Mi * point_3d_homo;

        Eigen::Vector2d pi_prime = (1.0 / Yi(2)) * Eigen::Vector2d(Yi(0), Yi(1));
        Eigen::Vector2d  error_i = pi_prime - pi.transpose();

        error_set.push_back(error_i);
    }

    error.resize(error_set.size() * 2, 1);
    for (int i = 0; i < error_set.size(); ++i) {
        error(2 * i, 0)     = error_set[i](0);
        error(2 * i + 1, 0) = error_set[i](1);
    }

    return true;
}

bool jacobian(const Eigen::Vector3d& point_3d, 
              const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                    Eigen::MatrixXd& J) {
    int N = camera_matrices.size();
    J.resize(2 * N, 3);
    Eigen::Vector4d point_3d_homo(point_3d(0), point_3d(1), point_3d(2), 1.0);

    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<double, 3, 4> Mi = camera_matrices[i];
        Eigen::Vector3d pi_homo = Mi * point_3d_homo;

        Eigen::Matrix<double, 1, 3> Mi_row_1 = Mi.block<1, 3>(0, 0);
        Eigen::Matrix<double, 1, 3> Mi_row_2 = Mi.block<1, 3>(1, 0);
        Eigen::Matrix<double, 1, 3> Mi_row_3 = Mi.block<1, 3>(2, 0);

        Eigen::Matrix<double, 1, 3> Jix = (pi_homo(2) * Mi_row_1 - pi_homo(0) * Mi_row_3) / pow(pi_homo(2), 2.0);
        Eigen::Matrix<double, 1, 3> Jiy = (pi_homo(2) * Mi_row_2 - pi_homo(1) * Mi_row_3) / pow(pi_homo(2), 2.0);

        J.block<1, 3>(2 * i, 0)     = Jix;
        J.block<1, 3>(2 * i + 1, 0) = Jiy;
    }

    return true;
}

bool nonlinear_estimate_3d_point(const Eigen::MatrixXd& image_points,
                                 const std::vector<Eigen::Matrix<double, 3, 4>>& camera_matrices,
                                       Eigen::Vector3d& estimated_3d_point) {

    linear_estimate_3d_point(image_points, camera_matrices, estimated_3d_point);

    for (int i = 0; i < 10; ++i) {
        Eigen::MatrixXd error;
        Eigen::MatrixXd J;
        reprojection_error(estimated_3d_point, image_points, camera_matrices, error);
        jacobian(estimated_3d_point, camera_matrices, J);
        estimated_3d_point = estimated_3d_point - (J.transpose() * J).inverse() * J.transpose() * error;
    }

}

Eigen::Vector3d camera1tocamera2(const Eigen::Vector4d& P_homo, 
                                 const Eigen::Matrix<double, 3, 4>& RT) {
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    
    A.block<3, 4>(0, 0) = RT;
    A.block<1, 4>(3, 0) = Eigen::Vector4d(0, 0, 0, 1).transpose();

    Eigen::Vector4d P_prime_homo = A * P_homo;
    P_prime_homo = P_prime_homo / P_prime_homo(3);
    
    return Eigen::Vector3d(P_prime_homo(0), P_prime_homo(1), P_prime_homo(2));
}

bool estimate_RT_from_E(const Eigen::Matrix3d& E, 
                        const Eigen::MatrixXd& image_points,
                        const Eigen::Matrix3d& K, 
                              Eigen::Matrix<double, 3, 4>& estimated_RT) {
    
    std::vector<Eigen::Matrix<double, 3, 4>> estimated_RT_vec;

    estimate_initial_RT(E, estimated_RT_vec);

    Eigen::Matrix<double, 3, 4> I0 = Eigen::Matrix<double, 3, 4>::Zero();
    I0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    std::vector<Eigen::Matrix<double, 3, 4>> camera_matrices;
    Eigen::Matrix<double, 3, 4> tmp3x4 = K * I0;
    camera_matrices.push_back(tmp3x4);
    tmp3x4 = Eigen::Matrix<double, 3, 4>::Zero();
    camera_matrices.push_back(tmp3x4);

    std::vector<int> count = {0, 0, 0, 0};

    for(int i = 0; i < estimated_RT_vec.size(); ++i) {
        Eigen::Matrix<double, 3, 4> RTi = estimated_RT_vec[i];
        Eigen::Matrix<double, 3, 4> M2i = K * RTi;
        camera_matrices[1] = M2i;
        for(int j = 0; j < image_points.rows(); j = j + 2) {
            Eigen::Vector3d pointj_3d;
            nonlinear_estimate_3d_point(image_points.block<2, 2>(j, 0), camera_matrices, pointj_3d);
            Eigen::Vector4d pointj_3d_homo = Eigen::Vector4d(pointj_3d(0), pointj_3d(1), pointj_3d(2), 1.0);
            Eigen::Vector3d pointj_prime = camera1tocamera2(pointj_3d_homo, RTi);
            if(pointj_3d_homo(2) > 0 && pointj_prime(2) > 0) {
                count[i] += 1;
            }
        }
    }

    int maxIndex = std::max_element(count.begin(), count.end()) - count.begin(); 

    estimated_RT = estimated_RT_vec[maxIndex];

    return true;
}