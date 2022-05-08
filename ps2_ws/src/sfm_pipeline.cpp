#include "sfm_pipeline.hpp"

bool triangulate(const std::vector<std::vector<int>>& match_idx, 
                 const std::vector<Eigen::Matrix<double, 3, 4>>& motion,
                 const Eigen::MatrixXd& match_points,
                 const Eigen::Matrix3d& K,
                       std::vector<Eigen::Vector3d>& structure) {
    
    int num_cameras = match_idx.size();
    int num_points  = match_idx[0].size();

    std::vector<Eigen::Matrix<double, 3, 4>> all_camera_matrices;

    for (int i = 0; i < num_cameras; ++i) {
        Eigen::Matrix<double, 3, 4> tmp = K * motion[i];
        all_camera_matrices.push_back(tmp);
    }

    int rows = 0;
    
    for (int i = 0; i < num_points; ++i) {
        
        std::vector<int> valid_cameras;
        
        for (int k = 0; k < match_idx.size(); ++k) {
            if(match_idx[k][i] >= 0)
                valid_cameras.push_back(k);
        }

        if(valid_cameras.empty()) {
            continue;
        }

        std::vector<Eigen::Matrix<double, 3, 4>> camera_matrices;

        for (int k = 0; k < valid_cameras.size(); ++k) {
            camera_matrices.push_back(all_camera_matrices[valid_cameras[k]]);
        }

        Eigen::MatrixXd image_points; 

        image_points.resize(valid_cameras.size(), 2);

        for (int k = 0; k < valid_cameras.size(); ++k) {
            image_points.block<1, 2>(k, 0) = match_points.block<1, 2>(match_idx[k][i], 0);
        }

        Eigen::Vector3d estimated_3d_point;

        nonlinear_estimate_3d_point(image_points, camera_matrices, estimated_3d_point);

        structure.push_back(estimated_3d_point);
    }

    /*for (int i = 0; i < structure.size(); ++i) {
        std::cout << structure[i](0) << ", " << structure[i](1) << ", " << structure[i](2) << std::endl;
    }
    std::cout << std::endl;*/
}

bool bundle_adjustment(Frame& frame) {

    int num_cameras = frame.motion_.size();
    int num_points  = frame.structure_.size();
    int num_observations = frame.match_idx_[0].size();

    std::vector<Eigen::Matrix<double, 3, 2>> motion_angle_axis;
    
    for (int i = 0; i < num_cameras; ++i) {
        Eigen::Matrix<double, 3, 2> tmp = Eigen::Matrix<double, 3, 2>::Zero();
        Eigen::AngleAxisd rotation_vector;
        rotation_vector.fromRotationMatrix(frame.motion_[i].block<3, 3>(0, 0));
        tmp.block<3, 1>(0, 0) = (rotation_vector.axis() * rotation_vector.angle()).transpose();
        tmp.block<3, 1>(0, 1) = frame.motion_[i].block<3, 1>(0, 3);

        motion_angle_axis.push_back(tmp);
    }


    /*for (int k = 0; k < motion_angle_axis.size(); ++k) {
        for (int i = 0; i < motion_angle_axis[k].rows(); ++i) {
            for (int j = 0; j < motion_angle_axis[k].cols(); ++j) {
                std::cout << motion_angle_axis[k](i, j) << ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;*/

}