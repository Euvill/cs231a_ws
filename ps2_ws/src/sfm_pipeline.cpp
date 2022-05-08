#include "sfm_pipeline.hpp"

bool triangulate(const std::vector<Eigen::Matrix<double, 3, 4>>& motion,
                 const std::vector<Eigen::Vector4d>& match_points,
                 const Eigen::Matrix3d& K,
                       std::vector<std::vector<double>>& structure) {
    
    int num_points  = match_points.size();

    std::vector<Eigen::Matrix<double, 3, 4>> all_camera_matrices;

    for (int i = 0; i < 2; ++i) {
        Eigen::Matrix<double, 3, 4> tmp = K * motion[i];
        all_camera_matrices.push_back(tmp);
    }
    
    for (int i = 0; i < num_points; ++i) {
        
        Eigen::MatrixXd image_points; 

        image_points.resize(2, 2);

        image_points.block<1, 2>(0, 0) = Eigen::Vector2d(match_points[i](0), match_points[i](1)).transpose();
        image_points.block<1, 2>(1, 0) = Eigen::Vector2d(match_points[i](2), match_points[i](3)).transpose();

        Eigen::Vector3d estimated_3d_point;

        nonlinear_estimate_3d_point(image_points, all_camera_matrices, estimated_3d_point);

        std::vector<double> tmp{estimated_3d_point(0), estimated_3d_point(1), estimated_3d_point(2)};
        structure.push_back(tmp);
    }

    /*for (int i = 0; i < structure.size(); ++i) {
        std::cout << structure[i](0) << ", " << structure[i](1) << ", " << structure[i](2) << std::endl;
    }
    std::cout << std::endl;*/
}

bool bundle_adjustment(Frame& frame) {

     /*int num_cameras = frame.motion_.size();
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


   for (int k = 0; k < motion_angle_axis.size(); ++k) {
        for (int i = 0; i < motion_angle_axis[k].rows(); ++i) {
            for (int j = 0; j < motion_angle_axis[k].cols(); ++j) {
                std::cout << motion_angle_axis[k](i, j) << ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;*/

}