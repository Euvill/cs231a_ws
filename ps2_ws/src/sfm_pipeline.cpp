#include "sfm_pipeline.hpp"

std::vector<std::shared_ptr<double>> camera_parameter;

bool triangulate(const std::vector<Eigen::Matrix4d>& motion,
                 const std::vector<std::vector<double>>& match_points,
                 const Eigen::Matrix3d& K,
                       std::vector<std::vector<double>>& structure) {
    
    int num_points  = match_points.size();

    std::vector<Eigen::Matrix<double, 3, 4>> all_camera_matrices;

    for (int i = 0; i < 2; ++i) {
        Eigen::Matrix<double, 3, 4> tmp = K * motion[i].block<3, 4>(0, 0);
        all_camera_matrices.push_back(tmp);
    }
    
    for (int i = 0; i < num_points; ++i) {
        
        Eigen::MatrixXd image_points; 

        image_points.resize(2, 2);

        image_points.block<1, 2>(0, 0) = Eigen::Vector2d(match_points[i][0], match_points[i][1]).transpose();
        image_points.block<1, 2>(1, 0) = Eigen::Vector2d(match_points[i][2], match_points[i][3]).transpose();

        Eigen::Vector3d estimated_3d_point;

        nonlinear_estimate_3d_point(image_points, all_camera_matrices, estimated_3d_point);

        std::vector<double> tmp{estimated_3d_point(0), estimated_3d_point(1), estimated_3d_point(2)};
        structure.push_back(tmp);
    }
}

bool bundle_adjustment(Frame& frame, bool show) {

    ceres::Problem problem;

    int num_cameras = frame.motion_.size();

    for (int i = 0; i < num_cameras; ++i) {

        for (int j = 0; j < frame.match_points_.size(); ++j) {

            ceres::CostFunction *cost_function;

            cost_function = SnavelyReprojectionError::Create(frame.match_points_[j][2 * i + 0], frame.match_points_[j][2 * i + 1]);

            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

            double *camera = camera_parameter[frame.camera_index_[i]].get();

            double *point3d = new double[3];
            memcpy(point3d, &frame.structure_[j][0], 3 * sizeof(double));

            problem.AddResidualBlock(cost_function, loss_function, camera, point3d);
        }
    }

    ceres::Solver::Options options;
    options.logging_type = ceres::SILENT;

    if (show) {
        std::cout << "BA problem file loaded..." << std::endl;
        std::cout << "BA problem have " << num_cameras << " cameras and " << frame.structure_.size() << " points. " << std::endl;
        std::cout << "Forming " << frame.match_points_.size() << " observations. " << std::endl;

        std::cout << "Solving ceres BA ... " << std::endl;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    }

    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (show) {    
        std::cout << summary.FullReport() << "\n"; 
    }
}

bool ResetCamVector(int index) {
    
    double* cam = camera_parameter[index].get();

    cam[0] = 0; 
    cam[1] = 0; 
    cam[2] = 0; 
    cam[3] = 0; 
    cam[4] = 0; 
    cam[5] = 0; 

}

bool RtMatrixToCam(const Eigen::Matrix4d& RT, 
                   const double focal_length, const double r1, const double r2) {
        
    Eigen::AngleAxisd rotation_vector;
        
    rotation_vector.fromRotationMatrix(RT.block<3, 3>(0, 0));
        
    Eigen::Vector3d tmpr = rotation_vector.axis() * rotation_vector.angle();
    Eigen::Vector3d tmpt = RT.block<3, 1>(0, 3);

    std::shared_ptr<double> cam(new double[9], std::default_delete<double[]>());

    cam.get()[0] = tmpr(0); 
    cam.get()[1] = tmpr(1); 
    cam.get()[2] = tmpr(2); 
    cam.get()[3] = tmpt(0); 
    cam.get()[4] = tmpt(1); 
    cam.get()[5] = tmpt(2); 
    cam.get()[6] = focal_length; 
    cam.get()[7] = r1; 
    cam.get()[8] = r2; 

    camera_parameter.push_back(cam);
}

bool CamToRtMatrix(std::shared_ptr<double> cam, Eigen::Matrix4d& RT) {

    Eigen::Vector3d axis_angle(cam.get()[0], cam.get()[1], cam.get()[2]);

    double axis_angle_0_2 = pow(axis_angle(0), 2.0);
    double axis_angle_1_2 = pow(axis_angle(1), 2.0);
    double axis_angle_2_2 = pow(axis_angle(2), 2.0);

    double angle  = sqrt(axis_angle_0_2 + axis_angle_1_2 + axis_angle_2_2);

    double angle2 = pow(angle, 2.0);
    double tmp0 = sqrt(axis_angle_0_2 / angle2);
    double tmp1 = sqrt(axis_angle_1_2 / angle2);
    double tmp2 = sqrt(axis_angle_2_2 / angle2);

    tmp0 = axis_angle(0) < 0 ? -tmp0 : tmp0;
    tmp1 = axis_angle(1) < 0 ? -tmp1 : tmp1;
    tmp2 = axis_angle(2) < 0 ? -tmp2 : tmp2;

    axis_angle = Eigen::Vector3d(tmp0, tmp1, tmp2);

    Eigen::AngleAxisd rotation_vector(angle,axis_angle);

    RT.block<3, 3>(0, 0) = rotation_vector.matrix();
    RT.block<3, 1>(0, 3) = Eigen::Vector3d(cam.get()[3], cam.get()[4], cam.get()[5]);
}