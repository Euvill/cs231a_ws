#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "sfm_pipeline.hpp"

std::vector<Eigen::Matrix3d> fundamental_matrices;
std::vector<Eigen::MatrixXd> matches_subset;

std::vector<Frame> frames;

double focal_length = 719.5459;

void data_loaded(void) {

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
    f_tmp4 << -2.66322904e-08,  3.22928035e-07, -1.79473476e-04,
               1.98701325e-06, -4.02694866e-07,  6.93923582e-03,
              -5.76424460e-04, -7.13155017e-03, -3.92654749e-02;
    fundamental_matrices.push_back(f_tmp1);
    fundamental_matrices.push_back(f_tmp2);
    fundamental_matrices.push_back(f_tmp3);
    fundamental_matrices.push_back(f_tmp4);

    std::ifstream file_;
    file_.open("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/dense_matches.txt");
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

            if(str.size() != 0) {
                number.push_back(std::stod(str));
                str.clear();
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

    std::cout << "Load data ..." << std::endl;

    data_loaded();

    std::cout << "Load data success!" << std::endl;

    int N = matches_subset.size();

    int point3d_size = 0;

    std::cout << "Begin to process frames ..." << std::endl;

    for (int i = 0; i < N; ++i) {
        Frame frame(matches_subset[i], focal_length, fundamental_matrices[i], 480, 640, i);

        point3d_size += frame.structure_.size();

        /*bundle_adjustment(frame, false);

        CamToRtMatrix(camera_parameter[i], frame.motion_[0]);
        CamToRtMatrix(camera_parameter[i + 1], frame.motion_[1]);*/

        frames.push_back(frame);
    }
    //camera_parameter.clear();

    std::vector<Eigen::Matrix4d> cam_RTs;
    cam_RTs.push_back(Eigen::Matrix4d::Identity());
    for (int i = 0; i < N; ++i) {
        cam_RTs.push_back(cam_RTs[i] * frames[i].motion_[0] * frames[i].motion_[1]);
        RtMatrixToCam(cam_RTs[i], focal_length, 0 , 0);
    }
    RtMatrixToCam(cam_RTs[N], focal_length, 0 , 0);

    for (int i = 0; i < N; ++i) {
        frames[i].motion_[0] = cam_RTs[i];
        frames[i].motion_[1] = cam_RTs[i + 1];
        frames[i].structure_.clear();
        triangulate(frames[i].motion_, frames[i].match_points_, frames[i].K_, frames[i].structure_);
    }    

    ceres::Problem problem;

    for (int k = 0; k < frames.size(); ++k) {

        for (int i = 0; i < frames[k].camera_index_.size(); ++i){
            
            for (int j = 0; j < frames[k].match_points_.size(); ++j) {

                ceres::CostFunction *cost_function;

                cost_function = SnavelyReprojectionError::Create(frames[k].match_points_[j][2 * i + 0], 
                                                                 frames[k].match_points_[j][2 * i + 1]);

                ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

                double *camera = camera_parameter[frames[k].camera_index_[i]].get();

                double *point3d = new double[3];
                memcpy(point3d, &frames[k].structure_[j][0], 3 * sizeof(double));

                problem.AddResidualBlock(cost_function, loss_function, camera, point3d);
            }
        }
    }

    ceres::Solver::Options options;
    options.logging_type = ceres::SILENT;
    
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n"; 
    
    for (int m = 0; m < N; ++m) {
        CamToRtMatrix(camera_parameter[m],   frames[m].motion_[0]);
        CamToRtMatrix(camera_parameter[m+1], frames[m].motion_[1]);
    }

    std::cout << "Frames processed!" << std::endl;
 
    /*for (int m = 0; m < N; ++m) {
        for (int k = 0; k < frames[m].motion_.size(); ++k) {
            for (int i = 0; i < frames[m].motion_[k].rows(); ++i) {
                for (int j = 0; j < frames[m].motion_[k].cols(); ++j) {
                    std::cout << frames[m].motion_[k](i, j) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;  
        }
    }

      for (int k = 0; k < cam_RTs.size(); ++k) {
        for (int i = 0; i < cam_RTs[k].rows(); ++i) {
            for (int j = 0; j < cam_RTs[k].cols(); ++j) {
                std::cout << cam_RTs[k](i, j) << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;  
    }*/

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    cloud->width  = point3d_size;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    std::cout << "point3d_size: " << point3d_size << std::endl;

    int index = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < frames[i].structure_.size(); ++j) {

            cloud->points[index].x = frames[i].structure_[j][0];
            cloud->points[index].y = frames[i].structure_[j][1];
            cloud->points[index].z = frames[i].structure_[j][2];

            std::cout << index << ", "
                      << cloud->points[index].x << ", "
                      << cloud->points[index].y << ", "
                      << cloud->points[index].z << std::endl;

            index = index + 1;
        }
    }

    std::cout << "Visualization." << std::endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Visualizer_Viewer")); 
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample_cloud");		

    viewer->setBackgroundColor(0, 0, 0);		//窗口背景色，默认[0,0,0]，范围[0~255,0~255,0~255]
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample_cloud");			//设置点的大小，默认 1
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "sample_cloud");	//设置点云显示的颜色，rgb 在 [0,1] 范围

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}