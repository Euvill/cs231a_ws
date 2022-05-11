#include <iostream>
#include <unordered_set>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "sfm_pipeline.hpp"

std::vector<Eigen::Matrix3d> fundamental_matrices;
std::vector<Eigen::MatrixXd> matches_subset;
std::vector<Eigen::MatrixXd> matches_fullset;

std::vector<Frame> frames;

double focal_length = 719.5459;

void load_point_txt(const std::string& str, std::vector<Eigen::MatrixXd>& matches_set) {
    std::ifstream file_;
    file_.open(str);
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
            matches_set.push_back(tmp_zero);

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
                        matches_set[index].resize(4, number.size());
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
                for (int i = 0; i < matches_set[index].cols(); ++i) {
                    matches_set[index](row_index ,i) = number[i];
                }
                number.clear();
                row_index = row_index + 1;
                flag_find_one_row = false;
            }
        }
    }

    file_.close();
}

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

    load_point_txt("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/matches_subset.txt", matches_subset);
    load_point_txt("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/dense_matches.txt", matches_fullset);
   
}

int main(void) {

    std::cout << "Load data ..." << std::endl;

    data_loaded();

    std::cout << "Load data success!" << std::endl;

    int N = matches_subset.size();

    std::cout << "Begin to process frames ..." << std::endl;

    for (int i = 0; i < N; ++i) {
        Frame frame(matches_subset[i], focal_length, fundamental_matrices[i], 480, 640, i);

        bundle_adjustment(frame, false);

        CamToRtMatrix(camera_parameter[i], frame.motion_[0]);
        CamToRtMatrix(camera_parameter[i + 1], frame.motion_[1]);

        frames.push_back(frame);
    }
    camera_parameter.clear();

    std::vector<Eigen::Matrix4d> cam_RTs;
    cam_RTs.push_back(frames[0].motion_[0]);
    for (int i = 0; i < N; ++i) {
        cam_RTs.push_back(cam_RTs[i] * frames[i].motion_[0] * frames[i].motion_[1]);
        RtMatrixToCam(cam_RTs[i], focal_length, 0 , 0);
    }
    RtMatrixToCam(cam_RTs[N], focal_length, 0 , 0);

    merged_frames(frames, cam_RTs);

    frames[0].match_points_.clear();

    std::cout << "Full set processing..." << std::endl;

    for (int index = 0; index < N; ++index) {

        if (index == 0) {
            for (int j = 0; j < matches_fullset[index].cols(); ++j) {
                std::vector<double> tmp{matches_fullset[index](0, j), 
                                        matches_fullset[index](1, j), 
                                        matches_fullset[index](2, j), 
                                        matches_fullset[index](3, j)};
                frames[0].match_points_.push_back(tmp);
            }
        } else {
            std::vector<std::vector<double>> match_points_;
            for (int j = 0; j < matches_fullset[index].cols(); ++j) {
                std::vector<double> tmp{matches_fullset[index](0, j), 
                                        matches_fullset[index](1, j), 
                                        matches_fullset[index](2, j), 
                                        matches_fullset[index](3, j)};
                match_points_.push_back(tmp);
            }

            std::unordered_set<int> hash_set;
            for (int i = 0; i < frames[0].match_points_.size(); ++i) {
                bool con_point = false;
                for (int j = 0; j < match_points_.size(); ++j) {
                    double diff_x = frames[0].match_points_[i][2 * index + 0] - match_points_[j][0];
                    double diff_y = frames[0].match_points_[i][2 * index + 1] - match_points_[j][1];
                    double tmp = fabs(diff_x) + fabs(diff_y);
                    if (tmp < 0.1) { 
                        frames[0].match_points_[i].push_back(match_points_[j][2]);
                        frames[0].match_points_[i].push_back(match_points_[j][3]);
                        hash_set.emplace(j);
                        con_point = true;
                        break;
                    }
                }
                if (!con_point) {
                    frames[0].match_points_[i].push_back(non_pix);
                    frames[0].match_points_[i].push_back(non_pix);
                }
            }
            for (int i = 0; i < match_points_.size(); ++i) {
                if(hash_set.count(i) == 0) {
                    std::vector<double> match_points;
                    for (int j = 0; j < index * 2; ++j)
                        match_points.push_back(non_pix);
                    match_points.push_back(match_points_[i][0]);
                    match_points.push_back(match_points_[i][1]);
                    match_points.push_back(match_points_[i][2]);
                    match_points.push_back(match_points_[i][3]);
                    frames[0].match_points_.push_back(match_points);
                }
            }
        }

        std::cout << "Set " << index << " processed." << std::endl;
    }

    std::cout << "Begin to triangulate." << std::endl;

    triangulate(frames[0].motion_, frames[0].match_points_, frames[0].K_, frames[0].structure_);

    std::cout << "Triangulation end." << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    cloud->width  = frames[0].structure_.size();
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    int index = 0;
   
    for (int j = 0; j < frames[0].structure_.size(); ++j) {

        cloud->points[index].x = frames[0].structure_[j][0];
        cloud->points[index].y = frames[0].structure_[j][1];
        cloud->points[index].z = frames[0].structure_[j][2];

        index = index + 1;
    }

    pcl::VoxelGrid<pcl::PointXYZ> sor;

    sor.setInputCloud(cloud);

    sor.setLeafSize(0.01f, 0.01f, 0.01f);
     
    sor.filter(*cloud_filtered);
     
    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
              << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

    pcl::io::savePCDFileASCII("output_downsampled.pcd", *cloud_filtered);

    std::cout << "Visualization." << std::endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Visualizer_Viewer")); 
	viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "cloud_filtered");		

    viewer->setBackgroundColor(0, 0, 0);	
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,  2, "cloud_filtered");		
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud_filtered");	

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}