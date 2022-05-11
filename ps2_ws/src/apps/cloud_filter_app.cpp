#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common_headers.h>
#include <boost/thread/thread.hpp>
 
int main(void) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
 
	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ>("../data/output_downsampled.pcd", *cloud);
 
	std::cerr << "Cloud before filtering: " << std::endl;
	std::cerr << *cloud << std::endl;
    
	pcl::visualization::CloudViewer viewerori("ori cloud");
	viewerori.showCloud(cloud, "ori cloud");
	while (!viewerori.wasStopped())
	{
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}
 
	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(5);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_filtered);
 
	std::cerr << "Cloud after filtering: " << std::endl;
	std::cerr << *cloud_filtered << std::endl;
 
	pcl::visualization::CloudViewer viewer("after filtered");
	viewer.showCloud(cloud_filtered, "cloud filtered0");
	while (!viewer.wasStopped())
	{
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}
 
	sor.setNegative(true);
	sor.filter(*cloud_filtered);
 
 
	pcl::visualization::CloudViewer viewer1("points be filtered");
	viewer1.showCloud(cloud_filtered, "cloud filtered1");
	while (!viewer1.wasStopped())
	{
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}
 
	return (0);
}