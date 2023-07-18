
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string filename);
std::vector<std::string> pyListSortDir(std::string path);
void generateBVImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, int x_max_ind, int y_max_ind, cv::Mat &mat_local_image);

int max_global_x_ind;
int max_global_y_ind;

void imagePadding(cv::Mat &img, int cor_x, int cor_y);
void match2Image(cv::Mat img1, cv::Mat img2, int max_x1, int max_y1, int max_x2, int max_y2);