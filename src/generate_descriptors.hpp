#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string filename);
std::vector<std::string> pyListSortDir(std::string path);
void imagePadding(cv::Mat &img, int cor_x, int cor_y);

void generateImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, int &x_max_ind, int &y_max_ind, cv::Mat &mat_local_image);
