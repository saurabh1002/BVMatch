#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <Eigen/Core>

int readPointCloud(std::vector<Eigen::Vector3d> &point_cloud, const std::string filename);
std::vector<std::string> pyListSortDir(std::string path);
void imagePadding(cv::Mat &img, int cor_x, int cor_y);
std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d> &frame,
                                             Eigen::Vector3d voxel_size);
void generateImage(std::vector<Eigen::Vector3d> &point_cloud, int x_max_ind, int y_max_ind, cv::Mat &mat_local_image);
