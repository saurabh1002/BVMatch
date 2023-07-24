
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

void match2Image(cv::Mat img1, cv::Mat img2, int max_x1, int max_y1, int max_x2, int max_y2);