
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// custumed functions, may be faster
#define _DBL_EPSILON 2.2204460492503131e-16f
#define atan2_p1 (0.9997878412794807f * 57.29577951308232f)
#define atan2_p3 (-0.3258083974640975f * 57.29577951308232f)
#define atan2_p5 (0.1555786518463281f * 57.29577951308232f)
#define atan2_p7 (-0.04432655554792128f * 57.29577951308232f)

class BVFT
{
public:
    BVFT() {}
    BVFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
        : keypoints(keypoints), descriptors(descriptors) {}

    std::vector<cv::KeyPoint> keypoints; // keypoints coordinates
    cv::Mat descriptors;                 // keypoint descriptors
    cv::Mat angle;                       // dominant orientations
};

// lesat square ICP using SVD
cv::Mat get_trans_icp(const cv::Mat &src_, const cv::Mat &dst_);

// lesat square ICP using SVD
cv::Mat get_trans_icp_3_points(const cv::Mat &src_, const cv::Mat &dst_);

// RANSAC of rigid transform
cv::Mat estimateICP(const cv::Mat &src, const cv::Mat &dst, std::vector<int> &inliers_ind, int max_iters_user = 10000);

void writeMatToFile(cv::Mat &m, const char *filename);
int writeMatToBin(const cv::Mat &m, const std::string filename);

cv::Mat matAbsAtan2(cv::Mat &y, cv::Mat &x);

cv::Mat matCos(cv::Mat &x);

// utils of fft
cv::Mat ifft2shift(cv::Mat &fft_img);
cv::Mat fft2(const cv::Mat &img);
inline cv::Mat ifft2(const cv::Mat &img);

// BVFT descriptor extraction
BVFT detectBVFT(cv::Mat img1);
