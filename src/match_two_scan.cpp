
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>

#include "utils.hpp"
#include "match_two_scan.hpp"
#include "bvftdescriptors.h"

void match2Image(cv::Mat img1, cv::Mat img2, int max_x1, int max_y1, int max_x2, int max_y2)
{
    float rows = img1.rows, cols = img1.cols;
    BVFT bvfts1 = detectBVFT(img1);
    BVFT bvfts2 = detectBVFT(img2);

    bvfts2.keypoints.insert(bvfts2.keypoints.end(), bvfts2.keypoints.begin(), bvfts2.keypoints.end());

    cv::Mat temp(bvfts2.keypoints.size(), bvfts2.descriptors.cols, CV_32F, cv::Scalar{0});
    bvfts2.descriptors.copyTo(temp(cv::Rect(0, 0, bvfts2.descriptors.cols, bvfts2.keypoints.size() / 2)));
    int areas = 6;
    int feautre_size = bvfts2.descriptors.cols / areas / areas;
    for (int i = 0; i < areas * areas; i++) // areas*areas
    {
        bvfts2.descriptors(cv::Rect((areas * areas - i - 1) * feautre_size, 0, feautre_size, bvfts2.keypoints.size() / 2)).copyTo(temp(cv::Rect(i * feautre_size, bvfts2.keypoints.size() / 2, feautre_size, bvfts2.keypoints.size() / 2))); // i*norient
    }
    bvfts2.descriptors = temp.clone();

    cv::BFMatcher matcher; //(NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(bvfts1.descriptors, bvfts2.descriptors, matches);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    std::vector<cv::DMatch>::iterator it_end = matches.end();
    for (std::vector<cv::DMatch>::iterator it = matches.begin(); it != it_end; it++)
    {
        cv::Point2f point_local_1 = (cv::Point2f(max_y1, max_x1) -
                                     cv::Point2f(bvfts1.keypoints[it->queryIdx].pt.y, bvfts1.keypoints[it->queryIdx].pt.x)) *
                                    0.4;
        points1.push_back(point_local_1);

        point_local_1 = (cv::Point2f(max_y2, max_x2) -
                         cv::Point2f(bvfts2.keypoints[it->trainIdx].pt.y, bvfts2.keypoints[it->trainIdx].pt.x)) *
                        0.4;
        points2.push_back(point_local_1);
    }
    cv::Mat keypoints1(points1);
    keypoints1 = keypoints1.reshape(1, keypoints1.rows); // N*2
    cv::Mat keypoints2(points2);
    keypoints2 = keypoints2.reshape(1, keypoints2.rows);

    cv::Mat inliers;

    std::vector<int> inliers_ind;
    cv::Mat rigid = estimateICP(keypoints1, keypoints2, inliers_ind);

    if (inliers_ind.size() < 4)
        std::cout << "few inlier points\n";
    std::cout << "find transform: \n"
              << rigid << std::endl;

    std::vector<cv::DMatch> good_matches;

    for (int i = 0; i < inliers_ind.size(); i++)
        good_matches.push_back(matches[inliers_ind[i]]);

    cv::Mat matchesGoodImage;

    for (int i = 0; i < img1.rows; i++)
        for (int j = 0; j < img1.cols; j++)
            if (img1.ptr<uint8_t>(i)[j] <= 10)
                img1.ptr<uint8_t>(i)[j] = 10;

    for (int i = 0; i < img2.rows; i++)
        for (int j = 0; j < img2.cols; j++)
            if (img2.ptr<uint8_t>(i)[j] <= 10)
                img2.ptr<uint8_t>(i)[j] = 10;

    cv::normalize(img1, img1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(img2, img2, 0, 255, cv::NORM_MINMAX);

    cv::drawKeypoints(img1, bvfts1.keypoints, img1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img2, bvfts2.keypoints, img2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::drawMatches(img1, bvfts1.keypoints, img2, bvfts2.keypoints, good_matches, matchesGoodImage, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("matchesGoodImage", matchesGoodImage);
    cv::imwrite("match.png", matchesGoodImage);
    cv::waitKey(0);
}

int main(int argc, char **argv)
{

    std::string img1_path = argv[1];
    std::string img2_path = argv[2];

    std::vector<Eigen::Vector3d> point_cloud1;
    std::vector<Eigen::Vector3d> point_cloud2;

    // read point clouds
    readPointCloud(point_cloud1, img1_path);
    readPointCloud(point_cloud2, img2_path);
    std::cout << "Read PointClouds\n";

    // apply rotation transform to test the rotation invariance
    // float rotation_angle = 137.0 / 180 * CV_PI;
    // for (auto &point : point_cloud1)
    // {
    //     Eigen::Vector3d tmp(point);
    //     point.x() = std::cos(rotation_angle) * tmp.x() + std::sin(rotation_angle) * tmp.y();
    //     point.y() = -std::sin(rotation_angle) * tmp.x() + std::cos(rotation_angle) * tmp.y();
    // }

    cv::Mat img1, img2;
    int max_x1, max_y2, max_x2, max_y1; // used for localizing the center of the images

    // generate bv images
    generateImage(point_cloud1, max_x1, max_y1, img1);
    generateImage(point_cloud2, max_x2, max_y2, img2);
    std::cout << "Generated BV Images\n";
    cv::imshow("img1", img1);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // padding to make fft faster
    imagePadding(img1, max_x1, max_y1);
    imagePadding(img2, max_x2, max_y2);
    std::cout << "Done Padding\n";

    // perform matching
    match2Image(img1, img2, max_x1, max_y1, max_x2, max_y2);
}
