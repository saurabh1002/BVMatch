
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <python2.7/Python.h>

#include "match_two_scan.hpp"
#include "bvftdescriptors.h"

int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string filename)
{
    point_cloud.clear();
    std::ifstream binfile(filename.c_str(), std::ios::binary);
    if (!binfile)
    {
        throw std::runtime_error("file \"" + filename + "\" cannot open");
        return -1;
    }
    else
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::vector<float> tmp;
        while (1)
        {
            double s;
            pcl::PointXYZ point;
            binfile.read((char *)&s, sizeof(double));
            if (binfile.eof())
                break;
            tmp.push_back(s);
            point.x = s;
            binfile.read((char *)&s, sizeof(double));
            tmp.push_back(s);
            point.y = s;
            binfile.read((char *)&s, sizeof(double));
            tmp.push_back(s);
            point.z = s;
            point_cloud.push_back(point);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    }
    binfile.close();
    return 1;
}

std::vector<std::string> pyListSortDir(std::string path)
{
    std::vector<std::string> ret;

    Py_Initialize();

    // PyRun_SimpleString("import os");
    PyObject *module_name = PyString_FromString("os");
    PyObject *os_module = PyImport_Import(module_name);
    PyObject *os_list = PyObject_GetAttrString(os_module, "listdir");
    // PyObject* os_list = PyObject_GetAttrString(os_module, "");

    PyObject *ArgList = PyTuple_New(1);
    PyObject *py_path = PyString_FromString(path.c_str());
    PyTuple_SetItem(ArgList, 0, py_path);

    PyObject *files = PyObject_CallObject(os_list, ArgList);
    // PyObject*
    PyList_Sort(files);
    // PyObject_CallMethod(PyList,"sort", 'O', files);
    // PyObject* files_sort_ = PyObject_CallObject(files_sort,ArgList);
    for (int i = 0; i < PyList_Size(files); i++)
    {
        char *temp;
        PyObject *item = PyList_GetItem(files, i);
        // std::cout << item << std::endl;
        PyArg_Parse(item, "s", &temp);
        // std::cout << std::string(temp) << std::endl;
        ret.push_back(std::string(temp));
    }
    // PyObject* re = PyRun_SimpleString("os.list()");
    Py_Finalize();
    return ret;
}

void generateBVImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, int x_max_ind, int y_max_ind, cv::Mat &mat_local_image)
{
    float resolution = 0.4;
    pcl::VoxelGrid<pcl::PointXYZ> down_size_filter;
    down_size_filter.setLeafSize(resolution, resolution, resolution / 2);
    down_size_filter.setInputCloud(point_cloud.makeShared());
    down_size_filter.filter(point_cloud);

    float x_min = 10000, y_min = 10000, x_max = -100000, y_max = -100000;
    for (int i = 0; i < point_cloud.size(); i++)
    {
        if (point_cloud.points[i].y < x_min)
            x_min = point_cloud.points[i].y;
        if (point_cloud.points[i].y > x_max)
            x_max = point_cloud.points[i].y;
        if (point_cloud.points[i].x < y_min)
            y_min = point_cloud.points[i].x;
        if (point_cloud.points[i].x > y_max)
            y_max = point_cloud.points[i].x;
    }
    int x_min_ind = int(x_min / resolution);
    x_max_ind = int(x_max / resolution);
    int y_min_ind = int(y_min / resolution);
    y_max_ind = int(y_max / resolution);

    int x_num = x_max_ind - x_min_ind + 1;
    int y_num = y_max_ind - y_min_ind + 1;
    mat_local_image = cv::Mat(y_num, x_num, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < point_cloud.size(); i++)
    {
        int x_ind = x_max_ind - int((point_cloud.points[i].y) / resolution);
        int y_ind = y_max_ind - int((point_cloud.points[i].x) / resolution);
        if (x_ind >= x_num || y_ind >= y_num)
            continue;
        mat_local_image.at<uint8_t>(y_ind, x_ind) += 1;
    }
    uint8_t max_pixel = 0;
    for (int i = 0; i < x_num; i++)
        for (int j = 0; j < y_num; j++)
        {
            if (mat_local_image.at<uint8_t>(j, i) > max_pixel)
                max_pixel = mat_local_image.at<uint8_t>(j, i);
        }
    for (int i = 0; i < x_num; i++)
        for (int j = 0; j < y_num; j++)
        {
            if (mat_local_image.at<uint8_t>(j, i) * 10 > 100)
            {
                mat_local_image.at<uint8_t>(j, i) = 100;
                continue;
            }
            mat_local_image.at<uint8_t>(j, i) = uint8_t(mat_local_image.at<uint8_t>(j, i) * 10);
            if (uint8_t(mat_local_image.at<uint8_t>(j, i)) == 0)
            {
                mat_local_image.at<uint8_t>(j, i) = 10;
                continue;
            }
        }
}

void imagePadding(cv::Mat &img, int cor_x, int cor_y)
{
    int pad_size = 200;
    cv::copyMakeBorder(img, img, pad_size / 2, pad_size / 2, pad_size / 2, pad_size / 2, cv::BORDER_CONSTANT, cv::Scalar(10));

    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);

    int row_pad = (m - img.rows) / 2;
    int col_pad = (n - img.cols) / 2;
    // take this step to make fft faster
    cv::copyMakeBorder(img, img, row_pad, (m - img.rows) % 2 ? row_pad + 1 : row_pad,
                       col_pad, (n - img.cols) % 2 ? col_pad + 1 : col_pad, cv::BORDER_CONSTANT, cv::Scalar(10));
    cor_x += col_pad + pad_size / 2;
    cor_y += row_pad + pad_size / 2;
}

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

    pcl::PointCloud<pcl::PointXYZ> point_cloud1;
    pcl::PointCloud<pcl::PointXYZ> point_cloud2;

    // read point clouds
    readPointCloud(point_cloud1, img1_path);
    readPointCloud(point_cloud2, img2_path);
    std::cout << "Read PointClouds\n";

    // apply rotation transform to test the rotation invariance
    float rotation_angle = 137.0 / 180 * CV_PI;
    for (int i = 0; i < point_cloud1.size(); i++)
    {
        pcl::PointXYZ tmp = point_cloud1[i];
        point_cloud1[i].x = std::cos(rotation_angle) * tmp.x + std::sin(rotation_angle) * tmp.y;
        point_cloud1[i].y = -std::sin(rotation_angle) * tmp.x + std::cos(rotation_angle) * tmp.y;
    }

    cv::Mat img1, img2;
    int max_x1, max_y2, max_x2, max_y1; // used for localizing the center of the images

    // generate bv images
    generateBVImage(point_cloud1, max_x1, max_y1, img1);
    generateBVImage(point_cloud2, max_x2, max_y2, img2);
    std::cout << "Generated BV Images\n";

    // padding to make fft faster
    imagePadding(img1, max_x1, max_y1);
    imagePadding(img2, max_x2, max_y2);
    std::cout << "Done Padding\n";

    // perform matching
    match2Image(img1, img2, max_x1, max_y1, max_x2, max_y2);
}
