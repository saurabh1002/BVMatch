#include <string>
#include <vector>
#include <Eigen/Core>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <python2.7/Python.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include "utils.hpp"

int readPointCloud(std::vector<Eigen::Vector3d> &point_cloud_downsampled, const std::string filename)
{
    std::ifstream binfile(filename.c_str(), std::ios::binary);
    if (!binfile)
    {
        throw std::runtime_error("file cannot open");
        return -1;
    }
    else
    {
        std::vector<float> tmp;
        while (1)
        {
            double s;
            binfile.read((char *)&s, sizeof(double));
            if (binfile.eof())
                break;
            auto x = s;
            binfile.read((char *)&s, sizeof(double));
            auto y = s;
            binfile.read((char *)&s, sizeof(double));
            auto z = s;

            point_cloud_downsampled.emplace_back(Eigen::Vector3d(x, y, z));
        }
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

void imagePadding(cv::Mat &img, int cor_x, int cor_y)
{
    int pad_size = 200;

    cv::copyMakeBorder(img, img, pad_size / 2, pad_size / 2, pad_size / 2, pad_size / 2, cv::BORDER_CONSTANT, cv::Scalar(10));

    // Extending image
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

void generateImage(std::vector<Eigen::Vector3d> &point_cloud_eigen, int x_max_ind, int y_max_ind, cv::Mat &mat_local_image)
{
    pcl::PointCloud<pcl::PointXYZ> point_cloud;
    point_cloud.clear();
    for (const auto &point_eigen : point_cloud_eigen)
    {
        pcl::PointXYZ point;
        point.x = point_eigen.x();
        point.y = point_eigen.y();
        point.z = point_eigen.z();
        point_cloud.push_back(point);
    }

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