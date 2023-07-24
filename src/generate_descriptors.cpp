#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Core>

#include "bvftdescriptors.h"
#include "utils.hpp"

int main(int argc, char **argv)
{
    std::string bin_path = argv[1];

    std::vector<std::string> bin_file_names = pyListSortDir(bin_path);
    std::cout << bin_file_names.size() << " Scan Files Found" << std::endl;

    for (int i = 0; i < bin_file_names.size(); i++)
    {

        std::vector<Eigen::Vector3d> point_cloud;
        int max_local_x_ind, max_local_y_ind;
        cv::Mat mat_local_image;

        readPointCloud(point_cloud, bin_path + bin_file_names[i]);

        // generate BV image, recording the cornet points
        generateImage(point_cloud, max_local_x_ind, max_local_y_ind, mat_local_image);

        // padding to make fft faster
        std::cout << "processing: " << i + 1 << "/" << bin_file_names.size() << std::endl;
        imagePadding(mat_local_image, max_local_x_ind, max_local_y_ind);

        BVFT bvft = detectBVFT(mat_local_image);

        // save descriptor to mat
        writeMatToBin(bvft.descriptors, ("des_" + std::to_string(10000 + i) + "_" + std::to_string(max_local_x_ind) + "_" + std::to_string(max_local_y_ind) + ".bin").c_str());
    }
}
