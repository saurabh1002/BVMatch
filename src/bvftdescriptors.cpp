#include "bvftdescriptors.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <fstream>

int norient = 12;
int nscale = 4;
int pad_size = 138;

// lesat square ICP using SVD
cv::Mat get_trans_icp(const cv::Mat &src_, const cv::Mat &dst_)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 3*N
    int used_points = src_.cols;

    // normalize
    cv::Mat src_x = src_.row(0);
    cv::Mat src_y = src_.row(1);
    cv::Mat dst_x = dst_.row(0);
    cv::Mat dst_y = dst_.row(1);
    float mean_src_x = cv::sum(src_x)[0] / used_points;
    float mean_src_y = cv::sum(src_y)[0] / used_points;
    float mean_dst_x = cv::sum(dst_x)[0] / used_points;
    float mean_dst_y = cv::sum(dst_y)[0] / used_points;

    cv::Mat temp_src = src_.clone();
    cv::Mat temp_dst = dst_.clone();
    temp_src.row(0) = src_x - mean_src_x;
    temp_src.row(1) = src_y - mean_src_y;
    temp_dst.row(0) = dst_x - mean_dst_x;
    temp_dst.row(1) = dst_y - mean_dst_y;

    cv::Mat temp_src_x = temp_src.row(0);
    cv::Mat temp_src_y = temp_src.row(1);
    cv::Mat temp_dst_x = temp_dst.row(0);
    cv::Mat temp_dst_y = temp_dst.row(1);

    cv::Mat temp_sqrt_src, temp_sqrt_dst;
    cv::sqrt(temp_src_x.mul(temp_src_x) + temp_src_y.mul(temp_src_y), temp_sqrt_src);
    cv::sqrt(temp_dst_x.mul(temp_dst_x) + temp_dst_y.mul(temp_dst_y), temp_sqrt_dst);
    float mean_src_dis = cv::sum(temp_sqrt_src)[0] / used_points;
    float mean_dst_dis = cv::sum(temp_sqrt_dst)[0] / used_points;

    float src_sf = std::sqrt(2) / mean_src_dis;
    float dst_sf = std::sqrt(2) / mean_dst_dis;

    cv::Mat src_trans = (cv::Mat_<float>(3, 3) << src_sf, 0, -src_sf * mean_src_x,
                         0, src_sf, -src_sf * mean_src_y,
                         0, 0, 1);
    cv::Mat dst_trans = (cv::Mat_<float>(3, 3) << dst_sf, 0, -dst_sf * mean_dst_x,
                         0, dst_sf, -dst_sf * mean_dst_y,
                         0, 0, 1);

    temp_src = src_trans * src_;
    temp_dst = dst_trans * dst_;
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> ti = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    t1 = std::chrono::steady_clock::now();

    // SVD
    cv::Mat A(2 * used_points, 4, CV_32F, cv::Scalar{0});
    cv::Mat xp(2 * used_points, 1, CV_32F, cv::Scalar{0});
    const float *temp_src_ptr_x = temp_src.ptr<float>(0);
    const float *temp_src_ptr_y = temp_src.ptr<float>(1);
    const float *temp_dst_ptr_x = temp_dst.ptr<float>(0);
    const float *temp_dst_ptr_y = temp_dst.ptr<float>(1);
    for (int k = 0; k < used_points; k++)
    {

        cv::Mat temp = (cv::Mat_<float>(1, 4) << temp_src_ptr_x[k], -temp_src_ptr_y[k],
                        1, 0);
        temp.copyTo(A.row(2 * k));
        temp = (cv::Mat_<float>(1, 4) << temp_src_ptr_y[k], temp_src_ptr_x[k],
                0, 1);
        temp.copyTo(A.row(2 * k + 1)); // = ;
        xp.ptr<float>(2 * k)[0] = temp_dst_ptr_x[k];
        xp.ptr<float>(2 * k + 1)[0] = temp_dst_ptr_y[k];
    }

    cv::Mat D_t;
    cv::Mat U(used_points * 2, used_points * 2, CV_32F, cv::Scalar{0});
    cv::Mat D(used_points * 2, 4, CV_32F, cv::Scalar{0});
    cv::Mat V(4, 4, CV_32F, cv::Scalar{0});
    cv::SVDecomp(A, D_t, U, V);

    D = cv::Mat::diag(1.0 / D_t);

    cv::Mat h = V * D * U.t() * xp;

    cv::Mat H = (cv::Mat_<float>(3, 3) << h.ptr<float>(0)[0], -h.ptr<float>(1)[0], h.ptr<float>(2)[0],
                 h.ptr<float>(1)[0], h.ptr<float>(0)[0], h.ptr<float>(3)[0],
                 0, 0, 1);
    cv::Mat inv_dst_trans;
    cv::invert(dst_trans, inv_dst_trans);

    H = inv_dst_trans * H * src_trans;

    cv::SVDecomp(H(cv::Rect(0, 0, 2, 2)), D, U, V);

    cv::Mat S = cv::Mat::eye(2, 2, CV_32F);
    if (cv::determinant(U) * cv::determinant(V) < 0)
        S.ptr<float>(1)[1] = -1;
    H(cv::Rect(0, 0, 2, 2)) = U * S * V.t();
    t2 = std::chrono::steady_clock::now();
    ti = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    return H;
}

// lesat square ICP using SVD
cv::Mat get_trans_icp_3_points(const cv::Mat &src_, const cv::Mat &dst_)
{
    std::cout << "get trans icp 3 points\n";
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    int samples = 3;

    const float *src_ptr_x = src_.ptr<float>(0);
    const float *src_ptr_y = src_.ptr<float>(1);
    const float *dst_ptr_x = dst_.ptr<float>(0);
    const float *dst_ptr_y = dst_.ptr<float>(1);

    float mean_src_x = (src_ptr_x[0] + src_ptr_x[1] + src_ptr_x[2]) / 3;
    float mean_src_y = (src_ptr_y[0] + src_ptr_y[1] + src_ptr_y[2]) / 3;
    float mean_dst_x = (dst_ptr_x[0] + dst_ptr_x[1] + dst_ptr_x[2]) / 3;
    float mean_dst_y = (dst_ptr_y[0] + dst_ptr_y[1] + dst_ptr_y[2]) / 3;

    cv::Mat temp_src = src_.clone();
    cv::Mat temp_dst = dst_.clone();

    float *temp_src_ptr_x = temp_src.ptr<float>(0);
    float *temp_src_ptr_y = temp_src.ptr<float>(1);
    float *temp_dst_ptr_x = temp_dst.ptr<float>(0);
    float *temp_dst_ptr_y = temp_dst.ptr<float>(1);

    temp_src_ptr_x[0] = src_ptr_x[0] - mean_src_x;
    temp_src_ptr_x[1] = src_ptr_x[1] - mean_src_x;
    temp_src_ptr_x[2] = src_ptr_x[2] - mean_src_x;
    temp_src_ptr_y[0] = src_ptr_y[0] - mean_src_y;
    temp_src_ptr_y[1] = src_ptr_y[1] - mean_src_y;
    temp_src_ptr_y[2] = src_ptr_y[2] - mean_src_y;

    temp_dst_ptr_x[0] = dst_ptr_x[0] - mean_dst_x;
    temp_dst_ptr_x[1] = dst_ptr_x[1] - mean_dst_x;
    temp_dst_ptr_x[2] = dst_ptr_x[2] - mean_dst_x;
    temp_dst_ptr_y[0] = dst_ptr_y[0] - mean_dst_y;
    temp_dst_ptr_y[1] = dst_ptr_y[1] - mean_dst_y;
    temp_dst_ptr_y[2] = dst_ptr_y[2] - mean_dst_y;

    float mean_src_dis = std::sqrt(temp_src_ptr_x[0] * temp_src_ptr_x[0] + temp_src_ptr_y[0] * temp_src_ptr_y[0]) + std::sqrt(temp_src_ptr_x[1] * temp_src_ptr_x[1] + temp_src_ptr_y[1] * temp_src_ptr_y[1]) + std::sqrt(temp_src_ptr_x[2] * temp_src_ptr_x[2] + temp_src_ptr_y[2] * temp_src_ptr_y[2]);
    mean_src_dis /= 3;
    float mean_dst_dis = std::sqrt(temp_dst_ptr_x[0] * temp_dst_ptr_x[0] + temp_dst_ptr_y[0] * temp_dst_ptr_y[0]) + std::sqrt(temp_dst_ptr_x[1] * temp_dst_ptr_x[1] + temp_dst_ptr_y[1] * temp_dst_ptr_y[1]) + std::sqrt(temp_dst_ptr_x[2] * temp_dst_ptr_x[2] + temp_dst_ptr_y[2] * temp_dst_ptr_y[2]);
    mean_dst_dis /= 3;

    float src_sf = std::sqrt(2) / mean_src_dis;
    float dst_sf = std::sqrt(2) / mean_dst_dis;

    cv::Mat src_trans = (cv::Mat_<float>(3, 3) << src_sf, 0, -src_sf * mean_src_x,
                         0, src_sf, -src_sf * mean_src_y,
                         0, 0, 1);
    cv::Mat dst_trans = (cv::Mat_<float>(3, 3) << dst_sf, 0, -dst_sf * mean_dst_x,
                         0, dst_sf, -dst_sf * mean_dst_y,
                         0, 0, 1);

    temp_src_ptr_x[0] = src_sf * src_ptr_x[0] - src_sf * mean_src_x;
    temp_src_ptr_x[1] = src_sf * src_ptr_x[1] - src_sf * mean_src_x;
    temp_src_ptr_x[2] = src_sf * src_ptr_x[2] - src_sf * mean_src_x;
    temp_src_ptr_y[0] = src_sf * src_ptr_y[0] - src_sf * mean_src_y;
    temp_src_ptr_y[1] = src_sf * src_ptr_y[1] - src_sf * mean_src_y;
    temp_src_ptr_y[2] = src_sf * src_ptr_y[2] - src_sf * mean_src_y;

    temp_dst_ptr_x[0] = dst_sf * dst_ptr_x[0] - dst_sf * mean_dst_x;
    temp_dst_ptr_x[1] = dst_sf * dst_ptr_x[1] - dst_sf * mean_dst_x;
    temp_dst_ptr_x[2] = dst_sf * dst_ptr_x[2] - dst_sf * mean_dst_x;
    temp_dst_ptr_y[0] = dst_sf * dst_ptr_y[0] - dst_sf * mean_dst_y;
    temp_dst_ptr_y[1] = dst_sf * dst_ptr_y[1] - dst_sf * mean_dst_y;
    temp_dst_ptr_y[2] = dst_sf * dst_ptr_y[2] - dst_sf * mean_dst_y;

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> ti = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    t1 = std::chrono::steady_clock::now();

    cv::Mat A(2 * samples, 4, CV_32F, cv::Scalar{0});
    cv::Mat xp(2 * samples, 1, CV_32F, cv::Scalar{0});

    for (int k = 0; k < samples; k++)
    {
        float *A_x_0 = A.ptr<float>(2 * k);
        A_x_0[0] = temp_src_ptr_x[k];
        A_x_0[1] = -temp_src_ptr_y[k];
        A_x_0[2] = 1;
        A_x_0[3] = 0;

        float *A_x_1 = A.ptr<float>(2 * k + 1);
        A_x_1[0] = temp_src_ptr_y[k];
        A_x_1[1] = temp_src_ptr_x[k];
        A_x_1[2] = 0;
        A_x_1[3] = 1;

        xp.ptr<float>(2 * k)[0] = temp_dst_ptr_x[k];
        xp.ptr<float>(2 * k + 1)[0] = temp_dst_ptr_y[k];
    }

    cv::Mat D_t;
    cv::Mat U(samples * 2, samples * 2, CV_32F, cv::Scalar{0});
    cv::Mat D(samples * 2, 4, CV_32F, cv::Scalar{0});
    cv::Mat V(4, 4, CV_32F, cv::Scalar{0});

    cv::SVDecomp(A, D_t, U, V);

    D = cv::Mat::diag(1.0 / D_t);
    cv::Mat h = V * D * U.t() * xp;
    cv::Mat H = (cv::Mat_<float>(3, 3) << h.ptr<float>(0)[0], -h.ptr<float>(1)[0], h.ptr<float>(2)[0],
                 h.ptr<float>(1)[0], h.ptr<float>(0)[0], h.ptr<float>(3)[0],
                 0, 0, 1);

    cv::Mat inv_dst_trans = (cv::Mat_<float>(3, 3) << 1.0 / dst_sf, 0, mean_dst_x,
                             0, 1.0 / dst_sf, mean_dst_y,
                             0, 0, 1);

    H = inv_dst_trans * H * src_trans;
    cv::SVDecomp(H(cv::Rect(0, 0, 2, 2)), D, U, V);
    cv::Mat S = cv::Mat::eye(2, 2, CV_32F);
    float U_det = U.ptr<float>(0)[0] * U.ptr<float>(1)[1] - U.ptr<float>(0)[1] * U.ptr<float>(1)[0];
    float V_det = V.ptr<float>(0)[0] * V.ptr<float>(1)[1] - V.ptr<float>(0)[1] * V.ptr<float>(1)[0];
    if (U_det * V_det < 0)
        S.ptr<float>(1)[1] = -1;

    H(cv::Rect(0, 0, 2, 2)) = U * S * V.t();

    return H;
}

// RANSAC of rigid transform
cv::Mat estimateICP(const cv::Mat &src, const cv::Mat &dst, std::vector<int> &inliers_ind, int max_iters_user)
{
    // N*2
    int samples = src.rows;

    std::cout << "perform RANSAC on " << samples << " cv::matches" << std::endl;

    cv::Mat src_homo(samples, 3, CV_32F, cv::Scalar{0});
    cv::Mat dst_homo(samples, 3, CV_32F, cv::Scalar{0});
    src_homo(cv::Rect(2, 0, 1, samples)) = 1;
    dst_homo(cv::Rect(2, 0, 1, samples)) = 1;
    src.copyTo(src_homo(cv::Rect(0, 0, 2, samples)));
    dst.copyTo(dst_homo(cv::Rect(0, 0, 2, samples)));
    cv::Mat src_homo_t = src_homo.t();
    cv::Mat dst_homo_t = dst_homo.t();

    double max_iteration = 1.0 * samples * (samples - 1) / (2) + 3;
    if (max_iteration > 10000)
        max_iteration = 10000;
    max_iteration = max_iters_user;

    int max_consensus_number = 2;
    float min_neighbor_dis = 2;
    float err_t = 2;
    int used_points = 3;
    int ind_vec[used_points];
    cv::Mat consensus_T;
    cv::RNG rng((unsigned)time(NULL));

    for (int i = 0; i < max_iteration; i++)
    {
        try
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < used_points; k++)
            {
                ind_vec[k] = int((samples - 1) * rng.uniform((float)0, (float)1));
            }
            bool is_neighbor = false;
            for (int k = 0; k < used_points; k++)
            {
                for (int l = k + 1; l < used_points; l++)
                {
                    if (!is_neighbor)
                        is_neighbor = ind_vec[k] == ind_vec[l];
                    if (!is_neighbor)
                        is_neighbor = cv::sum(cv::abs(src_homo.row(ind_vec[k]) - src_homo.row(ind_vec[l])))[0] < 3;
                    if (!is_neighbor)
                        is_neighbor = cv::sum(cv::abs(dst_homo.row(ind_vec[k]) - dst_homo.row(ind_vec[l])))[0] < 3;
                }
            }
            if (is_neighbor)
                continue;

            cv::Mat src_(3, used_points, CV_32F, cv::Scalar{0});
            cv::Mat dst_(3, used_points, CV_32F, cv::Scalar{0});
            for (int k = 0; k < used_points; k++)
            {
                cv::Mat temp_t = src_homo(cv::Rect(0, ind_vec[k], 3, 1)).t();
                temp_t.copyTo(src_.col(k)); // = src.row(ind_vec[k]);
                temp_t = dst_homo(cv::Rect(0, ind_vec[k], 3, 1)).t();
                temp_t.copyTo(dst_.col(k)); // = src.row(ind_vec[k]);
            }
            cv::Mat T = get_trans_icp_3_points(src_, dst_);
            cv::Mat err;

            cv::Mat err_temp = T * src_homo_t - dst_homo_t;
            err_temp = err_temp.mul(err_temp);

            cv::reduce(err_temp, err, 0, cv::REDUCE_SUM); // one row

            int consensus_num = cv::sum((err < err_t * err_t) / 255)[0];
            if (consensus_num > max_consensus_number)
            {
                max_consensus_number = consensus_num;
                consensus_T = T.clone();
            }
        }
        catch (cv::Exception &e)
        {
            continue;
        }
    }

    cv::Mat err;
    cv::reduce((consensus_T * src_homo.t() - dst_homo.t()).mul(consensus_T * src_homo.t() - dst_homo.t()), err, 0, cv::REDUCE_SUM);
    for (int i = 0; i < err.cols; i++)
    {
        if (err.ptr<float>(0)[i] < err_t * err_t)
            inliers_ind.push_back(i);
    }
    cv::Mat inliers_src(3, inliers_ind.size(), CV_32F, cv::Scalar{0});
    cv::Mat inliers_dst(3, inliers_ind.size(), CV_32F, cv::Scalar{0});
    for (int i = 0; i < inliers_ind.size(); i++)
    {

        cv::Mat temp_t = src_homo(cv::Rect(0, inliers_ind[i], 3, 1)).t();
        temp_t.copyTo(inliers_src.col(i)); // = src.row(ind_vec[k]);
        temp_t = dst_homo(cv::Rect(0, inliers_ind[i], 3, 1)).t();
        temp_t.copyTo(inliers_dst.col(i)); // = src.row(ind_vec[k]);
    }
    consensus_T = get_trans_icp(inliers_src, inliers_dst);

    std::cout << "find " << inliers_ind.size() << " inliers" << std::endl;

    return consensus_T;
}

void writeMatToFile(cv::Mat &m, const char *filename)
{
    std::ofstream fout(filename);
    int mat_type = m.type();
    if (!fout || mat_type >= 8)
    {
        std::cout << "File Not Opened or multi-channel cv::mat" << std::endl;
        return;
    }

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            if (mat_type % 8 == 0)
                fout << int(m.at<uint8_t>(i, j)) << "\t";
            else if (mat_type % 8 == 1)
                fout << m.at<int8_t>(i, j) << "\t";
            else if (mat_type % 8 == 2)
                fout << m.at<uint16_t>(i, j) << "\t";
            else if (mat_type % 8 == 3)
                fout << m.at<int16_t>(i, j) << "\t";
            else if (mat_type % 8 == 4)
                fout << m.at<int>(i, j) << "\t";
            else if (mat_type % 8 == 5)
                fout << m.at<float>(i, j) << "\t";
            else if (mat_type % 8 == 6)
                fout << m.at<double>(i, j) << "\t";
        }
        fout << std::endl;
    }

    fout.close();
}

int writeMatToBin(const cv::Mat &m, const std::string filename)
{
    std::ofstream fout(filename.c_str(), std::ios::binary);
    int mat_type = m.type();
    if (!fout || mat_type >= 8)
    {
        std::cout << "File Not Opened or multi-channel cv::mat" << std::endl;
        return 1;
    }

    float s;
    // s=m.rows;
    // fout.write((char*)&s, sizeof(float));
    // s=m.cols;
    // fout.write((char*)&s, sizeof(float));
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            if (mat_type % 8 == 0)
                s = int(m.at<uint8_t>(i, j));
            else if (mat_type % 8 == 1)
                s = m.at<int8_t>(i, j);
            else if (mat_type % 8 == 2)
                s = m.at<uint16_t>(i, j);
            else if (mat_type % 8 == 3)
                s = m.at<int16_t>(i, j);
            else if (mat_type % 8 == 4)
                s = m.at<int>(i, j);
            else if (mat_type % 8 == 5)
                s = m.at<float>(i, j);
            else if (mat_type % 8 == 6)
                s = m.at<double>(i, j);
            fout.write((char *)&s, sizeof(float));
        }
    }

    fout.close();
    return 0;
}

cv::Mat matAbsAtan2(cv::Mat &y, cv::Mat &x)
{
    cv::Mat ax = cv::abs(x);
    cv::Mat ay = cv::abs(y);

    // x>y情况
    cv::Mat re1(y.rows, y.cols, CV_32FC1, cv::Scalar{1});
    cv::Mat re2(y.rows, y.cols, CV_32FC1, cv::Scalar{1});
    cv::multiply(re1, (ax >= ay) / 255, re1, 1, CV_32F);
    cv::Mat c = ay / (ax + _DBL_EPSILON);
    c = re1.mul(c);
    cv::Mat c2 = c.mul(c);
    c2 = (((atan2_p7 * c2 + atan2_p5).mul(c2) + atan2_p3).mul(c2) + atan2_p1).mul(c);
    re1 = c2.mul(re1);

    // x<y情况
    cv::multiply(re2, (ax < ay) / 255, re2, 1, CV_32F);
    c = ax / (ay + _DBL_EPSILON);
    c = c.mul(re2);
    c2 = c.mul(c);
    c2 = 90.0 - (((atan2_p7 * c2 + atan2_p5).mul(c2) + atan2_p3).mul(c2) + atan2_p1).mul(c);
    re2 = c2.mul(re2);

    // 两种情况结合
    cv::Mat ret = re1 + re2;

    // 区分一二象限
    cv::multiply(ret, (x < 0) / 255, re1, 1, CV_32F);
    cv::multiply(ret, (x >= 0) / 255, re2, 1, CV_32F);
    re1 = 180.0 - re1;
    cv::multiply(re1, (x < 0) / 255, re1, 1, CV_32F);
    ret = re1 + re2;
    /*
    multiply(ret, (y<0)/255, re1, 1,CV_32F);
    multiply(ret, (y>=0)/255, re2, 1,CV_32F);
    re1 = 360.0 - re1;
    multiply(re1, (y<0)/255, re1, 1,CV_32F);
    ret = re1+re2;
    */
    return ret / 180 * CV_PI;
}

cv::Mat matCos(cv::Mat &x)
{
    cv::Mat temp_x = CV_PI / 2 - x;
    cv::Mat x2 = temp_x.mul(temp_x);
    cv::Mat y = temp_x - temp_x.mul(x2) / 6 + temp_x.mul(x2).mul(x2) / 120 - temp_x.mul(x2).mul(x2).mul(x2) / 5040;
    return y;
}

// utils of fft
cv::Mat ifft2shift(cv::Mat &fft_img);
cv::Mat fft2(const cv::Mat &img);
inline cv::Mat ifft2(const cv::Mat &img);

cv::Mat fft2(const cv::Mat &img)
{
    cv::Mat m_src;
    cv::Mat m_fourier(img.rows, img.cols, CV_32FC2, cv::Scalar(0, 0));
    if (img.type() < 8)
    {
        cv::Mat m_for_fourier[] = {cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F)};
        cv::merge(m_for_fourier, 2, m_src);
    }
    else if (img.type() < 16)
        m_src = img;
    else
    {
        std::cout << "FFT input type channel error, input type is " << img.type() << std::endl;
        return cv::Mat(img.rows, img.cols, CV_32FC2, cv::Scalar{0});
    }
    cv::dft(m_src, m_fourier);
    return m_fourier;
}
inline cv::Mat ifft2(const cv::Mat &img)
{
    cv::Mat m_src;
    cv::Mat m_fourier(img.rows, img.cols, CV_32FC2, cv::Scalar(0, 0));
    if (img.type() < 8)
    {
        cv::Mat m_for_fourier[] = {cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F)};
        cv::merge(m_for_fourier, 2, m_src);
    }
    else if (img.type() < 16)
    {
        // std::cout << "CVC2" << std::endl;
        m_src = img;
    }
    else
    {
        std::cout << "FFT input type channel error, input type is " << img.type() << std::endl;
        return cv::Mat(img.rows, img.cols, CV_32FC2, cv::Scalar{0});
    }
    // std::cout << img.size << std::endl;

    // std::cout << m_src.size << std::endl;
    cv::dft(m_src, m_fourier, cv::DFT_INVERSE | cv::DFT_SCALE);
    return m_fourier;
}
cv::Mat ifft2shift(cv::Mat &fft_img)
{
    int rows = fft_img.rows;
    int cols = fft_img.cols;

    int cy = rows / 2 + rows % 2;
    int cx = cols / 2 + cols % 2;

    cv::Mat ret(rows, cols, fft_img.type(), cv::Scalar{0}); // fft_img.clone();

    fft_img(cv::Rect(0, rows - cy, cols, cy)).copyTo(ret(cv::Rect(0, 0, cols, cy)));
    fft_img(cv::Rect(0, 0, cols, rows - cy)).copyTo(ret(cv::Rect(0, cy, cols, rows - cy)));

    cv::Mat tmp = ret.clone();
    tmp(cv::Rect(cols - cx, 0, cx, rows)).copyTo(ret(cv::Rect(0, 0, cx, rows)));
    tmp(cv::Rect(0, 0, cols - cx, rows)).copyTo(ret(cv::Rect(cx, 0, cols - cx, rows)));

    return ret;
}

// BVFT descriptor extraction
BVFT detectBVFT(cv::Mat img1)
{
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    cv::normalize(img1, img1, 0, 255, cv::NORM_MINMAX);

    int rows = img1.rows;
    int cols = img1.cols;

    cv::Mat radius(rows, cols, CV_32FC1, cv::Scalar{0});
    cv::Mat theta(rows, cols, CV_32FC1, cv::Scalar{0});

    // low-pass filter
    cv::Mat lowpass_filter(rows, cols, CV_32FC1, cv::Scalar{0});
    float lowpass_cutoff = 0.45;
    int sharpness = 15;
    for (int x = 0; x < cols; x += 1)
    {
        for (int y = 0; y < rows; y += 1)
        {
            float x_range = -1 / 2.0 + x * 1.0 / cols;
            if (cols % 2)
                x_range = -1 / 2.0 + x * 1.0 / (cols - 1);
            float y_range = -1 / 2.0 + y * 1.0 / rows;
            if (rows % 2)
                y_range = -1 / 2.0 + y * 1.0 / (rows - 1);
            radius.ptr<float>(y)[x] = std::sqrt(y_range * y_range + x_range * x_range);
            theta.ptr<float>(y)[x] = std::atan2(-y_range, x_range);
            lowpass_filter.ptr<float>(y)[x] = 1.0 / (1 + std::pow(radius.ptr<float>(y)[x] / lowpass_cutoff, 2 * sharpness));
        }
    }

    radius = ifft2shift(radius);
    theta = ifft2shift(theta);
    lowpass_filter = ifft2shift(lowpass_filter);

    radius.at<float>(0, 0) = 1;

    cv::Mat sintheta(rows, cols, CV_32FC1, cv::Scalar{0});
    cv::Mat costheta(rows, cols, CV_32FC1, cv::Scalar{0});
    for (int x = 0; x < cols; x += 1)
    {
        for (int y = 0; y < rows; y += 1)
        {
            sintheta.ptr<float>(y)[x] = std::sin(theta.ptr<float>(y)[x]);
            costheta.ptr<float>(y)[x] = std::cos(theta.ptr<float>(y)[x]);
        }
    }

    // Log-Gabor filter construction
    float min_wavelength = 3;
    float mult = 1.6;
    float sigma_on_f = 0.75;
    std::vector<cv::Mat> log_gabor;
    for (int s = 0; s < nscale; s++)
    {
        float wavelength = min_wavelength * pow(mult, s);
        float fo = 1.0 / wavelength;
        cv::Mat log_gabor_s;
        cv::log(radius / fo, log_gabor_s);
        log_gabor_s = -log_gabor_s.mul(log_gabor_s) / (2 * cv::log(sigma_on_f) * cv::log(sigma_on_f));
        cv::exp(log_gabor_s, log_gabor_s);
        log_gabor_s = log_gabor_s.mul(lowpass_filter);
        log_gabor_s.at<float>(0, 0) = 0;
        log_gabor.push_back(log_gabor_s);
    }

    // fft of the input BV image
    cv::Mat img_fft = fft2(img1);

    cv::Mat eo[nscale][norient];
    for (int i = 0; i < nscale; i++)
        for (int j = 0; j < norient; j++)
            eo[i][j] = cv::Mat(rows, cols, CV_32FC2, cv::Scalar{0});

    for (int o = 0; o < norient; o++)
    {
        float angle = o * CV_PI / norient;
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        cv::Mat spread(rows, cols, CV_32FC1, cv::Scalar{0});

        cv::Mat ytheta = (sintheta * cos_angle - costheta * sin_angle);
        cv::Mat xtheta = (costheta * cos_angle + sintheta * sin_angle);
        cv::Mat dtheta = matAbsAtan2(ytheta, xtheta);
        dtheta = cv::abs(dtheta) * norient;
        dtheta = cv::min(dtheta, CV_PI);
        spread = (matCos(dtheta) + 1) / 2;

        for (int s = 0; s < nscale; s++)
        {
            // oriented Log-Gabor filter
            cv::Mat filter(rows, cols, CV_32FC1, cv::Scalar{0});
            filter = log_gabor[s].mul(spread);

            // perform convolution using fft
            cv::Mat img_fft_channels[] = {cv::Mat::zeros(img1.size(), CV_32F), cv::Mat::zeros(img1.size(), CV_32F)};
            cv::split(img_fft, img_fft_channels);
            img_fft_channels[0] = img_fft_channels[0].mul(filter);
            img_fft_channels[1] = img_fft_channels[1].mul(filter);

            cv::Mat img_fft_filtered;
            merge(img_fft_channels, 2, img_fft_filtered);
            eo[s][o] = ifft2(img_fft_filtered);
        }
    }

    // detect FAST keypoints on the BV image
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    cv::Mat for_fast = img1.clone();
    cv::normalize(img1, for_fast, 0, 255, cv::NORM_MINMAX); //=img1.clone();

    std::vector<cv::KeyPoint> keypoints_raw;
    std::vector<cv::KeyPoint> keypoints;
    fast->detect(for_fast, keypoints_raw);
    float max_metric = 0;
    cv::KeyPointsFilter::removeDuplicated(keypoints_raw);

    // std::cout << "detected " << keypoints_raw.size() << " keypoints" << std::endl;

    int patch_size = 138;
    int patch_size_true = 96;
    cv::Mat keypoints_mask(for_fast.rows, for_fast.cols, CV_8U, cv::Scalar{0});

    // rm the keypoints at borders
    for (int i = 0; i < keypoints_raw.size(); i++)
    {
        bool at_border = (keypoints_raw[i].pt.x < patch_size / 2 || keypoints_raw[i].pt.y < patch_size / 2 || keypoints_raw[i].pt.x > img1.cols - patch_size / 2 || keypoints_raw[i].pt.y > img1.rows - patch_size / 2);
        if (!at_border)
            keypoints.push_back(keypoints_raw[i]);
    }

    // std::cout << "keep " << keypoints.size() << " keypoints" << std::endl;

    // calculate log-gabor responses
    cv::Mat CS[norient];
    for (int o = 0; o < norient; o++)
    {
        CS[o] = cv::Mat(rows, cols, CV_32FC1, cv::Scalar{0});
    }
    for (int o = 0; o < norient; o++)
    {
        for (int s = 0; s < nscale; s++)
        {
            cv::Mat img_channels[] = {cv::Mat::zeros(img1.size(), CV_32F), cv::Mat::zeros(img1.size(), CV_32F)};
            cv::split(eo[s][o], img_channels);
            cv::Mat EO_re = img_channels[0];
            cv::Mat EO_im = img_channels[1];
            cv::Mat An;
            cv::magnitude(EO_re, EO_im, An);
            CS[o] += cv::abs(An);
        }
    }

    // build MIM
    cv::Mat MIM(rows, cols, CV_8U, cv::Scalar{0});
    cv::Mat max_response(rows, cols, CV_32FC1, cv::Scalar{0});
    for (int o = 0; o < norient; o++)
    {
        MIM = cv::max(MIM, (CS[o] > max_response) / 255 * (o + 1));
        max_response = cv::max(CS[o], max_response);
    }

    // depress the pixels with low responses
    cv::Mat mim_mask;
    mim_mask = ((max_response > 0.1)) / 255;

    cv::multiply(MIM, mim_mask, MIM);

    int descriptor_orients = norient / 2;
    int areas = 6;
    cv::Mat decriptors(areas * areas * descriptor_orients, keypoints.size(), CV_32F, cv::Scalar{0});
    std::vector<int> kps_to_ignore(keypoints.size(), 0);
    cv::Mat kps_angle(1, keypoints.size(), CV_32F, cv::Scalar{0});
    ;

    bool descriptor_permute = 1;
    bool descriptor_rotate = 1;

    // weight kernel for dominant orientation computation
    float patch_main_orient_kernel[patch_size][patch_size];
    float patch_kernel_radius = patch_size / 2;
    for (int k = 0; k < patch_size; k++)
        for (int j = 0; j < patch_size; j++)
        {
            float dis = (k - patch_size / 2) * (k - patch_size / 2) + (j - patch_size / 2) * (j - patch_size / 2);
            dis = std::sqrt(dis) / patch_kernel_radius;
            if (dis > 1)
                dis = 1;
            patch_main_orient_kernel[k][j] = 1 - dis;
            patch_main_orient_kernel[k][j] *= patch_main_orient_kernel[k][j];
        }

    // describe every keypoint
    for (int k = 0; k < keypoints.size(); k++)
    {

        // find the patch position
        int x = keypoints[k].pt.x;
        int y = keypoints[k].pt.y;

        float x_low = cv::max(0, x - patch_size / 2 - patch_size % 2);
        float x_hig = cv::min(x + patch_size / 2 - patch_size % 2, cols);
        float y_low = cv::max(0, y - int(patch_size / 2) - patch_size % 2);
        float y_hig = cv::min(y + int(patch_size / 2) - patch_size % 2, rows);

        cv::Mat patch = MIM(cv::Rect(cv::Point(x_low, y_low), cv::Point(x_hig, y_hig))).clone();
        cv::Mat patch_mask = mim_mask(cv::Rect(cv::Point(x_low, y_low), cv::Point(x_hig, y_hig))).clone();
        cv::Mat patch_max_response = max_response(cv::Rect(cv::Point(x_low, y_low), cv::Point(x_hig, y_hig))).clone();

        // compute dominant orientation
        float hist[norient + 1] = {0};
        float hist_energy[norient + 1] = {0};
        for (int hy = 0; hy < patch.rows; hy++)
        {
            uint8_t *ptr_y = patch.ptr<uint8_t>(hy);
            float *ptr_m = patch_max_response.ptr<float>(hy);
            for (int hx = 0; hx < patch.cols; hx++)
            {
                hist[ptr_y[hx]] += patch_main_orient_kernel[hy][hx] * ptr_m[hx];
                hist_energy[ptr_y[hx]] += patch_main_orient_kernel[hy][hx];
            }
        }
        int max_orient = 1;
        for (int hist_i = 1; hist_i < norient + 1; hist_i++)
            if (hist[hist_i] > hist[max_orient])
                max_orient = hist_i;
        if (hist[(max_orient - 1) == 0 ? norient : (max_orient - 1)] > hist[(max_orient + 1) == (norient + 1) ? 1 : (max_orient + 1)])
            max_orient = (max_orient - 1) == 0 ? norient : (max_orient - 1);

        kps_angle.ptr<float>(0)[k] = (max_orient);

        // circle shift the values of the MIM patch
        if (descriptor_permute)
        {
            cv::Mat shang = ((norient) + patch) - (max_orient); //-norient/2-1; //uint8 4drop5in
            patch = shang - (shang >= norient) / 255 * norient + 1;
            patch = patch.mul(patch_mask);
        }

        // rotate the patch
        if (descriptor_rotate)
        {
            keypoints[k].angle = -180 * (max_orient - 1) / norient;
            cv::Size dst_sz(patch.cols, patch.rows);
            cv::Point2f center(static_cast<float>(patch.cols / 2.), static_cast<float>(patch.rows / 2.));
            float angle = -180 * (max_orient - 1) / norient;

            cv::Mat patch_te = patch.clone();
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(patch, patch, rot_mat, dst_sz, cv::INTER_NEAREST, cv::BORDER_REPLICATE);
        }

        cv::Mat patch_true = patch(cv::Rect((patch_size - patch_size_true) / 2, (patch_size - patch_size_true) / 2,
                                            patch_size_true, patch_size_true));

        int ys = patch_true.rows;
        int xs = patch_true.cols;
        // describing the patch
        for (int j = 0; j < areas; j++)
            for (int i = 0; i < areas; i++)
            {
                cv::Mat clip = patch_true(cv::Rect(j * ys / areas, i * xs / areas,
                                                   ys / areas, xs / areas))
                                   .clone();

                float hist[norient + 1] = {0};
                for (int hi = 0; hi < norient; hi++)
                    hist[hi] = 0;
                for (int hy = 0; hy < clip.rows; hy++)
                {
                    uint8_t *ptr_y = clip.ptr<uint8_t>(hy);
                    for (int hx = 0; hx < clip.cols; hx++)
                    {
                        float weight = (std::fabs(j * ys / areas + hx - patch_size_true / 2) + std::fabs(i * xs / areas + hy - patch_size_true / 2)) / patch_size_true;
                        weight = 1 - weight * weight;
                        hist[ptr_y[hx]] += 1;
                    }
                }
                float ker = std::fabs(i - areas / 2.0) + std::fabs(j - areas / 2.0);
                ker /= areas;
                ker = 1 - ker;
                for (int hist_i = 0; hist_i < descriptor_orients; hist_i++)
                    decriptors.ptr<float>(j * areas * descriptor_orients + i * descriptor_orients + hist_i)[k] = (hist[2 * hist_i + 1] + hist[2 * hist_i + 2]); //(hist[2*hist_i+1]+hist[2*hist_i+2]);//*ker;//;//
            }

        // normalize the descriptor
        float norm_sum = 0;
        for (int norm_i = 0; norm_i < decriptors.rows; norm_i++)
            norm_sum += decriptors.ptr<float>(norm_i)[k] * decriptors.ptr<float>(norm_i)[k];
        norm_sum = std::sqrt(norm_sum);
        for (int norm_i = 0; norm_i < decriptors.rows; norm_i++)
            decriptors.ptr<float>(norm_i)[k] /= norm_sum;
        float sum_main = 0, sum_main_rela = 0;
    }

    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_used = std::chrono::duration_cast<
        std::chrono::duration<float>>(t_end - t_start);

    decriptors = decriptors.t();
    BVFT bvft(keypoints, decriptors);
    bvft.angle = kps_angle.clone();

    return bvft; // BVFT(keypoints, decriptors);
}