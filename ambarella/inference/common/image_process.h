#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inference/common/net_process.h"

void image_resize_square(const cv::Mat &src, cv::Size dst_size, cv::Mat &dst_image);

void preprocess(nnctrl_ctx_t *nnctrl_ctx, const cv::Mat &srcRoI, 
                const cv::Size &dst_size, const int resize_type);

#endif //IMAGEPROCESS_H