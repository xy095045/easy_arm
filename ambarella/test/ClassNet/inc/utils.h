#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/time.h>
#include <thread>
#include <mutex>
#include <future>
#include <opencv2/opencv.hpp>

unsigned long get_current_time(void);
void ListPath(std::string const &path, std::vector<std::string> &paths);
void ListImages(std::string const &path, std::vector<std::string> &images);
// void preprocess(cv::Mat img, cv::Mat& inp_img, int crop_height, int input_width, int input_height);
// void get_output(int8_t* dpuOut, int sizeOut, float scale, int oc, int oh, int ow, float* result);

#endif