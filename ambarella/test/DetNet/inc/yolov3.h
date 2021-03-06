/*
  The following source code derives from Darknet
*/

#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

using namespace std;

#define ROUND_UP_32(x) ((x)&0x1f ? (((x)&0xffffffe0) + 32) : (x))
#define LAYER_P(w) (ROUND_UP_32(4 * w))/4

#define YOLO_NET_IN_HEIGHT 416
#define YOLO_NET_IN_WIDTH 416
#define g_anchorCnt 3
#define g_classificationCnt 4
#define NMS_THRESHOLD 0.45f
#define CONF 0.24

const std::vector<float> g_anchors{8.95, 8.57, 12.43, 26.71, 19.71, 14.43,
                                 26.36, 58.52, 36.09, 25.55, 64.42, 42.90,
                                 96.44, 79.10, 158.37, 115.59, 218.65, 192.90};
// const std::vector<float> g_anchors{10,13, 16,30, 33,23,
//                                   30,61, 62,45, 59,119, 
//                                   116,90, 156,198, 373,326};


const int g_layer0_w = (YOLO_NET_IN_WIDTH / 32); //32)
const int g_layer0_h = (YOLO_NET_IN_HEIGHT / 32);   //32)

const int g_layer1_w = (YOLO_NET_IN_WIDTH / 16);     //16)
const int g_layer1_h = (YOLO_NET_IN_HEIGHT / 16);    //16)

const int g_layer2_w = (YOLO_NET_IN_WIDTH / 8);     //8)
const int g_layer2_h = (YOLO_NET_IN_HEIGHT / 8);    //8)

const int g_yolo_layer_channel = (g_anchorCnt *(g_classificationCnt+5));

vector<vector<float>> yolo_run(float* node0, float* node1, float* node2, int img_h, int img_w);

int entry_index(int loc, int anchorC, int w, int h, int lWidth, int lHeight);
float sigmoid(float p);
float overlap(float x1, float w1, float x2, float w2);
float cal_iou(vector<float> box, vector<float>truth);

//for amba version
void detect(vector<vector<float>> &boxes, const float* lOutput, \
			int lHeight, int lWidth, int num, \
			int sHeight, int sWidth);
vector<vector<float>> applyNMS(vector<vector<float>>& boxes,
							const float thres, bool sign_nms=true);

#endif  //__UTILS_H__