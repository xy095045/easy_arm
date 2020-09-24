/*******************************************************************************
 * segnet_test.c
 *
 * Author: foweiw
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <dirent.h>

#include "opencv2/opencv.hpp"

#include "inference/common/utils.h"
#include "inference/common/vproc_process.h"
#include "inference/common/net_process.h"

#define SEGMENT_NUM (200000)
#define SegOutputWidth 500
#define SegOutputHeight 400
#define SegOutputChannel 2
#define THRESH 0.3

const static int g_canvas_id = 1;
const static char *net_in_name = '0';
const static char *net_out_name = "507";

typedef struct seg_ctx_s {
    cavalry_ctx_t cavalry_ctx;
    vproc_ctx_t vproc_ctx;
    nnctrl_ctx_t nnctrl_ctx;
} seg_ctx_t;

static void set_net_io(nnctrl_ctx_t *nnctrl_ctx){
	nnctrl_ctx->net.net_in.in_num = 1;
    strcpy(nnctrl_ctx->net.net_in.in_desc[0].name, net_in_name);
	nnctrl_ctx->net.net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->net.net_out.out_num = 1;
    strcpy(nnctrl_ctx->net.net_out.out_desc[0].name, net_out_name); 
	nnctrl_ctx->net.net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
}

static int init_param(mtcnn_ctx_t *mtcnn_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx; 
    memset(nnctrl_ctx, 0, sizeof(nnctrl_ctx_t));

    nnctrl_ctx->verbose = 0;
    nnctrl_ctx->reuse_mem = 1;
    nnctrl_ctx->cache_en = 1;
    nnctrl_ctx->buffer_id = g_canvas_id;
    nnctrl_ctx->log_level = 0;

    strcpy(nnctrl_ctx->PNet[eSegNet].net_file, "./segnet.bin"); 

    return rval;
}

static int segment_init(seg_ctx_t *seg_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &seg_ctx->nnctrl_ctx; 

    rval = init_net_context(&seg_ctx->nnctrl_ctx, &seg_ctx->cavalry_ctx, 
                            nnctrl_ctx->verbose, nnctrl_ctx->cache_en);

    set_net_io(nnctrl_ctx);
    rval = init_net(nnctrl_ctx, nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);
    rval = load_net(nnctrl_ctx);

    if (rval < 0) {
        printf("init net context, return %d\n", rval);
    }

    return rval;
}

static void segment_deinit(seg_ctx_t *seg_ctx)
{
    deinit_net_context(&seg_ctx->nnctrl_ctx, &seg_ctx->cavalry_ctx);
    DPRINT_NOTICE("segment_deinit\n");
}

int segment_run(seg_ctx_t *seg_ctx, float *output)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &seg_ctx->nnctrl_ctx;

    rval = nnctrl_run_net(nnctrl_ctx->net.net_id, &nnctrl_ctx->net.result, NULL, NULL, NULL);

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_run_net() failed, return %d\n", rval);
    }

    // parse the output of classnet
    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->net.net_m.mem_size, nnctrl_ctx->net.net_m.phy_addr, 0, 1);
    }
    
     float *score_addr = (float *)(nnctrl_ctx->net.net_m.virt_addr
        + nnctrl_ctx->net.net_out.out_desc[0].addr - nnctrl_ctx->net.net_m.phy_addr);

    int output_c = nnctrl_ctx->net.net_out.out_desc[0].dim.depth;
    int output_h = nnctrl_ctx->net.net_out.out_desc[0].dim.height;
    int output_w = nnctrl_ctx->net.net_out.out_desc[0].dim.width;
    int output_p = nnctrl_ctx->net.net_out.out_desc[0].dim.pitch;

    std::cout << "poutput size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
                                  << output_w << "--output_p: " << output_p << "--" << std::endl;

    float seg_output[SegOutputHeight * 504];
    memcpy(seg_output, score_addr, SegOutputHeight * 504 * sizeof(float));

    for (int h = 0; h < output_h; h++)
    {
        memcpy(output + h * output_w, seg_output + h * 504, output_w * sizeof(float));
    }

    return rval;
}

void preprocess(seg_ctx_t *seg_ctx, cv::Mat &srcRoI)
{
    nnctrl_ctx_t *nnctrl_ctx = &seg_ctx->nnctrl_ctx;

    // int channel = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.depth;
    int height = nnctrl_ctx->net.net_in.in_desc[0].dim.height;
    int width = nnctrl_ctx->net.net_in.in_desc[0].dim.width;
    // std::cout << "--channel: " << channel << "--height: " << height << "--width: " << width << "--" << std::endl;

    cv::Size szNet = cv::Size(width, height);
    cv::Mat detRoI = srcRoI;

    cv::resize(detRoI, detRoI, szNet, 0, 0); //cv::INTER_LINEAR

    std::vector<cv::Mat> channel_s;
    cv::split(detRoI, channel_s);

    for (int i=0; i<3; i++)
    {
        memcpy(nnctrl_ctx->net.net_in.in_desc[0].virt + i * height * width, channel_s[2-i].data, height * width); // bgr2rgb
    }
    
    // sycn address
    cavalry_mem_sync_cache(nnctrl_ctx->net.net_in.in_desc[0].size, nnctrl_ctx->net.net_in.in_desc[0].addr, 1, 0);
}

void postprocess(float* output, cv::Mat seg_mat)
{
    uint8_t colorB[] = {0, 0};
    uint8_t colorG[] = {0, 0};
    uint8_t colorR[] = {0, 255};
    float thresh = 0.3;

    for (int row = 0; row < SegOutputHeight; row++) {
        for (int col = 0; col < SegOutputWidth; col++) {
            int posit;
            if (output[row * SegOutputWidth + col] > thresh) {
                posit = 1;
            }
            else {
                posit = 0;
            }
            // int i = row * SegOutputWidth * SegOutputChannel + col * SegOutputChannel;
            // auto max_ind = std::max_element(output + i, output + i + SegOutputChannel);
            // uint8_t posit = (uint8_t)std::distance(output + i, max_ind);
            seg_mat.at<cv::Vec3b>(row, col) = cv::Vec3b(colorB[posit], colorG[posit], colorR[posit]);
        }
    }
}

void predict_segnet(mtcnn_ctx_t *mtcnn_ctx, cv::Mat &src_img, int img_idx)
{
    int rval = 0;

    // preprocess input to the network
    preprocess(mtcnn_ctx, src_img, eSegNet);
    std::cout << "4-----model preprocess done!!!" << std::endl;

    rval = nnctrl_run_segnet(mtcnn_ctx);
    std::cout << "5-----model run done!!!" << std::endl; 

    cv::Mat seg_mat(SegOutputHeight, SegOutputWidth, CV_8UC3);
    postprocess(mtcnn_ctx, seg_mat);
    std::cout << "6-----model postprocess done!!!" << std::endl; 

	std::stringstream index;
	index << img_idx;

    for (int i = 0; i < seg_mat.rows * seg_mat.cols * 3; i++) {
      src_img.data[i] = src_img.data[i] * 0.4 + seg_mat.data[i] * 0.6;
    }

	cv::imwrite("result/result_00" + index.str() + ".jpg", src_img);
}

int main()
{
    std::cout << "start..." << std::endl;

    // Amba init 
    int rval = 0;
    mtcnn_ctx_t mtcnn_ctx;
    
    memset(&mtcnn_ctx, 0, sizeof(mtcnn_ctx_t));
    rval = init_param(&mtcnn_ctx);
    rval = mtcnn_init(&mtcnn_ctx);
    std::cout << "1-----model init done!!!" << std::endl;

    std::vector<std::string> images;
	std::string test_images = "./test_images/";
	std::cout << "Reading Test Images " << std::endl;
	ListImages(test_images, images);
	if (images.size() == 0) {
		std::cerr << "\nError: No images exist in " << test_images << std::endl;
	} else {
		std::cout << "total Test images : " << images.size() << std::endl;
	}

	cv::Mat img;
	unsigned long time_start, time_end;
	for (unsigned int img_idx = 0; img_idx < images.size(); img_idx++) {
		std::cout << test_images + images.at(img_idx) << std::endl;
		img = cv::imread(test_images + images.at(img_idx));
        time_start = get_current_time();
        predict_segnet(&mtcnn_ctx, img, img_idx);
        time_end = get_current_time();
        std::cout << "classnet cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;
    }
    mtcnn_deinit(&mtcnn_ctx);

    std::cout << "End of game!!!" << std::endl;

    return 0;
}
