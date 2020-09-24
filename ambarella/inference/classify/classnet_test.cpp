/*******************************************************************************
 * classnet_test.c
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

#include <sstream>

#include "opencv2/opencv.hpp"

#include "inference/common/utils.h"
#include "inference/common/vproc_process.h"
#include "inference/common/net_process.h"

#define CLASS_NUM (100)

const static int g_canvas_id = 1;
const static char *net_in_name = '0';
const static char *net_out_name = "192";


typedef struct classnet_ctx_s {
    cavalry_ctx_t cavalry_ctx;
    vproc_ctx_t vproc_ctx;
    nnctrl_ctx_t nnctrl_ctx;
} classnet_ctx_s;


static void set_net_io(nnctrl_ctx_t *nnctrl_ctx){
	nnctrl_ctx->net.net_in.in_num = 1;
    strcpy(nnctrl_ctx->net.net_in.in_desc[0].name, net_in_name);
	nnctrl_ctx->net.net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->net.net_out.out_num = 1;
    strcpy(nnctrl_ctx->net.net_out.out_desc[0].name, net_out_name); 
	nnctrl_ctx->net.net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
}

static int init_param(classnet_ctx_s *classify_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &classify_ctx->nnctrl_ctx; 
    memset(nnctrl_ctx, 0, sizeof(nnctrl_ctx_t));

    nnctrl_ctx->verbose = 0;
    nnctrl_ctx->reuse_mem = 1;
    nnctrl_ctx->cache_en = 1;
    nnctrl_ctx->buffer_id = g_canvas_id;
    nnctrl_ctx->log_level = 0;

    strcpy(nnctrl_ctx->net.net_file, "./classnet.bin"); 

    return rval;
}

static int classnet_init(classnet_ctx_s *classify_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &classify_ctx->nnctrl_ctx; 

    rval = init_net_context(&classify_ctx->nnctrl_ctx, &classify_ctx->cavalry_ctx, 
                            nnctrl_ctx->verbose, nnctrl_ctx->cache_en);

    set_net_io(nnctrl_ctx);
    rval = init_net(nnctrl_ctx, nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);
    rval = load_net(nnctrl_ctx);

    if (rval < 0) {
        printf("init net context, return %d\n", rval);
    }

    return rval;
}

static void classnet_deinit(classnet_ctx_s *classify_ctx)
{
    deinit_net_context(&classify_ctx->nnctrl_ctx, &classify_ctx->cavalry_ctx);
    DPRINT_NOTICE("mtcnn_deinit\n");
}

int classnet_run(classnet_ctx_s *classify_ctx, float *output)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &classify_ctx->nnctrl_ctx;

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

    for (int i=0; i < CLASS_NUM; i++) {
        output[i] = DIM1_DATA(score_addr, i);
    }
    return rval;
}

void preprocess(classnet_ctx_s *classify_ctx, cv::Mat &srcRoI)
{
    nnctrl_ctx_t *nnctrl_ctx = &classify_ctx->nnctrl_ctx;

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

int postprocess(const float *output)
{
    float id_max = -100;
    int class_idx = 0;
    for (int id = 0; id < CLASS_NUM; id++) {
        if (output[id] > id_max) {
            id_max = output[id];
            class_idx = id;
        }
    }
    return class_idx;
}

void image_dir_infer(const std::string &image_dir){
    classnet_ctx_s classify_ctx;
    std::vector<std::string> images;
    std::ofstream save_result;
    float output[CLASS_NUM];
    int class_idx = -1;
    memset(&classify_ctx, 0, sizeof(classnet_ctx_s));
    rval = init_param(&classify_ctx);
    rval = classnet_init(&classify_ctx);
    ListImages(image_dir, images);
    std::cout << "total Test images : " << images.size() << std::endl;
    save_result.open("./cls_result.txt");
    for (size_t index = 0; index < images.size(); index++) {
		std::stringstream temp_str;
        temp_str << image_dir << images[index];
		std::cout << temp_str.str() << std::endl;
		img = cv::imread(temp_str.str());
        time_start = get_current_time();
        preprocess(&classify_ctx, src_img);
        classnet_run(&classify_ctx, output);
        class_idx = postprocess(output);
        time_end = get_current_time();
        std::cout << "classnet cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;

        save_result << images[index] << " " << class_idx << "\n";
    }
    save_result.close();
    classnet_deinit(&classify_ctx);
}

void image_txt_infer(const std::string &image_dir, const std::string &image_txt_path){
    classnet_ctx_s classify_ctx;
    std::vector<std::string> images;
    std::ofstream save_result;
    std::ifstream read_txt;
    float output[CLASS_NUM];
    int class_idx = -1;
    std::string line_data;

    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return;
    }
    
    memset(&classify_ctx, 0, sizeof(classnet_ctx_s));
    init_param(&classify_ctx);
    classnet_init(&classify_ctx);
    save_result.open("./cls_result.txt");
    while(std::getline(infile, line_data)){
        if(line_data.empty()){
            continue;
        }
        size_t index = line_data.find_first_of(' ', 0);
        std::string image_name = ine_data.substr(0, index);
        std::stringstream image_path;
        image_path << image_dir << image_name;
        std::cout << image_path.str() << std::endl;
        img = cv::imread(image_path.str());
        time_start = get_current_time();
        preprocess(&classify_ctx, src_img);
        classnet_run(&classify_ctx, output);
        class_idx = postprocess(output);
        time_end = get_current_time();
        std::cout << "classnet cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;

        save_result << image_name << " " << class_idx << "\n";
    }
    read_txt.close();
    save_result.close();
    classnet_deinit(&classify_ctx);
}

int main()
{
    std::cout << "start..." << std::endl;
    std::string image_dir = "./test_images/";
    image_dir_infer(image_dir);
    std::cout << "End of game!!!" << std::endl;
    return 0;
}
