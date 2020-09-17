/*******************************************************************************
 * detnet_test.c
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

// #include "iav_ioctl.h"
#include "cavalry_ioctl.h"

#include "cavalry_mem.h"
#include "vproc.h"
#include "nnctrl.h"
#include "utils.h"
#include "yolov3.h"

#define MAX_FILE_NAME_LEN			(128)
static int g_canvas_id = 1;

// add max and min to math
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

/**
 * @brief submodule to operate array data
 *
 */
#define BLOB_DATA(a, K, H, W, n, k, h, w) (a)[(( (n) * (K) + (k) ) * (H) + (h) ) * (W) + (w)]
#define DIM2_DATA(a, W, h, w) (a)[(h) * (W) + (w)]
#define DIM1_DATA(a, i) (a)[(i)]

/**
 * @brief submodule to log
 *
 */

static int gsc_log_level;

enum LogLevel {
    LogLevel_None = 0x00,
    LogLevel_Error = 0x01,
    LogLevel_Notice = 0x02,
    LogLevel_Debug = 0x03,
    LogLevel_Verbose = 0x04,
};

void set_log_level(enum LogLevel level)
{
    gsc_log_level = level;
}

#define DASSERT(expr) do { \
	if (!(expr)) { \
		printf("assertion failed: %s\n\tAt file: %s\n\tfunction: %s, line %d\n", #expr, __FILE__, __FUNCTION__, __LINE__); \
	} \
} while (0)

#define D_LOG_PRINT_BY_LEVEL(level, tag, format, args...)  do { \
		if (gsc_log_level >= level) { \
			printf("%s", tag); \
			printf(format, ##args);  \
		} \
	} while (0)

#define D_LOG_PRINT_TRACE_BY_LEVEL(level, tag, format, args...)  do { \
		if (gsc_log_level >= level) { \
			printf("%s", tag); \
			printf(format, ##args);  \
			printf("            [trace] file %s.\n            function: %s: line %d\n", __FILE__, __FUNCTION__, __LINE__); \
		} \
	} while (0)

#define DPRINT_VERBOSE(format, args...)   D_LOG_PRINT_BY_LEVEL(LogLevel_Verbose, "            [Verbose]: ", format, ##args)
#define DPRINT_DEBUG(format, args...)   D_LOG_PRINT_BY_LEVEL(LogLevel_Debug, "        [Debug]: ", format, ##args)
#define DPRINT_NOTICE(format, args...)   D_LOG_PRINT_BY_LEVEL(LogLevel_Notice, "    [Notice]: ", format, ##args)
#define DPRINT_ERROR(format, args...)   D_LOG_PRINT_TRACE_BY_LEVEL(LogLevel_Error, "[Error]: ", format, ##args)

/**
 * @brief submodule to operate on cavalry driver
 */

typedef struct cavalry_ctx_s {
    int fd_cavalry;
} cavalry_ctx_t;

int cavalry_init_context(cavalry_ctx_t *cavalry_ctx, uint8_t verbose)
{
    int rval = 0;
    memset((void *)cavalry_ctx, 0, sizeof(cavalry_ctx_t));

    cavalry_ctx->fd_cavalry = open("/dev/cavalry", O_RDWR, 0);

    if (cavalry_ctx->fd_cavalry < 0) {
        DPRINT_ERROR("open /dev/cavalry failed\n");
        rval = -1;
    }

    rval = cavalry_mem_init(cavalry_ctx->fd_cavalry, verbose);

    if (rval < 0) {
        DPRINT_ERROR("cavalry_mem_init failed\n");
    }

    return 0;
}

void cavalry_deinit_context(cavalry_ctx_t *cavalry_ctx)
{
    if (cavalry_ctx->fd_cavalry >= 0) {
        close(cavalry_ctx->fd_cavalry);
    }
}

void cavalry_sync_cache(unsigned long size, unsigned long phys,
                        uint8_t clean, uint8_t invalid)
{
    if (cavalry_mem_sync_cache(size, phys, clean, invalid) < 0) {
        DPRINT_NOTICE("cavalry_mem_sync_cache failed\n");
    }
}

/**
 * @brief submodule to operate on vproc lib
 *
 */

typedef struct cv_mem_s {
    uint8_t *virt;
    unsigned long phys;
    unsigned long size;
} cv_mem_t;

typedef struct vproc_ctx_s {
    uint8_t cache_en;
    uint8_t is_rgb;

    cv_mem_t lib_mem;
    cv_mem_t yuv2rgb_mem;

    int total_cv_mem_size;
    int total_malloc_size;

    vect_desc_t y_desc;
    vect_desc_t uv_desc;
    vect_desc_t rgb_desc;
    vect_desc_t rgb_roi_desc;

    int max_vproc_batch_num;
} vproc_ctx_t;

int vproc_init_context(vproc_ctx_t *vproc_ctx, cavalry_ctx_t *cavalry_ctx,
                       uint8_t is_rgb, uint8_t cache_en, int max_vproc_batch_num)
{
    int rval = 0;
    struct vproc_version ver;
    uint32_t size = 0;
    memset(vproc_ctx, 0, sizeof(vproc_ctx_t));
    vproc_ctx->is_rgb = is_rgb;
    vproc_ctx->cache_en = cache_en;
    vproc_ctx->max_vproc_batch_num = max_vproc_batch_num;

    rval = vproc_get_version(&ver);

    if (rval < 0) {
        DPRINT_ERROR("vproc_get_version failed\n");
        rval = -1;
    }

    DPRINT_NOTICE("%s: %u.%u.%u, mod-time: 0x%x\n",
                    ver.description, ver.major, ver.minor, ver.patch, ver.mod_time);
    rval = vproc_init("/usr/local/vproc/vproc.bin", &size);

    if (rval < 0) {
        DPRINT_ERROR("vproc_init failed, can not init /usr/local/vproc/vproc.bin\n");
    }

    memset(&vproc_ctx->lib_mem, 0, sizeof(vproc_ctx->lib_mem));
    vproc_ctx->lib_mem.size = size;
    rval = cavalry_mem_alloc(&(vproc_ctx->lib_mem.size), &(vproc_ctx->lib_mem.phys), (void **) & (vproc_ctx->lib_mem.virt), vproc_ctx->cache_en);

    std::cout << "vproc_init_context_cavalry_mem_alloc: " << rval << std::endl;

    if (rval < 0) {
        DPRINT_ERROR("alloc_cv_mem failed\n");
    }

    vproc_ctx->total_cv_mem_size += vproc_ctx->lib_mem.size;
    vproc_load(cavalry_ctx->fd_cavalry, vproc_ctx->lib_mem.virt, vproc_ctx->lib_mem.phys, size);

    if (rval < 0) {
    }

    DPRINT_NOTICE("vproc use cavalry mem total %d bytes\n", vproc_ctx->total_cv_mem_size);
    DPRINT_NOTICE("vproc use malloc total %d bytes\n", vproc_ctx->total_malloc_size);

    return rval;
}

void vproc_deinit_context(vproc_ctx_t *vproc_ctx)
{

}

/**
 * @brief submodule to operate on nnctrl lib
 */
#define net_num 1
#define layer_output_num 3

 struct net_match {
	uint8_t net_id;

	struct net_run_cfg net_rev;
	struct net_mem net_m;

    struct net_cfg cfg;
    struct net_input_cfg net_in;
    struct net_output_cfg net_out;
    struct net_result result;

    char net_in_name[MAX_FILE_NAME_LEN];
    char net_out_name_1[MAX_FILE_NAME_LEN];
    char net_out_name_2[MAX_FILE_NAME_LEN];
    char net_out_name_3[MAX_FILE_NAME_LEN];

    char net_file[MAX_FILE_NAME_LEN];
};

// use index to find the netName
enum eNetName{
    eDetNet = 0,
};

typedef struct nnctrl_ctx_s {
    uint8_t verbose;
    uint8_t reuse_mem;
    uint8_t cache_en;
    uint8_t buffer_id;
    uint8_t log_level;

    struct net_match PNet[net_num];

    float* denet_feature[layer_output_num] ={NULL};
} nnctrl_ctx_t;

static void nnctrl_set_all_nets_io_before_init(nnctrl_ctx_t *nnctrl_ctx, int netId);
static int nnctrl_init_all_nets(nnctrl_ctx_t *nnctrl_ctx, uint8_t verbose, uint8_t cache_en, uint8_t reuse_mem, int netId);
static int nnctrl_load_all_nets(nnctrl_ctx_t *nnctrl_ctx, int netId);

int nnctrl_init_context(nnctrl_ctx_t *nnctrl_ctx,
                        cavalry_ctx_t *cavalry_ctx,
                        vproc_ctx_t *vproc_ctx,
                        uint8_t verbose, 
                        uint8_t cache_en, 
                        uint8_t reuse_mem)
{
    int rval = 0;
    struct nnctrl_version ver;

    nnctrl_ctx->cache_en = cache_en;
    rval = nnctrl_get_version(&ver);

    if (rval < 0) {
        printf("nnctrl_get_version failed");
    }

    DPRINT_NOTICE("%s: %u.%u.%u, mod-time: 0x%x\n",
                    ver.description, ver.major, ver.minor, ver.patch, ver.mod_time);
    rval = nnctrl_init(cavalry_ctx->fd_cavalry, verbose);

    if (rval < 0) {
        printf("nnctrl_init failed\n");
    }

    for (int id=0; id<net_num; id++)
    {
        nnctrl_set_all_nets_io_before_init(nnctrl_ctx, id); // resize io !!!!!!
        rval = nnctrl_init_all_nets(nnctrl_ctx, verbose, cache_en, reuse_mem, id); // init all nets !!!!!!

        rval = nnctrl_load_all_nets(nnctrl_ctx, id);

    }

    return rval;
}

void nnctrl_deinit_context(nnctrl_ctx_t *nnctrl_ctx)
{
    unsigned long size;
    unsigned long phy_addr;

    for (int id=0; id<net_num; id++)
    {
        if (nnctrl_ctx->PNet[id].net_m.virt_addr && nnctrl_ctx->PNet[id].net_m.mem_size) {
            size = nnctrl_ctx->PNet[id].net_m.mem_size;
            phy_addr = nnctrl_ctx->PNet[id].net_m.phy_addr;

            if (cavalry_mem_free(size, phy_addr,
                                nnctrl_ctx->PNet[id].net_m.virt_addr) < 0) {
                DPRINT_NOTICE("cavalry_mem_free failed\n");
            }
        }
    }
}

static void nnctrl_set_all_nets_io_before_init(nnctrl_ctx_t *nnctrl_ctx, int netId)
{
    // posenet 
	nnctrl_ctx->PNet[netId].net_in.in_num = 1;
	nnctrl_ctx->PNet[netId].net_in.in_desc[0].name = nnctrl_ctx->PNet[netId].net_in_name;
	nnctrl_ctx->PNet[netId].net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->PNet[netId].net_out.out_num = 3;
	nnctrl_ctx->PNet[netId].net_out.out_desc[0].name = nnctrl_ctx->PNet[netId].net_out_name_1;
	nnctrl_ctx->PNet[netId].net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
    nnctrl_ctx->PNet[netId].net_out.out_desc[1].name = nnctrl_ctx->PNet[netId].net_out_name_2;
	nnctrl_ctx->PNet[netId].net_out.out_desc[1].no_mem = 0; // let nnctrl lib allocate memory for output
    nnctrl_ctx->PNet[netId].net_out.out_desc[2].name = nnctrl_ctx->PNet[netId].net_out_name_3;
	nnctrl_ctx->PNet[netId].net_out.out_desc[2].no_mem = 0; // let nnctrl lib allocate memory for output
}

static int nnctrl_init_one_net(struct net_cfg *cfg, struct net_input_cfg *input_cfg, struct net_output_cfg *ouput_cfg, struct net_mem *mem,
                               uint8_t verbose, uint8_t reuse_mem, uint32_t batch_num, uint8_t cache_en, char *model_file_path)
{
    int rval = 0;
    int net_id = 0;

    cfg->net_file = model_file_path;
    cfg->verbose = verbose;
    cfg->reuse_mem = reuse_mem;
    cfg->net_loop_cnt = batch_num;

    net_id = nnctrl_init_net(cfg, input_cfg, ouput_cfg);

    if (net_id < 0) {
        DPRINT_ERROR("nnctrl_init_net failed for %s\n", model_file_path);
    }

    if (cfg->net_mem_total == 0) {
        DPRINT_ERROR("nnctrl_init_net get total size is zero for %s\n", model_file_path);
        net_id = -1;
    }

    mem->mem_size = cfg->net_mem_total;
    unsigned long size = mem->mem_size;
    unsigned long phy_addr;

    rval = cavalry_mem_alloc(&size, &phy_addr, (void **) & (mem->virt_addr),
                                cache_en);

    mem->mem_size = size;
    mem->phy_addr = phy_addr;

    if (rval < 0) {
        DPRINT_ERROR("cavalry_mem_alloc failed\n");
        net_id = -1;
    }

    if (mem->virt_addr == NULL) {
        DPRINT_ERROR("cavalry_mem_alloc is NULL\n");
        net_id = -1;
    }

    return net_id;
}

static int nnctrl_init_all_nets(nnctrl_ctx_t *nnctrl_ctx, uint8_t verbose, uint8_t cache_en, uint8_t reuse_mem, int netId)
{
    int rval;

    //net init
    nnctrl_ctx->PNet[netId].net_id = -1;

    rval = nnctrl_init_one_net(&nnctrl_ctx->PNet[netId].cfg, &nnctrl_ctx->PNet[netId].net_in, &nnctrl_ctx->PNet[netId].net_out,
                &nnctrl_ctx->PNet[netId].net_m,
                verbose, reuse_mem, 0/*posenet_batch_num*/, cache_en, nnctrl_ctx->PNet[netId].net_file);

    nnctrl_ctx->PNet[netId].net_id = rval;

    return rval;
}

static int nnctrl_load_all_nets(nnctrl_ctx_t *nnctrl_ctx, int netId)
{
    int rval = 0;

    // load net start
    rval = nnctrl_load_net(nnctrl_ctx->PNet[netId].net_id, &nnctrl_ctx->PNet[netId].net_m,
                &nnctrl_ctx->PNet[netId].net_in, &nnctrl_ctx->PNet[netId].net_out);

    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->PNet[netId].net_m.mem_size, nnctrl_ctx->PNet[netId].net_m.phy_addr, 1, 0);
    }
    // load net end

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_load_all_net error, return %d\n", rval);
    }

    return rval;
}

typedef struct mtcnn_ctx_s {
    cavalry_ctx_t cavalry_ctx;
    vproc_ctx_t vproc_ctx;
    nnctrl_ctx_t nnctrl_ctx;
} mtcnn_ctx_t;

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

    strcpy(nnctrl_ctx->PNet[eDetNet].net_in_name, "data");
    strcpy(nnctrl_ctx->PNet[eDetNet].net_out_name_1, "636");
    strcpy(nnctrl_ctx->PNet[eDetNet].net_out_name_2, "662");
    strcpy(nnctrl_ctx->PNet[eDetNet].net_out_name_3, "688");
    strcpy(nnctrl_ctx->PNet[eDetNet].net_file, "./detnet.bin"); 

    return rval;
}

static int mtcnn_init(mtcnn_ctx_t *mtcnn_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx; 

    set_log_level((enum LogLevel)(nnctrl_ctx->log_level));

    rval = cavalry_init_context(&mtcnn_ctx->cavalry_ctx, nnctrl_ctx->verbose);

    if (rval < 0) {
        printf("cavalry init error, return %d\n", rval);
    }

    rval = nnctrl_init_context(&mtcnn_ctx->nnctrl_ctx, &mtcnn_ctx->cavalry_ctx, &mtcnn_ctx->vproc_ctx,
                               nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);

    if (rval < 0) {
        printf("nnctrl init context, return %d\n", rval);
    }

    return rval;
}

static void mtcnn_deinit(mtcnn_ctx_t *mtcnn_ctx)
{
    nnctrl_deinit_context(&mtcnn_ctx->nnctrl_ctx);
    vproc_deinit_context(&mtcnn_ctx->vproc_ctx);
    cavalry_deinit_context(&mtcnn_ctx->cavalry_ctx);
    DPRINT_NOTICE("mtcnn_deinit\n");
}

int nnctrl_run_detnet(mtcnn_ctx_t *mtcnn_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx;

    rval = nnctrl_run_net(nnctrl_ctx->PNet[eDetNet].net_id, &nnctrl_ctx->PNet[eDetNet].result, NULL, NULL, NULL);

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_run_net() failed, return %d\n", rval);
    }

    // parse the output of classnet
    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->PNet[eDetNet].net_m.mem_size, nnctrl_ctx->PNet[eDetNet].net_m.phy_addr, 0, 1);
    }

	for (int i = 0; i < layer_output_num; i++)
	{
        float *score_addr = (float *)(nnctrl_ctx->PNet[eDetNet].net_m.virt_addr
                            + nnctrl_ctx->PNet[eDetNet].net_out.out_desc[i].addr - nnctrl_ctx->PNet[eDetNet].net_m.phy_addr);


        int output_c = nnctrl_ctx->PNet[eDetNet].net_out.out_desc[i].dim.depth;
        int output_h = nnctrl_ctx->PNet[eDetNet].net_out.out_desc[i].dim.height;
        int output_w = nnctrl_ctx->PNet[eDetNet].net_out.out_desc[i].dim.width;
        int output_p = nnctrl_ctx->PNet[eDetNet].net_out.out_desc[i].dim.pitch;

		nnctrl_ctx->denet_feature[i] = score_addr;	
    }

    return rval;
}

void preprocess(mtcnn_ctx_t *mtcnn_ctx, cv::Mat &srcRoI, int netId)
{
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx;

    // int channel = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.depth;
    int height = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.height;
    int width = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.width;
    // std::cout << "--channel: " << channel << "--height: " << height << "--width: " << width << "--" << std::endl;

    cv::Size szNet = cv::Size(width, height);
    cv::Mat detRoI = srcRoI;

    cv::resize(detRoI, detRoI, szNet, 0, 0); //cv::INTER_LINEAR

    std::vector<cv::Mat> channel_s;
    cv::split(detRoI, channel_s);

    for (int i=0; i<3; i++)
    {
        memcpy(nnctrl_ctx->PNet[netId].net_in.in_desc[0].virt + i * height * width, channel_s[2-i].data, height * width);
    }
    
    // sycn address
    cavalry_mem_sync_cache(nnctrl_ctx->PNet[netId].net_in.in_desc[0].size, nnctrl_ctx->PNet[netId].net_in.in_desc[0].addr, 1, 0);
}

std::vector<std::vector<float>> postprocess(mtcnn_ctx_t *mtcnn_ctx, cv::Mat &srcRoI)
{
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx;

    int img_h = srcRoI.rows;
    int img_w = srcRoI.cols;

    std::vector<std::vector<float>> final_results;
	final_results = yolo_run(nnctrl_ctx->denet_feature[0], nnctrl_ctx->denet_feature[1], nnctrl_ctx->denet_feature[2], img_h, img_w);
    // std::cout << "final results size: " << final_results.size() << std::endl;

    return final_results;
}

void predict_detnet(mtcnn_ctx_t *mtcnn_ctx, cv::Mat &src_img, int img_idx, std::vector<std::string> images)
{
    int rval = 0;

    // preprocess input to the network
    preprocess(mtcnn_ctx, src_img, eDetNet);
    std::cout << "4-----model preprocess done!!!" << std::endl;

    rval = nnctrl_run_detnet(mtcnn_ctx);
    std::cout << "5-----model run done!!!" << std::endl; 

    std::vector<std::vector<float>> boxes;
    boxes = postprocess(mtcnn_ctx, src_img);

	std::stringstream index;
	index << img_idx;

    std::ofstream f_result_orange;
    std::ofstream f_result_apple;
    std::ofstream f_result_pear;
    std::ofstream f_result_potato;
    f_result_orange.open("./det2d_results/orange.txt", std::ios::app);
    f_result_apple.open("./det2d_results/apple.txt", std::ios::app);
    f_result_pear.open("./det2d_results/pear.txt", std::ios::app);
    f_result_potato.open("./det2d_results/potato.txt", std::ios::app);

	for (int i = 0; i < boxes.size(); ++i)
	{
		float xmin = boxes[i][0];
		float ymin = boxes[i][1];
		float xmax = xmin + boxes[i][2];
		float ymax = ymin + boxes[i][3];
		int type = boxes[i][4];
        float confidence = boxes[i][5];
        if (type == 0) {
            f_result_orange << images.at(img_idx) << " " << confidence << " " << xmin 
                            << " " << ymin << " " << xmax << " " << ymax << "\n";
        }
        else if (type == 1)
        {
            f_result_apple << images.at(img_idx) << " " << confidence << " " << xmin 
                            << " " << ymin << " " << xmax << " " << ymax << "\n";
        }
        else if (type == 2)
        {
            f_result_pear << images.at(img_idx) << " " << confidence << " " << xmin 
                            << " " << ymin << " " << xmax << " " << ymax << "\n";
        }
        else
        {
            f_result_potato << images.at(img_idx) << " " << confidence << " " << xmin 
                            << " " << ymin << " " << xmax << " " << ymax << "\n";
        }

		cv::rectangle(src_img, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax),
		              cv::Scalar(0, 255, 255), 3, 1, 0);
	}

	cv::imwrite("result/result_00" + index.str() + ".jpg", src_img);

    f_result_orange.close();
    f_result_apple.close();
    f_result_pear.close();
    f_result_potato.close();
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
        predict_detnet(&mtcnn_ctx, img, img_idx, images);
        time_end = get_current_time();
        std::cout << "detnet cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;
    }
    mtcnn_deinit(&mtcnn_ctx);

    std::cout << "End of game!!!" << std::endl;

    return 0;
}
