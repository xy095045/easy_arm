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

// #include "iav_ioctl.h"
#include "cavalry_ioctl.h"

#include "cavalry_mem.h"
#include "vproc.h"
#include "nnctrl.h"
#include "utils.h"

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
#define segnet_num 200000
#define SegOutputWidth 500
#define SegOutputHeight 400
#define SegOutputChannel 2
#define THRESH 0.3

 struct net_match {
	uint8_t net_id;

	struct net_run_cfg net_rev;
	struct net_mem net_m;

    struct net_cfg cfg;
    struct net_input_cfg net_in;
    struct net_output_cfg net_out;
    struct net_result result;

    char net_in_name[MAX_FILE_NAME_LEN];
    char net_out_name[MAX_FILE_NAME_LEN];

    char net_file[MAX_FILE_NAME_LEN];
};

// use index to find the netName
enum eNetName{
    eSegNet = 0,
};

typedef struct nnctrl_ctx_s {
    uint8_t verbose;
    uint8_t reuse_mem;
    uint8_t cache_en;
    uint8_t buffer_id;
    uint8_t log_level;

    struct net_match PNet[net_num];

    float segnet_feature[segnet_num];;
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

	nnctrl_ctx->PNet[netId].net_out.out_num = 1;
	nnctrl_ctx->PNet[netId].net_out.out_desc[0].name = nnctrl_ctx->PNet[netId].net_out_name;
	nnctrl_ctx->PNet[netId].net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
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

    strcpy(nnctrl_ctx->PNet[eSegNet].net_in_name, "0");
    strcpy(nnctrl_ctx->PNet[eSegNet].net_out_name, "507");
    strcpy(nnctrl_ctx->PNet[eSegNet].net_file, "./segnet.bin"); 

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

int nnctrl_run_segnet(mtcnn_ctx_t *mtcnn_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx;

    rval = nnctrl_run_net(nnctrl_ctx->PNet[eSegNet].net_id, &nnctrl_ctx->PNet[eSegNet].result, NULL, NULL, NULL);

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_run_net() failed, return %d\n", rval);
    }

    // parse the output of classnet
    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->PNet[eSegNet].net_m.mem_size, nnctrl_ctx->PNet[eSegNet].net_m.phy_addr, 0, 1);
    }

    float *score_addr = (float *)(nnctrl_ctx->PNet[eSegNet].net_m.virt_addr
        + nnctrl_ctx->PNet[eSegNet].net_out.out_desc[0].addr - nnctrl_ctx->PNet[eSegNet].net_m.phy_addr);

    int output_c = nnctrl_ctx->PNet[eSegNet].net_out.out_desc[0].dim.depth;
    int output_h = nnctrl_ctx->PNet[eSegNet].net_out.out_desc[0].dim.height;
    int output_w = nnctrl_ctx->PNet[eSegNet].net_out.out_desc[0].dim.width;
    int output_p = nnctrl_ctx->PNet[eSegNet].net_out.out_desc[0].dim.pitch;

    std::cout << "poutput size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
                                  << output_w << "--output_p: " << output_p << "--" << std::endl;

    // FILE *fp = fopen("./output.bin", "w");
    // std::cout << "sizeof: " << sizeof(score_addr) << " " << std::endl;
    // fwrite(score_addr, sizeof(float), 400*504, fp);
    // fclose(fp);

    float seg_output[SegOutputHeight * 504];
    memcpy(seg_output, score_addr, SegOutputHeight * 504 * sizeof(float));

    for (int h = 0; h < output_h; h++)
    {
        memcpy(nnctrl_ctx->segnet_feature + h * output_w, seg_output + h * 504, output_w * sizeof(float));
    }

    return rval;
}

void preprocess(mtcnn_ctx_t *mtcnn_ctx, cv::Mat &srcRoI, int netId)
{
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx;

    // int channel = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.depth;
    int height = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.height;
    int width = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.width;
    int pitch = nnctrl_ctx->PNet[netId].net_in.in_desc[0].dim.pitch;
    // std::cout << "--height: " << height << "--width: " << width << "--pitch: " << pitch << "--" << std::endl;

    cv::Size szNet = cv::Size(width, height);
    // cv::Mat detRoI = srcRoI;

    cv::resize(srcRoI, srcRoI, szNet, 0, 0); //cv::INTER_LINEAR
    // detRoI = detRoI - a * b

    // give data virtual address to input net
    // // To Tensor 
    // double m = 0.0, std = 255.0;
    // detRoI = (detRoI - m) / (0.00001 + std);
    // // Normalize
    // double mean = {0.5070, 0.4865, 0.4409}, stdv = {0.2673, 0.2564, 0.2761};
    // detRoI = (detRoI - mean) / (0.00001 + stdv);

    std::vector<cv::Mat> channel_s;
    cv::split(srcRoI, channel_s);

    for (int i=0; i<3; i++)
    {
        for (int h=0; h<height; h++)
        {
            memcpy(nnctrl_ctx->PNet[netId].net_in.in_desc[0].virt + i * height * pitch + h * pitch, 
                   channel_s[i].data + h * width, width);
        }
    }
    
    // sycn address
    cavalry_mem_sync_cache(nnctrl_ctx->PNet[netId].net_in.in_desc[0].size, nnctrl_ctx->PNet[netId].net_in.in_desc[0].addr, 1, 0);
}

void postprocess(mtcnn_ctx_t *mtcnn_ctx, cv::Mat seg_mat)
{
    nnctrl_ctx_t *nnctrl_ctx = &mtcnn_ctx->nnctrl_ctx;
	float* output = nnctrl_ctx->segnet_feature;

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
