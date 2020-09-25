#include "inference/common/image_process.h"

void image_resize_square(const cv::Mat &src, cv::Size dst_size, cv::Mat &dst_image)
{

}

void preprocess(nnctrl_ctx_t *nnctrl_ctx, const cv::Mat &src_mat, 
                const cv::Size &dst_size, const int resize_type)
{
    if(src_mat.empty()){
        return;
    }
    cv::Mat dst_mat;
    std::vector<cv::Mat> channel_s;
    if(resize_type == 0){
        cv::resize(src_mat, dst_mat, dst_size, 0, 0, cv::INTER_NEAREST);
    }
    else if(resize_type == 1){
        image_resize_square(src_mat, dst_mat, dst_size);
    }
    
    cv::split(dst_mat, channel_s);
    for (int i=0; i<3; i++)
    {
        memcpy(nnctrl_ctx->net.net_in.in_desc[0].virt + i * height * width, channel_s[2-i].data, height * width); // bgr2rgb
    }
    
    // sycn address
    cavalry_mem_sync_cache(nnctrl_ctx->net.net_in.in_desc[0].size, nnctrl_ctx->net.net_in.in_desc[0].addr, 1, 0);
}