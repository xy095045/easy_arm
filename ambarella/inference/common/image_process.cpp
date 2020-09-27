#include "inference/common/image_process.h"

void image_resize_square(const cv::Mat &src, cv::Size dst_size, cv::Mat &dst_image)
{
    const int src_width = src.cols;
    const int src_height = src.rows;
    const int dst_width = dst_size.width;
    const int dst_height = dst_size.height;
    const float ratio = std::min(static_cast<float>(dst_width) / src_width, \
                                 static_cast<float>(dst_height) / src_height);
    const int new_width = static_cast<int>(src_width * ratio);
    const int new_height = static_cast<int>(src_height * ratio);
    const int pad_width = dst_width - new_width;
    const int pad_height = dst_height - new_height;
    const int top = pad_height / 2;
    const int bottom = pad_height - (pad_height / 2);
    const int left = pad_width / 2;
    const int right = pad_width - (pad_width / 2);
    cv::Mat resize_mat;
    cv::resize(src, resize_mat, cv::Size(pad_width, pad_height), 0, 0, cv::INTER_NEAREST);
    cv::copyMakeBorder(resize_mat, dst_image, top, bottom, left, right, cv::INTER_NEAREST, cv::Scalar(0, 0, 0));
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
        image_resize_square(src_mat, dst_size, dst_mat);
    }
    
    cv::split(dst_mat, channel_s);
    for (int i=0; i<3; i++)
    {
        memcpy(nnctrl_ctx->net.net_in.in_desc[0].virt + i * dst_size.height * dst_size.width, \
         channel_s[2-i].data, dst_size.height * dst_size.width); // bgr2rgb
    }
    
    // sycn address
    cavalry_mem_sync_cache(nnctrl_ctx->net.net_in.in_desc[0].size, nnctrl_ctx->net.net_in.in_desc[0].addr, 1, 0);
}