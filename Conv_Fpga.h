#pragma once
#include <string>
#include <cstdlib>
#include <iostream>

#define NOT_IMPLEMENTED(feature_name) {std::cerr << "ERROR:This feature \
    (" << feature_name << ") has not been implemented." << std::endl; exit(1);}
template <typename Dtype> class Conv_Fpga;
enum CONV_ENGINE
{
    DIRECT, GEMM, WINOGRAD
};
template <typename Dtype=int>
class Conv_Fpga
{

public:
    template<typename Dtype=int>
    Conv_Fpga() :is_arg_set(false),
        pad_h(0), pad_w(0), stride_h(0), stride_w(0),
        output_h(0), output_w(0)
    {
        col_buf = new Dtype[16 * 16];
        kernels[0][0][0] = 0;
        kernels[0][0][1] = 1;
        kernels[0][0][2] = 0;
        kernels[0][1][0] = 1;
        kernels[0][1][1] = 0;
        kernels[0][1][2] = 1;
        kernels[0][2][0] = 0;
        kernels[0][2][1] = 1;
        kernels[0][2][2] = 0;
        
        kernels[1][0][0] = 1;
        kernels[1][0][1] = 1;
        kernels[1][0][2] = 1;
        kernels[1][1][0] = 1;
        kernels[1][1][1] = 1;
        kernels[1][1][2] = 1;
        kernels[1][2][0] = 1;
        kernels[1][2][1] = 1;
        kernels[1][2][2] = 1;
    }

    ~Conv_Fpga()
    {
        delete data_buf;
        delete col_buf;
    }
    inline bool SetData(Dtype *data, bool reset_arg = false) 
    { 
        memcpy(data_buf, data, sizeof(Dtype) * 256);
        if (reset_arg)
        {
            is_arg_set = 0;
        }
        return true;
    }
    inline bool SetPaddingSize(int h, int w) { pad_h = h; pad_w = w; ++is_arg_set; return true; }
    inline bool SetStrideSize(int h, int w) { stride_h = h; stride_w = w; ++is_arg_set; return true; }
    inline void GetOutputSize(int *krnl_n, int *h, int *w) { *krnl_n = kernel_n; *h = output_h; *w = output_w; }
    inline bool GetResult(Dtype *data) { memcpy(data, out_buf, sizeof(Dtype) * output_h * output_w * kernel_n); return true; }
    void Perform16x162DConv(CONV_ENGINE engine);

private:
    inline bool IS_ARG_SET() { return is_arg_set == 2; }
    void Direct2DConv();
    void GEMM2DConv();
    void im2col();
    void kernel2row();
    void matrixMultiply();
    Dtype data_buf[16][16], *out_buf;
    Dtype *col_buf, *krnls_row_buf, *mat_mul_buf;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int output_h, output_w;
    int is_arg_set;

    static const int kernel_n = 2, kernel_h = 3, kernel_w = 3;
    int kernels[kernel_n][kernel_h][kernel_w];
    int kernel_size, img_size = 256;
};


