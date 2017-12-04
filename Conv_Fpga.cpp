#include "Conv_Fpga.h"


template <typename Dtype>
void Conv_Fpga<Dtype>::Perform16x162DConv(CONV_ENGINE engine)
{
    _ASSERT(IS_ARG_SET());

    output_h = (16 - kernel_h) / stride_h + 1;
    output_w = (16 - kernel_w) / stride_w + 1;
    kernel_size = kernel_h * kernel_w;
    out_buf = new Dtype[output_h * output_w * kernel_n];
    memset(out_buf, 0, sizeof(Dtype) * output_h * output_w * kernel_n);
    switch (engine) 
    {
    case DIRECT:
        Direct2DConv();
        break;
    case GEMM:
        col_buf = new Dtype[kernel_size * output_h * output_w];
        krnls_row_buf = new Dtype[kernel_n * kernel_size];
        GEMM2DConv();
        break;
    case WINOGRAD:
        NOT_IMPLEMENTED("WINOGRAD");
        break;
    default:
        break;
    }
}
template void Conv_Fpga<int>::Perform16x162DConv(CONV_ENGINE engine);
template <typename Dtype>
void Conv_Fpga<Dtype>::Direct2DConv()
{
    Dtype *temp = new Dtype[kernel_n];

    for (int img_h = 0; img_h < output_h; img_h += stride_h)
    {
        for (int img_w = 0; img_w < output_w; img_w += stride_w)
        {
            memset(temp, (Dtype)0, sizeof(Dtype) * kernel_n);
            for (int krnl_h = 0; krnl_h < kernel_h; ++krnl_h)
            {
                for (int krnl_w = 0; krnl_w < kernel_w; ++krnl_w)
                {
                    for (int ch = 0; ch < kernel_n; ++ch)
                    {
                        temp[ch] += kernels[ch][krnl_h][krnl_w] * (data_buf[img_h + krnl_h][img_w + krnl_w]);
                    }
                }
            }
            for (int ch = 0; ch < kernel_n; ++ch)
            {
                *(out_buf + ch * output_h * output_w + output_w * img_h + img_w) = temp[ch];
            }
        }
    }
}

template <typename Dtype>
void Conv_Fpga<Dtype>::GEMM2DConv()
{
    im2col();
    kernel2row();
    matrixMultiply();
}

template <typename Dtype>
void Conv_Fpga<Dtype>::im2col()
{
    int count = 0;
    for (int img_h = 0; img_h < output_h; img_h += stride_h)
    {
        for (int img_w = 0; img_w < output_w; img_w += stride_w)
        {
            
            for (int krnl_h = 0; krnl_h < kernel_h; ++krnl_h)
            {
                for (int krnl_w = 0; krnl_w < kernel_w; ++krnl_w)
                {

                    *(col_buf + (krnl_h * kernel_w + krnl_w) * output_h * output_w + count) 
                        = data_buf[img_h + krnl_h][img_w + krnl_w];
                }
            }
            ++count;
        }
    }
}

template <typename Dtype>
void Conv_Fpga<Dtype>::kernel2row()
{
    for (int krnl_n = 0; krnl_n < kernel_n; ++krnl_n)
    {
        memcpy(krnls_row_buf + krnl_n * kernel_size, kernels[krnl_n], sizeof(int) * kernel_size);
    }
}
template <typename Dtype>
void Conv_Fpga<Dtype>::matrixMultiply()
{
    // Multiply the matrix krnls_row_buf with col_buf
        
    for (int row = 0; row < kernel_n; ++row)
    {
        for (int col = 0; col < output_h * output_w; ++col)
        {
            for (int inner = 0; inner < kernel_size; ++inner)
            {
                *(out_buf + row * output_w * output_h + col) +=
                    *(krnls_row_buf + row * kernel_size + inner) * *(col_buf + inner * output_h * output_w + col);
            }
        }
    }
}

