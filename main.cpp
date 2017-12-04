#include <iostream>
#include "Conv_Fpga.h"
#include <ctime>
#include <cstdlib>
using namespace std;
int I[16][16];
#define RANDOM
void dataInit()
{
#ifdef RANDOM
    srand(time(NULL));

    for (int h = 0; h < 16; ++h)
    {
        for (int w = 0; w < 16; ++w)
        {
            I[h][w] = rand() % 256;
        }
    }
#else
    for (int h = 0; h < 16; ++h)
    {
        for (int w = 0; w < 16; ++w)
        {
            I[h][w] = 2;
        }
    }
#endif
}
void printData()
{
    for (int h = 0; h < 16; ++h)
    {
        for (int w = 0; w < 16; ++w)
        {
            cout << I[h][w] << " ";
        }
        cout << endl; 
    }
}
void print(int *result, int kernel_n, int output_h, int output_w)
{
    for (int krnl_n = 0; krnl_n < kernel_n; ++krnl_n)
    {
        cout << "Kernel " << krnl_n << ":" << endl;
        for (int out_h = 0; out_h < output_h; ++out_h)
        {
            for (int out_w = 0; out_w < output_w; ++out_w)
            {
                cout << *(result + krnl_n * output_h * output_w + out_h * output_w + out_w) << " ";
            }
            cout << endl;
        }
    }
}

int main()
{
    int kernel_n, output_h, output_w;
    Conv_Fpga<int>* conv_fpga = new Conv_Fpga<int>;
    dataInit();
    cout << "Data:" << endl;
    printData();
    conv_fpga->SetData((int *)I);
    conv_fpga->SetPaddingSize(0, 0);
    conv_fpga->SetStrideSize(1, 1);
    conv_fpga->Perform16x162DConv(DIRECT);
    conv_fpga->GetOutputSize(&kernel_n, &output_h, &output_w);
    int *result_direct = new int[kernel_n * output_h * output_w];
    conv_fpga->GetResult(result_direct);
    cout << "DIRECT:" << endl;
    print((int *)result_direct, kernel_n, output_h, output_w);
    conv_fpga->Perform16x162DConv(GEMM);
    int *result_gemm = new int[kernel_n * output_h * output_w];
    conv_fpga->GetResult(result_gemm);
    cout << "GEMM:" << endl;
    print((int *)result_gemm, kernel_n, output_h, output_w);
    if (memcmp(result_direct, result_gemm, sizeof(int) * kernel_n * output_h * output_w) == 0)
    {
        cout << "Correct Result!" << endl;
    }
    else
    {
        cerr << "Incorrect Result!" << endl;
    }
    return 0;
}