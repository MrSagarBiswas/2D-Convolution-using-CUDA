#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
// #include <cuda/cuda_runtime.h>
#include <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__constant__ long int filter[5000];

__global__ void convolution_kernel(long int *matrix, long int *result, int h, int w, int c, int r, int s, int k)
{
    extern __shared__ long int SM[];
    long int *s_matrix = SM;
    long int *s_result = SM + w;

    int filter_row = blockIdx.z;
    int filter_Number = blockIdx.y/c;
    int channel_number = blockIdx.y%c;
    int fid = r*s*c*filter_Number + r*s*channel_number + s*filter_row;

    int col = threadIdx.x;
    int row = blockIdx.x + filter_row - r/2;

    int m = channel_number*h + row;
    int n = col - s/2;

    s_matrix[col] = (row>=0 && row<h) ? matrix[m*w + col] : 0;
    s_result[col] = 0;

    __syncthreads();

    int rid = h*w*filter_Number + w*blockIdx.x + col;

    for(int i=0; i<s; ++i){
      if(n+i>=0 && n+i<w && col<w)
        s_result[col] += s_matrix[n+i]*filter[fid + i];
    }
    
    atomicAdd((unsigned long long int*)&result[rid], (unsigned long long int)s_result[col]);
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];


    auto start = std::chrono::high_resolution_clock::now(); 

    long int *matrix, *result;

    cudaMalloc(&matrix, w*h*c*sizeof(long int));
    cudaMalloc(&result, h*w*k*sizeof(long int));

    cudaMemcpy(matrix, h_mat, w*h*c*sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(filter, h_filter, r*s*c*k*sizeof(long int));
    cudaMemset(result, 0, h*w*k*sizeof(long int));

    dim3 grid(h, k*c, r);

    convolution_kernel<<<grid, w, (2*w)*sizeof(long int)>>>(matrix, result, h, w, c, r, s, k);

    cudaMemcpy(h_ans, result, h*w*k*sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(matrix);
    cudaFree(result);

    auto end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed1 = end - start;


    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
