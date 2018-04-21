#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 采用 <image.rows, image.cols>个线程，
// 声明核函数
__global__ void trans_kernel(const unsigned char *dev_rgb_image, unsigned char *dev_grey_image, int row, int col)
{
	int x = blockIdx.x; //row
	int y = threadIdx.x; //column
	if (x < row && y < col)
	{
		unsigned char b = dev_rgb_image[(x*blockDim.x + y) * 3];
		unsigned char g = dev_rgb_image[(x*blockDim.x + y) * 3 + 1];
		unsigned char r = dev_rgb_image[(x*blockDim.x + y) * 3 + 2];
		dev_grey_image[x*blockDim.x + y] = (r * 299 + g * 587 + b * 114 + 500) / 1000;
	}

}

//处理函数，rgb_image和grey_image是图像数据区域指针
extern "C" int trans_grey_cuda(const unsigned char *rgb_image, unsigned char *grey_image, int row, int col)
{
	unsigned char *dev_rgb_image;
	unsigned char *dev_grey_image;
	// 在GPU上分配内存：rgb图及灰度图
	int size = row * col;
	cudaMalloc((void**)&dev_rgb_image, size * 3);
	cudaMalloc((void**)&dev_grey_image, size * 1);
	// 将CPU上rgb图信息复制到GPU上
	cudaMemcpy(dev_rgb_image, rgb_image, size * 3, cudaMemcpyHostToDevice);
	// 调用核函数
	trans_kernel << < row, col >> >(dev_rgb_image, dev_grey_image, row, col);
	// 将得到的结果复制回主存
	cudaMemcpy(grey_image, dev_grey_image, size, cudaMemcpyDeviceToHost);
	// 释放显存
	cudaFree(dev_rgb_image);
	cudaFree(dev_grey_image);
	return 0;
}
