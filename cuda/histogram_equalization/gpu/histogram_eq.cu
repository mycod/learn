
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// 定义计算直方图的核函数
__global__ void dev_histo_kernel(uchar *dev_img, int *dev_histo, int size){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y;
	int offset = x + y * (blockDim.x*gridDim.x);
	// 使用块共享内存
	__shared__ int temp[256];
	// 初始化共享内存并同步
	temp[y] = 0;
	__syncthreads();
	// 对应像素值的统计处+1, 此处没有限制block数量
	if (offset < size){
		atomicAdd(&temp[dev_img[offset]], 1);
	}
	// 等所有的线程都完成像素值相加后，再进行所有block的共享temp相加
	__syncthreads();
	if (threadIdx.x == 0){
		atomicAdd(&(dev_histo[y]), temp[y]);
	}
	
}


// 定义计算累积分布函数的核函数
/*
# 正向计算				
span = 2				
循环开始：			while(span<=length	
	看每一位是否满足限制条件：（index+1）% span == 0 			
		满足条件则：加上前面距离间隔为span/2的值		
		若不满足条件则：		
	同步线程，等待一轮完成			
	span *= 2			
				
# 反向计算				
span = span / 8				
循环:		while(span > = 1)		
	看每一位是否满足条件：（index+1) % span == 0 且 (index+1) > span*2			
		满足：		
			满足 （index+1) % (span*2)!= 0 并且 index != length-1:	
				其值添加前面距离间隔为span 的值
		不满足：		
			保持原值	
	同步			
	span = span /2			
*/
// 采用对半相加再换算的方法

//__global__ void accumulate(int *dev_histo, int length){
//	int index = threadIdx.x;
//	int span = 2;
//	while (span <= length){
//		if ((index + 1) % span == 0){
//			dev_histo[index] += dev_histo[index - span / 2];
//		}
//		__syncthreads();
//		span *= 2;
//	}
//
//	span = span / 8;
//	while (span >= 1){
//		if ((index + 1) % span == 0 && (index + 1) > span * 2){
//			if ((index + 1) % (span * 2) != 0 && index != length - 1){
//				dev_histo[index] += dev_histo[index - span];
//			}
//		}
//		__syncthreads();
//		span /= 2;
//	}
//}

/*
span=1	
while( span  < length)	
	index大于 (span-1)时：
	
同步	
span*2	
*/
// 使用依次循环相加

__global__ void accumulate(int *dev_histo, int length){
	int index = threadIdx.x;
	int span = 1;
	while (span < length){
		if (index >(span - 1)){
			dev_histo[index] += dev_histo[index - span];
		}
		__syncthreads();
		span *= 2;
	}
}

// 定义计算转换像素值的核函数
__global__ void pixel_change(int *histo, uchar *pixel_changed, int size){
	int index = threadIdx.x;
	// 先找到最小像素
	__shared__ int accu_minPixel;
	if (histo[index] != 0 && histo[index - 1] == 0){
		accu_minPixel = histo[index];
	}
	// 对每一个像素值进行转换
	pixel_changed[index] = round((histo[index] - accu_minPixel) / float(size - accu_minPixel) * 255);
}

// 定义将图像像素替换的核函数
__global__ void image_trans(uchar *dev_img, uchar *pixel_changed)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	uchar img_pixel = dev_img[index];
	dev_img[index] = pixel_changed[img_pixel];
}

int main(int argc, char *argv[]){

	// 读取图片
	Mat image = imread("D:\\CUDA_program\\demo\\histogram_equalization\\cpu\\demo.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int size = image.rows * image.cols;
	// 计算消耗时间
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	// 为图片像素数据分配GPU内存
	uchar *dev_img;
	cudaMalloc((void**)&dev_img, sizeof(uchar)*size);
	// 将图片数据复制到GPU内存中
	cudaMemcpy(dev_img, image.data, sizeof(uchar)*size, cudaMemcpyHostToDevice);
	// 为直方图统计和颜色转换分配GPU内存
	int *dev_histo;
	uchar *dev_pixel_changed;
	cudaMalloc((void**)&dev_histo, sizeof(int) * 256);
	cudaMalloc((void**)&dev_pixel_changed, sizeof(uchar) * 256);
	// 初始化
	cudaMemset(dev_histo, 0, sizeof(int) * 256);
	cudaMemset(dev_pixel_changed, (uchar)0, sizeof(uchar) * 256);

	// 启动计算直方图的核函数
	int blocks = image.rows;
	dim3 threads(image.cols/256, 256);
	dev_histo_kernel<< <blocks, threads>> >(dev_img, dev_histo, size);

	

	//int debug_histo[256];
	//cudaMemcpy(debug_histo, dev_histo, sizeof(int) * 256, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout << debug_histo[i] << '\t';
	//}

	// 启动计算累积分布的函数
	accumulate << <1, 256 >> >(dev_histo, 256);

	//int debug_accu[256];
	//cudaMemcpy(debug_accu, dev_histo, sizeof(int) * 256, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout << debug_accu[i] << '\t';
	//}


	// 启动计算像素换算的函数
	pixel_change << <1, 256 >> >(dev_histo, dev_pixel_changed, size);

	//uchar debug_changed[256];
	//cudaMemcpy(debug_changed, dev_pixel_changed, 256, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout << (int)debug_changed[i] << '\t';
	//}

	// 启动转换图像的函数
	image_trans << <blocks, image.cols >> >(dev_img, dev_pixel_changed);
	// 将转换后的图片数据复制回来
	cudaMemcpy(image.data, dev_img, size, cudaMemcpyDeviceToHost);
	// 显示所用的时间
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, end);
	cout << "Time: " << elapsedTime << "ms" << endl;
	// 显示图片
	imshow("after_equalization", image);

	waitKey();
	cudaFree(dev_histo);
	cudaFree(dev_img);
	cudaFree(dev_pixel_changed);
	return 0;
}

