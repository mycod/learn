
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// �������ֱ��ͼ�ĺ˺���
__global__ void dev_histo_kernel(uchar *dev_img, int *dev_histo, int size){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y;
	int offset = x + y * (blockDim.x*gridDim.x);
	// ʹ�ÿ鹲���ڴ�
	__shared__ int temp[256];
	// ��ʼ�������ڴ沢ͬ��
	temp[y] = 0;
	__syncthreads();
	// ��Ӧ����ֵ��ͳ�ƴ�+1, �˴�û������block����
	if (offset < size){
		atomicAdd(&temp[dev_img[offset]], 1);
	}
	// �����е��̶߳��������ֵ��Ӻ��ٽ�������block�Ĺ���temp���
	__syncthreads();
	if (threadIdx.x == 0){
		atomicAdd(&(dev_histo[y]), temp[y]);
	}
	
}


// ��������ۻ��ֲ������ĺ˺���
/*
# �������				
span = 2				
ѭ����ʼ��			while(span<=length	
	��ÿһλ�Ƿ�����������������index+1��% span == 0 			
		���������򣺼���ǰ�������Ϊspan/2��ֵ		
		��������������		
	ͬ���̣߳��ȴ�һ�����			
	span *= 2			
				
# �������				
span = span / 8				
ѭ��:		while(span > = 1)		
	��ÿһλ�Ƿ�������������index+1) % span == 0 �� (index+1) > span*2			
		���㣺		
			���� ��index+1) % (span*2)!= 0 ���� index != length-1:	
				��ֵ���ǰ�������Ϊspan ��ֵ
		�����㣺		
			����ԭֵ	
	ͬ��			
	span = span /2			
*/
// ���ö԰�����ٻ���ķ���

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
	index���� (span-1)ʱ��
	
ͬ��	
span*2	
*/
// ʹ������ѭ�����

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

// �������ת������ֵ�ĺ˺���
__global__ void pixel_change(int *histo, uchar *pixel_changed, int size){
	int index = threadIdx.x;
	// ���ҵ���С����
	__shared__ int accu_minPixel;
	if (histo[index] != 0 && histo[index - 1] == 0){
		accu_minPixel = histo[index];
	}
	// ��ÿһ������ֵ����ת��
	pixel_changed[index] = round((histo[index] - accu_minPixel) / float(size - accu_minPixel) * 255);
}

// ���彫ͼ�������滻�ĺ˺���
__global__ void image_trans(uchar *dev_img, uchar *pixel_changed)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	uchar img_pixel = dev_img[index];
	dev_img[index] = pixel_changed[img_pixel];
}

int main(int argc, char *argv[]){

	// ��ȡͼƬ
	Mat image = imread("D:\\CUDA_program\\demo\\histogram_equalization\\cpu\\demo.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int size = image.rows * image.cols;
	// ��������ʱ��
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	// ΪͼƬ�������ݷ���GPU�ڴ�
	uchar *dev_img;
	cudaMalloc((void**)&dev_img, sizeof(uchar)*size);
	// ��ͼƬ���ݸ��Ƶ�GPU�ڴ���
	cudaMemcpy(dev_img, image.data, sizeof(uchar)*size, cudaMemcpyHostToDevice);
	// Ϊֱ��ͼͳ�ƺ���ɫת������GPU�ڴ�
	int *dev_histo;
	uchar *dev_pixel_changed;
	cudaMalloc((void**)&dev_histo, sizeof(int) * 256);
	cudaMalloc((void**)&dev_pixel_changed, sizeof(uchar) * 256);
	// ��ʼ��
	cudaMemset(dev_histo, 0, sizeof(int) * 256);
	cudaMemset(dev_pixel_changed, (uchar)0, sizeof(uchar) * 256);

	// ��������ֱ��ͼ�ĺ˺���
	int blocks = image.rows;
	dim3 threads(image.cols/256, 256);
	dev_histo_kernel<< <blocks, threads>> >(dev_img, dev_histo, size);

	

	//int debug_histo[256];
	//cudaMemcpy(debug_histo, dev_histo, sizeof(int) * 256, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout << debug_histo[i] << '\t';
	//}

	// ���������ۻ��ֲ��ĺ���
	accumulate << <1, 256 >> >(dev_histo, 256);

	//int debug_accu[256];
	//cudaMemcpy(debug_accu, dev_histo, sizeof(int) * 256, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout << debug_accu[i] << '\t';
	//}


	// �����������ػ���ĺ���
	pixel_change << <1, 256 >> >(dev_histo, dev_pixel_changed, size);

	//uchar debug_changed[256];
	//cudaMemcpy(debug_changed, dev_pixel_changed, 256, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout << (int)debug_changed[i] << '\t';
	//}

	// ����ת��ͼ��ĺ���
	image_trans << <blocks, image.cols >> >(dev_img, dev_pixel_changed);
	// ��ת�����ͼƬ���ݸ��ƻ���
	cudaMemcpy(image.data, dev_img, size, cudaMemcpyDeviceToHost);
	// ��ʾ���õ�ʱ��
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, end);
	cout << "Time: " << elapsedTime << "ms" << endl;
	// ��ʾͼƬ
	imshow("after_equalization", image);

	waitKey();
	cudaFree(dev_histo);
	cudaFree(dev_img);
	cudaFree(dev_pixel_changed);
	return 0;
}

