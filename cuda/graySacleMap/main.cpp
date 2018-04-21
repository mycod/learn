#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
using namespace cv;

extern "C" cudaError_t trans_grey_cuda(const unsigned char *rgb_image, unsigned char *grey_image, int row, int col);

int main(){
	// 读取图片及相关信息
	Mat rgb_image;
	rgb_image = imread("Tulips.jpg");
	// 在CPU上创建灰度图片所用空间
	Mat grey_image;
	grey_image.create(rgb_image.size(), CV_8UC1);
	// 调用GPU函数进行处理
	int res = 1;
	res = trans_grey_cuda(rgb_image.data, grey_image.data, rgb_image.rows, rgb_image.cols);
	// 显示图像
	if (res){ return 0; }
	//namedWindow("Gray_Transform_WithGPU");
	imshow("Gray_Transform_WithGPU", grey_image);
	//将图像写入路径
	waitKey(0);
	return 0;
}