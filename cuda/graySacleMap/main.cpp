#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
using namespace cv;

extern "C" cudaError_t trans_grey_cuda(const unsigned char *rgb_image, unsigned char *grey_image, int row, int col);

int main(){
	// ��ȡͼƬ�������Ϣ
	Mat rgb_image;
	rgb_image = imread("Tulips.jpg");
	// ��CPU�ϴ����Ҷ�ͼƬ���ÿռ�
	Mat grey_image;
	grey_image.create(rgb_image.size(), CV_8UC1);
	// ����GPU�������д���
	int res = 1;
	res = trans_grey_cuda(rgb_image.data, grey_image.data, rgb_image.rows, rgb_image.cols);
	// ��ʾͼ��
	if (res){ return 0; }
	//namedWindow("Gray_Transform_WithGPU");
	imshow("Gray_Transform_WithGPU", grey_image);
	//��ͼ��д��·��
	waitKey(0);
	return 0;
}