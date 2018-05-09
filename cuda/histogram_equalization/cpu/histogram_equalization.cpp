#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include <ctime>
//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) // 设置入口地址  

using namespace std;
using namespace cv;

int *histo = new int[256](); // 记录直方图信息，累计信息
int *pixel_changed = new int[256](); // 记录变换后像素值

int main(int argc, char *argv[]){
	

	// 读入图片
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	int size = image.rows * image.cols;
    clock_t start, end;
    start = clock();
	// 统计直方图的信息
	int pixel_min = 255;
	for (int i = 0; i < size; i++)
	{
		int pixel = image.data[i];
		histo[pixel]++;
		if (pixel < pixel_min)
		{
			pixel_min = pixel;
		}
	}
	// 计算得到累计信息
	int temp = 0;
	for (int i = 0; i < 256; i++)
	{
		if (histo[i] != 0)
		{
			histo[i] += temp;
			temp = histo[i];
		}
	}
	// 转换对应的像素值
	int tst = histo[pixel_min];
	for (int i = 0; i < 256; i++)
	{
		if (histo[i] != 0)
		{
			pixel_changed[i] = round((histo[i] - histo[pixel_min]) / float((size - histo[pixel_min])) * 255);
		}
	}
	// 转换图像像素并输出
	for (int i = 0; i < size; i++)
	{
		int pixel = image.data[i];
		image.data[i] = pixel_changed[pixel];
	}
    end = clock();
	cout << "cost time:" << end - start << " ms" << endl;
	imshow("img_afte_histogramEqualization", image);

	waitKey();
	delete histo;
	delete pixel_changed;
	return 0;
}
