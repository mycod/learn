#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) // 设置入口地址  

using namespace std;
using namespace cv;




void histo_equalize(uchar *source, int size){
	int *histo = new int[256](); // 记录直方图信息，累计信息
	int *pixel_changed = new int[256](); // 记录变换后像素值
		// 统计直方图的信息
	int pixel_min = 255;
	for (int i = 0; i < size; i++)
	{
		int pixel = source[i];
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
		int pixel = source[i];
		source[i] = pixel_changed[pixel];
	}
	delete histo;
	delete pixel_changed;
}

int main(int argc, char *argv[]){
	// 读入图片
	Mat image = imread(argv[1]);
	int size = image.rows * image.cols;
	uchar *b = new uchar[size];
	uchar *g = new uchar[size];
	uchar *r = new uchar[size];
	for(int i = 0; i < size; i++){
		b[i] = image.data[3*i];
		g[i] = image.data[3*i+1];
		r[i] = image.data[3*i+2];
	}
	histo_equalize(b, size);
	histo_equalize(g, size);
	histo_equalize(r, size);
	for(int i = 0; i < size; i++){
		image.data[3*i] = b[i];
		image.data[3*i+1] = g[i];
		image.data[3*i+2] = r[i];
	}
	imshow("img_afte_histogramEqualization_rgb", image);
	waitKey();
	return 0;
}
