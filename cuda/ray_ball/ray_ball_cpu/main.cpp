#include <cmath>
#include <highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

#define INF 2e10f 
#define rnd(x) (x*rand()/RAND_MAX)  
#define SPHERES 40 //球体数量
#define DIM 800    //图像尺寸

struct Sphere
{
	float r, g, b;
	float radius;
	float x, y, z;

	float hit(float source_x, float source_y, float *n)
	{
		float dx = source_x - x;
		float dy = source_y - y;

		if (dx*dx + dy*dy < radius*radius)
		{
			float dz = sqrt(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrt(radius*radius);
			return dz + z;
		}
		return -INF;
	}
};

Sphere s[SPHERES];


int main()
{
	// 新建显示的source
	Mat source = Mat(DIM, DIM, CV_8UC3, Scalar::all(0));
	// 创建小球
	Sphere *balls = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		balls[i].r = rnd(255.0f);
		balls[i].g = rnd(255.0f);
		balls[i].b = rnd(255.0f);
		balls[i].x = rnd(1000.0f) - 500;
		balls[i].y = rnd(1000.0f) - 500;
		balls[i].z = rnd(1000.0f) - 500;
		balls[i].radius = rnd(100.0f) + 20;
	}

	// 以row, col形式遍历source图像，并对每一个像素进行计算和赋值
	for (int row = 0; row < source.rows; row++)
	{
		for (int col = 0; col < source.cols; col++)
		{
			// 获得像素点指针
			int offset = (row*source.cols + col) * 3;
			uchar *pixel = source.data + offset;
			// 设置计算用x, y值
			float x = row - DIM / 2;
			float y = col - DIM / 2;
			float bright = 0;// 用来临时放每个球对应的亮度
			float bright_maxDis = 0; // 记录离原点最远处小球的亮度，即我们要的值
			float max_distance_z = -INF;
			float distance_z = -INF;
			float r = 0, g = 0, b = 0;
			// 遍历每一个球，得到照射点离原点最远的球体颜色
			for (int i = 0; i < SPHERES; i++)
			{
				distance_z = balls[i].hit(x, y, &bright);
				if (distance_z > max_distance_z)
				{
					bright_maxDis = bright;
					max_distance_z = distance_z;
					r = balls[i].r*bright_maxDis;
					g = balls[i].g*bright_maxDis;
					b = balls[i].b*bright_maxDis;
				}
			}
			// 遍历完小球后，将得到的球体颜色值给源图对应像素点
			*pixel = (int)b;
			*(pixel+1) = (int)g;
			*(pixel+2) = (int)r;
		}
	}
	// 整图遍历完成后，显示图像
	imshow("ray_trace_cpu", source);
	waitKey();
	cin.get();
	return 0;
}

