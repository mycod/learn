#include <Windows.h>  
#include <highgui/highgui.hpp>
#include <cmath>
#define dim 500

using namespace cv;

//定义复数结构体
struct cucomplex
{
	float r;
	float i;
	cucomplex(float a, float b) :r(a), i(b) {}
	float magenitude(void) { return sqrt(r*r+i*i); }
	cucomplex operator* (const cucomplex& a)
	{
		return cucomplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cucomplex operator+ (const cucomplex& a)
	{
		return cucomplex(r + a.r, i + a.i);
	}
};

//定义julia函数
bool julia(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(dim / 2 - x) / (dim / 2);
	float jy = scale * (float)(dim / 2 - y) / (dim / 2);
	cucomplex c(-0.8, 0.156);
	cucomplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magenitude() > 1000) return false;
	}
	return true;
}


//main函数
int main()
{
	Mat image = Mat(Size(dim, dim), CV_8UC3, Scalar::all(10));
	for (int x = 0; x < dim; x++)
	{
		for (int y = 0; y < dim; y++)
		{
			if (julia(x, y) == true)
			{
				image.at<Vec3b>(x, y) = Vec3b(0, 0, 255);
			}
			else
			{
				image.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
			}
		}
	}
	imshow("julia picture", image);
	waitKey();
}