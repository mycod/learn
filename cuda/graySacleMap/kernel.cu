#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ���� <image.rows, image.cols>���̣߳�
// �����˺���
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

//��������rgb_image��grey_image��ͼ����������ָ��
extern "C" int trans_grey_cuda(const unsigned char *rgb_image, unsigned char *grey_image, int row, int col)
{
	unsigned char *dev_rgb_image;
	unsigned char *dev_grey_image;
	// ��GPU�Ϸ����ڴ棺rgbͼ���Ҷ�ͼ
	int size = row * col;
	cudaMalloc((void**)&dev_rgb_image, size * 3);
	cudaMalloc((void**)&dev_grey_image, size * 1);
	// ��CPU��rgbͼ��Ϣ���Ƶ�GPU��
	cudaMemcpy(dev_rgb_image, rgb_image, size * 3, cudaMemcpyHostToDevice);
	// ���ú˺���
	trans_kernel << < row, col >> >(dev_rgb_image, dev_grey_image, row, col);
	// ���õ��Ľ�����ƻ�����
	cudaMemcpy(grey_image, dev_grey_image, size, cudaMemcpyDeviceToHost);
	// �ͷ��Դ�
	cudaFree(dev_rgb_image);
	cudaFree(dev_grey_image);
	return 0;
}
