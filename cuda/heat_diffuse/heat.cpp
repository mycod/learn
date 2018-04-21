#include "stdlib.h"
#include "color.h"
#include <iostream>
#include <string>
#include "gl\glut.h"
using namespace std;

#define DIM 800 // 位图尺寸
#define MAX_TEMP 1.0f // 热源温度
#define MIN_TEMP 0.0001f 
#define SOURCE_NUM 1500
#define SPEED   0.25f// 扩散速度
#define rnd(a, b) rand()%(b-a)+a

unsigned char *output_img = new unsigned char[DIM*DIM * 4];
float *before_diffuse = new float[DIM*DIM]; // 用来保存热传导计算之前的数据
float *after_diffuse = new float[DIM*DIM]; // 用来保存热传导计算之后的数据

void debug(float *data, int row, int col, string info){
	cout << info + "============================= " << endl;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++){
			cout << data[i*row+j] << "   ";
		}
		cout << endl;
	}
}

void debug1(unsigned char *data, int length, string info){
	cout << info + "============================= " << endl;
	for (int i = 0; i < length; i++)
	{
		cout << int(data[i]) << "   ";
	}
	cout << endl;
}




// 恒温源结构体
struct HeatSource{
	// 用来存储热源的位置
	int(*location)[2] = new int[SOURCE_NUM][2];

	// 随机指定个生成热源
	HeatSource(){
		int count = 0;
		while (true){
			int x = rnd(192, 632);
			int y = rnd(184, 572);
			// 检查是否有重复位置
			for (int j = 0; j < count; j++){
				if (location[j][0] == x && location[j][1] == y){ continue; }
			}
			location[count][0] = x;
			location[count][1] = y;
			count++;
			if (count == SOURCE_NUM){ break; }
		}
	}

	~HeatSource(){ delete[] location; }
};
// 初始化热源
HeatSource heat;

// 添加热源：将热源处数值重置为MAX_TEMP
void add_source(HeatSource &heat_source){
	for (int r = 0; r < SOURCE_NUM; r++){
		// 取得每一个热源的x，y坐标
		int &x = heat_source.location[r][0];
		int &y = heat_source.location[r][1];
		// 将位图上该点的值修改为MAX_TEMP
		before_diffuse[x*DIM + y] = MAX_TEMP;
	}
}

// 一次热源扩散操作
void diffuse(){
	for (int r = 0; r < DIM; r++){
		for (int c = 0; c < DIM; c++){
			int offset = r*DIM + c;
			int bottom = r == DIM - 1 ? offset : (r + 1)*DIM + c;
			int top = r == 0 ? offset : (r - 1)*DIM + c;
			int left = c == 0 ? offset : r *DIM + c - 1;
			int right = c == DIM - 1 ? offset : r *DIM + c + 1;
			after_diffuse[offset] = before_diffuse[offset] + SPEED*(before_diffuse[top] + before_diffuse[bottom] + before_diffuse[left] + before_diffuse[right] - 4 * before_diffuse[offset]);
		}
	}
}

// 判断是否已经到达热平衡
bool balanced(){
	for (int i = 1; i < DIM*DIM; i++){
		if (before_diffuse[i] != before_diffuse[i - 1]){ return false; }
	}
	return true;
}

// 将热度值转化为颜色
void trans_color(){
	for (int i = 1; i < DIM*DIM; i++){
		// 将热度值转化为温度并存储在输出数组中 offset就是i
		float_to_color(output_img, after_diffuse, i);
	}
}

// 输出画面
void display(){
	// 清除屏幕
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	//像素读图
	//debug1(output_img, DIM*DIM, "display_image");
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, output_img);
	//双缓存交换缓存以显示图像
	glutSwapBuffers();
}

// 空闲函数
void idle_func(void) {
	if (balanced())
	{
		return;
	}
	// 交换数据
	//for (int i = 1; i < DIM*DIM; i++){ before_diffuse[i] = after_diffuse[i]; }
	float *temp = nullptr;
	temp = before_diffuse;
	before_diffuse = after_diffuse;
	after_diffuse = temp;
	temp = nullptr;
	// 添加热源
	add_source(heat);
	// 扩散计算
	diffuse();
	// 换算颜色
	trans_color();
	glutPostRedisplay();
}


int main(int argc, char *argv[]){
	// 初始化记录数组
	for (int i = 0; i < DIM*DIM; i++){ before_diffuse[i] = after_diffuse[i] = MIN_TEMP; }
	//debug(before_diffuse, DIM, DIM, "before_diffuse_init");
	//debug(after_diffuse, DIM, DIM, "after_diffuse_init");
	// 将热源放入
	add_source(heat);
	//debug(before_diffuse, DIM, DIM, "before_diffuse_add_source");
	// 扩散
	diffuse();
	//debug(before_diffuse, DIM, DIM, "before_diffuse_diffuse");
	//debug(after_diffuse, DIM, DIM, "after_diffuse_diffuse");
	// 转换颜色
	trans_color();
	//debug1(output_img, DIM*DIM, "display_image");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("heat_diffusion_cpu");
	glutDisplayFunc(display);
	glutIdleFunc(idle_func);
	glutMainLoop();
	delete &heat;
	delete before_diffuse;
	delete after_diffuse;
	delete output_img;
}






