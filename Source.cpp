#include<opencv2/opencv.hpp>
#include<cstdio>
#include "iostream"
#include<math.h>

using namespace cv;
using namespace std;

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;


int main() {

	Mat src = imread("13.png");
	Mat dst1, dst2;
	Mat	dst_con = src.clone();
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(5, 5));

	cvtColor(src, src, COLOR_BGR2GRAY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;


	int widthLimit = src.channels() * src.cols;
	for (int height = 0; height < src.rows; height++) {        
		for (int width = 0; width < widthLimit; width++) {
			uchar X_Y = src.at<uchar>(height, width);
			if (X_Y > 140) src.at<uchar>(height, width) = 255;
			else if (X_Y < 112)src.at<uchar>(height, width) = 0;
			else
			{
				X_Y += (130 - X_Y) * 3;
				src.at<uchar>(height, width) = X_Y;
			}
		}
	}

	blur(src, src, Size(5, 5));
	morphologyEx(src, src, MORPH_OPEN, Mat(), Point(-1, -1), 3);

	Sobel(src, grad_x, CV_16S, 1, 0, 3, 2, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  
	Sobel(src, grad_y, CV_16S, 0, 1, 3, 2, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	medianBlur(abs_grad_x, abs_grad_x, 5);
	medianBlur(abs_grad_y, abs_grad_y, 5);

	addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, dst1);
	erode(dst1, dst1, Mat(), Point(-1, -1));


	threshold(dst1, dst2, 30, 255, THRESH_BINARY);
	erode(dst2, dst2, Mat(), Point(-1, -1));

	findContours(dst2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i], true);
		cout << area << endl;
		if (area > 18000)
		{
		drawContours(dst_con, contours, i, 255, 2, 8, hierarchy);
		}

	}

	imshow("origin", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("contours", dst_con);

	waitKey(0);
}

