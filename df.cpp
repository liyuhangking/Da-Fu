#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cmath>
#include <string>
#include <cstring>
#include <queue>

using namespace cv;
using namespace std;

void findContours(Mat&frame);
void time(Mat&frame);
void drawcircle(Mat&frame);
int LeastSquaresCircleFitting(vector<Point2d> &m_Points, Point2d &Centroid, double &dRadius);
float getDistance(CvPoint pointO, CvPoint pointA);
 vector<Point2d> points;

int main()
{
  Mat frame;
  VideoCapture capture("RH.avi"); 
  	// if(!capture.isOpened())  // check if we succeeded
		// return -1;
	

  while (1)
  { 
    capture >> frame;
   
       if(frame.empty())
               break;
    findContours(frame);
    time(frame);
    namedWindow("frame",WINDOW_AUTOSIZE);
    imshow("frame", frame);
    waitKey(1);
  }
  return 0;
}

void findContours(Mat&frame)
{

    Mat dstImage1, dstImage2,dstImage3,dstImage4;
  
  //split color channels
    vector<Mat> imgChannels;
    split(frame,imgChannels);
    Mat midImage=imgChannels.at(2)-imgChannels.at(0);
    threshold(midImage,dstImage1,50,255,CV_THRESH_BINARY);
  //CLOSE
    Mat element2 = getStructuringElement(MORPH_RECT,Size(7,7));
    morphologyEx(dstImage1,dstImage2, MORPH_CLOSE, element2);
  //expend
    Mat element1=getStructuringElement(MORPH_RECT,Size(5,5));
    dilate(dstImage2, dstImage3,element1);

    //   Mat element3 = getStructuringElement(MORPH_RECT,Size(7,7));
    //  erode(dstImage3,dstImage4,element3);
    // namedWindow("dstImage3", WINDOW_AUTOSIZE);
     //imshow("dstImage3", dstImage3);

    vector<vector<Point>>contours;
    vector<Vec4i>hierarchy;
   

    findContours(dstImage2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));//extract all contours and establish a mesh contour structure
    int contour[2000] = { 0 };
  for (int i = 0; i < contours.size(); i++)//traverse all profiles
  {
    if (hierarchy[i][3] != -1) //with embedded profile，it indicates that it is a sub contour
    {
      contour[hierarchy[i][3]]++; //record the sub contour
    }
  }
  for (int j = 0; j < contours.size(); j++)//traverse all profiles
  {
    if (contour[j] == 1)//如果某轮廓对应数组的值为1，说明只要一个内嵌轮廓    
    {
      int num = hierarchy[j][2]; //record the embedded profile of profile
      RotatedRect box = minAreaRect(contours[num]);//contains all points of the profile
      Point2f vertex[4];
      box.points(vertex);//save the lower left corner,upper left corner,upper right corner,lower right corner into the point set 
      for (int i = 0; i < 4; i++)
      {
        line(frame, vertex[i], vertex[(i + 1) % 4], Scalar(255, 0, 0), 4, LINE_AA); //draw line
      }
  
      points.push_back(box.center);
			Point2d c;//圆心
			double r = 0;//半径
			LeastSquaresCircleFitting(points, c, r);//拟合圆
			circle(frame, c, r, Scalar(0, 0, 255), 2, 8);//绘制圆
			circle(frame, c, 6, Scalar(255, 0, 0), -1, 8);//绘制圆心
      Point2f center = (vertex[0] + vertex[2]) / 2;
      circle(frame,center,3,Scalar(43, 46, 255),FILLED);
      putText(frame, "target", center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0));//print word
    }
    if(contour[j] == 3)
     {
       vector<int> countersbox;
       int countersareabox[3];

       int temp[3];
       int index = 0;

       for(int k=0;k<contours.size();k++)//traverse all profiles
       {
         if(hierarchy[k][3]==j)//numbering sub profiles
         {
           countersbox.push_back(k);
         }

       }
       for(int l=0;l<3;l++)
       {
         int area = contourArea(contours[countersbox[l]]);//calculate sub contour area
         countersareabox[l]=area;
         temp[l]=area;
       }

       sort(temp,temp+3);//put it in order
  
       for(int s=0;s<3;s++)
       {
         if(countersareabox[s]==temp[0])
         {
           index = countersbox[s];
         }
       }
       
          int num= index; 
          RotatedRect box= minAreaRect(contours[num]); 
          Point2f vertex[4];
          box.points(vertex);
          Point2f center = (vertex[0] + vertex[2]) / 2;
       for (int i = 0; i < 4; i++)
       { 
          line(frame, vertex[i], vertex[(i + 1) % 4], Scalar(255, 0, 0), 4, LINE_AA);
          circle(frame,center,3,Scalar(43, 46, 255),FILLED);
        
       }
      points.push_back(box.center);
			Point2d c;//圆心
			double r = 0;//半径
			LeastSquaresCircleFitting(points, c, r);//拟合圆
			circle(frame, c, r, Scalar(0, 0, 255), 2, 8);//绘制圆
			circle(frame, c, 6, Scalar(255, 0, 0), -1, 8);//绘制圆心
       putText(frame, "destroyed", center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0));
     }
  } 
}

void time(Mat&frame)
{
  char str[20];
  double fps;
  double t_f,t_n;
  t_f=(double)getTickCount();
  t_n=((double)getTickCount()-t_f)/getTickFrequency();
  fps=0.000001/t_n;
  string FPSstring("FPS:");
  sprintf(str,"%.2f",fps);
  FPSstring+=str;
  putText(frame,FPSstring,Point(5,40),FONT_HERSHEY_SIMPLEX,1.5,Scalar(36,70,255));
}

// float getDistance(CvPoint pointO, CvPoint pointA)
// {
//     float distance;
//     distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
//     distance = sqrtf(distance);
//     return distance;
// }


  int LeastSquaresCircleFitting(vector<Point2d> &m_Points, Point2d &Centroid, double &dRadius)//拟合圆函数
{
	if (!m_Points.empty())
	{
		int iNum = (int)m_Points.size();
		if (iNum < 3)	return 1;
		double X1 = 0.0;
		double Y1 = 0.0;
		double X2 = 0.0;
		double Y2 = 0.0;
		double X3 = 0.0;
		double Y3 = 0.0;
		double X1Y1 = 0.0;
		double X1Y2 = 0.0;
		double X2Y1 = 0.0;
		vector<Point2d>::iterator iter;
		vector<Point2d>::iterator end = m_Points.end();
		for (iter = m_Points.begin(); iter != end; ++iter)
		{
			X1 = X1 + (*iter).x;
			Y1 = Y1 + (*iter).y;
			X2 = X2 + (*iter).x * (*iter).x;
			Y2 = Y2 + (*iter).y * (*iter).y;
			X3 = X3 + (*iter).x * (*iter).x * (*iter).x;
			Y3 = Y3 + (*iter).y * (*iter).y * (*iter).y;
			X1Y1 = X1Y1 + (*iter).x * (*iter).y;
			X1Y2 = X1Y2 + (*iter).x * (*iter).y * (*iter).y;
			X2Y1 = X2Y1 + (*iter).x * (*iter).x * (*iter).y;
		}
		double C = 0.0;
		double D = 0.0;
		double E = 0.0;
		double G = 0.0;
		double H = 0.0;
		double a = 0.0;
		double b = 0.0;
		double c = 0.0;
		C = iNum * X2 - X1 * X1;
		D = iNum * X1Y1 - X1 * Y1;
		E = iNum * X3 + iNum * X1Y2 - (X2 + Y2) * X1;
		G = iNum * Y2 - Y1 * Y1;
		H = iNum * X2Y1 + iNum * Y3 - (X2 + Y2) * Y1;
		a = (H * D - E * G) / (C * G - D * D);
		b = (H * C - E * D) / (D * D - G * C);
		c = -(a * X1 + b * Y1 + X2 + Y2) / iNum;
		double A = 0.0;
		double B = 0.0;
		double R = 0.0;
		A = a / (-2);
		B = b / (-2);
		R = double(sqrt(a * a + b * b - 4 * c) / 2);
		Centroid.x = A;
		Centroid.y = B;
		dRadius = R;
		return 0;
	}
	else
		return 1;

	return 0;
}


 