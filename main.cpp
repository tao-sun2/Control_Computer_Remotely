//The frame of objectDetection is learned from 
//https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/objectDetection/objectDetection.cpp

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

double calculateCos(Point s, Point m, Point e);

void trackDetection();


/** Global variables */
CascadeClassifier fist_cascade;
CascadeClassifier hand_cascade;
vector<Point> points;
int name = 0;
int group1 = 0;
int group2 = 0;
int group3 = 0;
bool isHandPre = false;
bool isHandNow = false;
bool recordPoints = false;
Point currentPosition;

/** @function main */
int main(int argc, const char** argv)
{

	//-- 1. Load the cascades
	//Here I use a trained cascade xml file from the GitHub 
	//https://github.com/Aravindlivewire/Opencv/blob/8e69ca9fdba640486de61ccc639847ee0035464b/haarcascade/fist.xml 
	if (!fist_cascade.load("fist.xml"))
	{
		cout << "--(!)Error loading fist cascade\n";
		return -1;
	};

	//Here I use a trained cascade xml file from the GitHub 
	//https://github.com/Aravindlivewire/Opencv/blob/8e69ca9fdba640486de61ccc639847ee0035464b/haarcascade/hand.xml 
	if (!hand_cascade.load("hand.xml"))
	{
		cout << "--(!)Error loading hand cascade\n";
		return -1;
	};


	//-- 2. Read the video stream
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	Mat frame;
	while (capture.read(frame))
	{

		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		//-- 3. Apply the classifier to the frame
		isHandPre = isHandNow;
		detectAndDisplay(frame);
		if (isHandPre^isHandNow) {
			recordPoints = !recordPoints;
			if (!recordPoints) {
				trackDetection();
			}
		}


		int c = waitKey(1);
		if (c == 27)
		{
			break; // escape
		}
	}
	return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	bool temp1 = false;
	bool temp2 = false;

	//-- Detect hands
	std::vector<Rect> hands;
	hand_cascade.detectMultiScale(frame_gray, hands);

	for (size_t i = 0; i < hands.size(); i++)
	{
		isHandNow = false;
		temp2 = true;
		Point center(hands[i].x + hands[i].width / 2, hands[i].y + hands[i].height / 2);
		cout << "Hand_Center x: " << center.x << " Hand_Center y: " << center.y << endl;

		ellipse(frame, center, Size(hands[i].width / 2, hands[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		if (!isHandPre) {
			currentPosition = center;

			SetCursorPos(640 - center.x, center.y);
		}
	}

	//-- Detect fists
	std::vector<Rect> fists;
	fist_cascade.detectMultiScale(frame_gray, fists);

	for (size_t i = 0; i < fists.size(); i++)
	{
		if (!temp2) {
			isHandNow = true;
			temp1 = true;
			Point center(fists[i].x + fists[i].width / 2, fists[i].y + fists[i].height / 2);
			cout << "Fist_Center x: " << center.x << " Fist_Center y: " << center.y << endl;
			points.push_back(center);
			ellipse(frame, center, Size(fists[i].width / 2, fists[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4);
		}
	}


	//-- Show what you got
	Mat reverse_img;
	flip(frame, reverse_img, 1); //1代表水平方向旋转180度
	imshow("Capture", reverse_img);
}


/** @function calculateCos */
double calculateCos(Point s, Point m, Point e) {
	double x1 = s.x - m.x;
	double y1 = s.y - m.y;
	double x2 = e.x - m.x;
	double y2 = e.y - m.y;
	return (x1*x2 + y1*y2) / (sqrt(x1*x1 + y1*y1)*sqrt(x2*x2 + y2*y2));
}


/** @function trackDetection */
void trackDetection() {
	Mat image = Mat(480, 640, CV_8UC3);
	string s = to_string(name);
	name++;
	vector<int> hull;
	convexHull(Mat(points), hull, true);

	//准备参数
	int hullcount = (int)hull.size(); //凸包的边数
	Point point0 = points[hull[hullcount - 1]]; //连接凸包边的坐标点

	Point st = point0;
	double cosin = 0; 			//绘制凸包的边
	Point m;
	Point e;
	for (int i = 0; i < hullcount; i++) {
		Point point = points[hull[i]];
		line(image, point0, point, Scalar(0, 0, 255), 1, 8, 0);

		point0 = point;
		if (i<hullcount - 1) {
			m = point;
			e = points[hull[i + 1]];
			cosin = calculateCos(st, m, e);
			if (cosin >= 0.258819)						//75 degree
				group1++;
			else if (cosin >= -0.258819)	            // 105 degree
				group2++;
			else group3++;
		}
		st = m;
	}
	cosin = calculateCos(m, e, points[hull[0]]);
	if (cosin >= 0.258819)						//75 degree
		group1++;
	else if (cosin >= -0.258819)	            // 105 degree
		group2++;
	else group3++;

	s += ".png";
	Mat rev_img;
	flip(image, rev_img, 1);//1代表水平方向旋转180度
	imwrite(s.c_str(), rev_img);

	points.clear();
	cout << group1 << " " << group2 << " " << group3 << endl;


	if (group1 >= group2&&group1 >= group3) {
		cout << "triangle" << endl;
		SetCursorPos(640 - currentPosition.x, currentPosition.y);
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); //按下左键
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0); //松开左键
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); //按下左键
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0); //松开左键
	}
	else if (group3 > 6) {
		cout << "circle" << endl;
		SetCursorPos(640 - currentPosition.x, currentPosition.y);
		mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0); //按下右键
		Sleep(10);
		mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0); //松开右键
	}
	else {
		cout << "rectangle" << endl;
		SetCursorPos(640 - currentPosition.x, currentPosition.y);
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); //按下左键
		Sleep(10);
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0); //松开左键
	}

	cout << "reset" << endl;
	points.clear();
	group1 = 0;
	group2 = 0;
	group3 = 0;
};
