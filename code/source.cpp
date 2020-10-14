/*
// File name               : source.cpp
//
// Desscription            : Contactless Fingerprint detection. The code segments hand from the image frame and then 
//							 removes palm data. By finding contours, ROI is drawm on the fingerprints. 
//
// Copyright               : Copyright © 2020 RAR
//
// Author                  : Rakshit Adderi Rohit
//
// Date                    : 06/June/2020
*/

#include <opencv2/opencv.hpp>
#include <iostream>

#define FINGER_WIDTH 60
#define CONTOUR_SIZE 200

using namespace std;
using namespace cv;

int intensity_l = 4;
int intensity_h = 30;
int palm_size = FINGER_WIDTH;

class finger_print_detection
{
public:
	Mat skin_tone_segmentation(Mat src)
	{
		Mat temp1, skin, gray;
		// Converting source image to grayscale
		cvtColor(src, gray, CV_BGR2GRAY);

		Mat temp2(Size(src.cols, src.rows), CV_8UC1);
		temp1 = src.clone();
		for (int i = 0; i < temp1.rows; i++)
		{
			for (int j = 0; j < temp1.cols; j++)
			{
				temp1.at<Vec3b>(i, j)[2] = 0; // Removing red channel
				// Selecting intensity value of either bule or green channel
				// Converting 3 channel image to 1 channel image
				if (temp1.at<Vec3b>(i, j)[0] >= temp1.at<Vec3b>(i, j)[1])
					temp2.at<uchar>(i, j) = temp1.at<Vec3b>(i, j)[0];
				else
					temp2.at<uchar>(i, j) = temp1.at<Vec3b>(i, j)[1];
				// Subtracting grayscale of the source image with the 1 channel image generated
				temp2.at<uchar>(i, j) = gray.at<uchar>(i, j) - temp2.at<uchar>(i, j);
			}
		}
		// Segmentation of skin in the image captured
		inRange(temp2, Scalar(intensity_l), Scalar(intensity_h), skin);
		return (skin);
	}

	Mat palm_deletion(Mat with_pam)
	{
		Mat no_palm;
		// Removal of palm
		no_palm = with_pam.clone();

		int dist; // Width of human hand in a row
		int begin, end; // Start and End pixels of the human hand in a row
		for (int i = 0; i < with_pam.rows; i++)
		{
			dist = 0;
			begin = 0;
			end = 0;
			for (int j = 0; j < with_pam.cols; j++)
			{
				if (with_pam.at<uchar>(i, j) == 255)
				{
					dist++; // Calculating width of human hand in a row
					if (dist == 1)
						begin = j; // Start pixel
					else if (dist > 1)
						end = j; // End pixel

					if (j == (with_pam.cols - 1))
					{
						// Delete human hand data except fingers in a row
						// This is a special condition when the human hand is at the edge of the image
						for (int k = begin; k <= end; k++)
							no_palm.at<uchar>(i, k) = 0;
					}
				}
				else
				{
					// Delete human hand data except fingers in a row
					if ((end - begin) > palm_size)
					{
						for (int k = begin; k <= end; k++)
							no_palm.at<uchar>(i, k) = 0;
					}
					// Reset parametrs for next row
					dist = 0;
					begin = 0;
					end = 0;
				}
			}
		}
		return(no_palm);
	}

	void finger_print_roi(Mat src, Mat fingers)
	{
		Mat temp;

		vector <vector<Point>> contours;
		vector <Vec4i> heirarchy;
		// To find all the contours
		findContours(fingers, contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

		for (int icontours = 0; icontours < contours.size(); icontours++)
		{
			if (contours[icontours].size() > CONTOUR_SIZE) // Discarding smaller contours
			{
				// Retreving rotated rectangle data over a contour
				RotatedRect finger_rectroi = minAreaRect(contours[icontours]); 
				
				Point2f vertices[4];
				// Vertices of the rotated rectangle
				finger_rectroi.points(vertices);
				
				// Modifying centroid, height/width of the rectangle to fit rectangle on the fingerprints
				if (finger_rectroi.size.width > finger_rectroi.size.height)
				{
					finger_rectroi.size.width = (float)(1.5)*finger_rectroi.size.height;
					finger_rectroi.center = (vertices[2] + vertices[3]) * 0.5 + (vertices[0] - vertices[3]) * 0.2;
				}
				else if (finger_rectroi.size.width <= finger_rectroi.size.height)
				{
					finger_rectroi.size.height = (float)(1.5) *finger_rectroi.size.width;
					finger_rectroi.center = (vertices[1] + vertices[2]) * 0.5 + (vertices[0] - vertices[1]) * 0.2;
				}
				else
				{
				}
				// find vertices of ROI
				finger_rectroi.points(vertices);
				// Draw ROI on fingerprints
				for (int i = 0; i < 4; i++)
					line(src, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
			}
		}
		imshow("Fingerprint", src);
	}
};

void main()
{
	finger_print_detection object;

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		cout << "Error" << endl;

	// Create control panel window
	namedWindow("Control", CV_WINDOW_AUTOSIZE);

	// Create trackbars
	createTrackbar("intensity_l", "Control", &intensity_l, 255);
	createTrackbar("intensity_h", "Control", &intensity_h, 255);

	for (;;)
	{
		Mat frame_rgb, hand, fingers;
		cap >> frame_rgb; // get a new frame from camera

		// Filter to remove any niose
		//medianBlur(frame_rgb, frame_rgb, 5);

		// Skin tone detection
		hand = object.skin_tone_segmentation(frame_rgb);
		// Smoothening of segmented image
		medianBlur(hand, hand, 15);
		imshow("With palm", hand);

		// Removal of palm
		fingers = object.palm_deletion(hand);
		imshow("No Palm", fingers);

		// To find contours and draw ROI on finger prints
		object.finger_print_roi(frame_rgb, fingers);
		if (waitKey(1) >= 0) break;
	}
	cap.release();
}