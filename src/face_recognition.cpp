#include "cv.h"
#include "highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <cstdlib>


using namespace cv;
using namespace std;


int main(int argc, char *argv[]) 
{
    //OpenCV video capture object
    VideoCapture camera;

    //OpenCV image object
    Mat image;
    Mat img;

    //Initialize face detector object
    CascadeClassifier face_detect;
    face_detect.load("haarcascade_frontalface_default.xml");

    //camera id . Associated to device number in /dev/videoX
    int cam_id; 
	
    //check user args
    switch(argc)
    {
	case 1: //no argument provided, so try /dev/video0
		cam_id = 0;  
		break; 
	case 2: //an argument is provided. Get it and set cam_id
		cam_id = atoi(argv[1]);
		break; 
	default: 
		cout << "Invalid number of arguments. Call program as: webcam_capture [video_device_id]. " << endl; 
		cout << "EXIT program." << std::endl; 
		break; 
    }

    //advertising to the user 
    cout << "Opening video device " << cam_id << endl;

    //open the video stream and make sure it's opened
    if( !camera.open(cam_id) ) 
	{
        cout << "Error opening the camera. May be invalid device id. EXIT program." << endl;
        return -1;
    }

    //capture loop. Out of user press a key
    while(1)
	{
	
	//Read image and check it
        if(!camera.read(image)) 
	{
            cout << "No frame" << endl;
            waitKey();
        }
        
	//Copy frame for visualization
    	img = image.clone();

	// Convert the current frame to grayscale:
        Mat gray;
        cvtColor(img, gray, CV_BGR2GRAY);

	
	//Detect faces as rectangles
	vector< Rect_<int> > faces;
    	face_detect.detectMultiScale(gray, faces);
	
	

	//For each detected face...
	for (int i = 0; i < faces.size(); i++)
	{
		
	    // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
	    // Write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(img, face_i, CV_RGB(0, 255,0), 1);


	}
	
	

	//show image in a window
        imshow("faces", img);

	//Waits 1 millisecond to check if a key has been pressed. If so, breaks the loop. Otherwise continues.
        if(waitKey(1) >= 0) break;
    }   
}
