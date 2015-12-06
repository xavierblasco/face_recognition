#include "cv.h"
#include "highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <cstdlib>
#include "opencv2/imgproc/imgproc.hpp" //To Image Processing


using namespace cv;
using namespace std;


int main(int argc, char *argv[]) 
{
    //OpenCV video capture object
    VideoCapture camera;

    //OpenCV image object
    Mat image; //Original image
    Mat img;   //Cloned image
    Mat gray; //Used when convert image to gray 

    //Load Hat and Moustache pictures
    Mat hat = imread("../img/hat.png", -1); // load hat picture with aplpha channel
    Mat hat_resized; //hat image once resized
    Mat moustache = imread("../img/moustache.png", -1); // load moustache picture with aplpha channel
    Mat moustache_resized; //moustache image once resized

    //Define 3 variables to get pixels combination on hat and moustache fussion
    double color_pixel_0, color_pixel_1, color_pixel_2;



    if(! hat.data ) // Check for invalid input image 
    {
        cout <<  "Could not open or find the image <- hat.png ->" << endl ;
        return -1;
    }

    if(! moustache.data ) // Check for invalid input image 
    {
        cout <<  "Could not open or find the image <- moustache.png ->" << endl ;
        return -1;
    }




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
	    
	    
            //Add hat and moustache to picture
	    int facew = face_i.width; //face width
	    int faceh = face_i.height; //face height
        Size hat_size(facew,faceh); //resize hat picture giving same face width (hat picture is squared)
        resize(hat, hat_resized, hat_size );

        Size moustache_size(facew/2,faceh/2); //resize moustache picture giving half face width because moustache pictures is squared
        resize(moustache, moustache_resized, moustache_size );

        double hat_locate = 0.50; //Variable to move up hat from face position
        double moustache_locate_y = 0.50; //Variable to move down moustache from face position
        double moustache_move_x = (facew - moustache_resized.size[0])/2; //Variable to move right moustache from face position


	    //Overlay hat and moustache
        for ( int j = 0; j < faceh ; j++)
	    {
                for ( int k = 0; k < facew; k++)
                {


                    // determine the opacity of the foregrond pixel, using its fourth (alpha) channel for pictures picture.
                    double alpha_hat = hat_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
                    color_pixel_0 = (hat_resized.at<cv::Vec4b>(j, k)[0] * (alpha_hat)) + ((img.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0])* (1.0-alpha_hat));
                    color_pixel_1 = (hat_resized.at<cv::Vec4b>(j, k)[1] * (alpha_hat)) + ((img.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1])* (1.0-alpha_hat));
                    color_pixel_2 = (hat_resized.at<cv::Vec4b>(j, k)[2] * (alpha_hat)) + ((img.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2])* (1.0-alpha_hat));



                    if((face_i.y +j-(faceh*hat_locate))>0){
                        img.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0] = color_pixel_0 ;
                        img.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1] = color_pixel_1 ;
                        img.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2] = color_pixel_2 ;
                    }

                    if((j<(faceh/2))&&(k<(facew/2))){
                        // determine the opacity of the foregrond pixel, using its fourth (alpha) channel for moustache picture.
                        double alpha_moustache = moustache_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
                        color_pixel_0 = (moustache_resized.at<cv::Vec4b>(j, k)[0] * (alpha_moustache)) + ((img.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0])* (1.0-alpha_moustache));
                        color_pixel_1 = (moustache_resized.at<cv::Vec4b>(j, k)[1] * (alpha_moustache)) + ((img.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1])* (1.0-alpha_moustache));
                        color_pixel_2 = (moustache_resized.at<cv::Vec4b>(j, k)[2] * (alpha_moustache)) + ((img.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2])* (1.0-alpha_moustache));


                        img.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0] = color_pixel_0 ;
                        img.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1] = color_pixel_1 ;
                        img.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2] = color_pixel_2 ;

                    }


                }
	    	
         }
	    
	    

		
	}
	
	

	//show image in a window
        imshow("faces", img);

	//Waits 1 millisecond to check if a key has been pressed. If so, breaks the loop. Otherwise continues.
        if(waitKey(1) >= 0) break;
    }   
}
