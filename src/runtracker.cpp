#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

//#define VIDEOFILE "/home/vince/Videos/jedaah/2_4km_compressed.avi"
#define VIDEOFILE "/home/vince/Videos/space/1Drone_2.mp4"
#define ANNOTATIONSFILE "/home/vmachado/Videos/tracking/small_targets/video07-24.vatic.txt"
#define ANNOTATIONSOUTFILE "/tmp/out.struck.txt"
#define DATAASSOCIATIONPERIOD 25000
//#define OUTVIDEOFILE "/home/vince/Videos/jedaah/2_4km_tracking_kcf.avi"
#define OUTVIDEOFILE "/home/vince/Videos/space/test_kfc_movement_hog_false.avi"

using namespace std;
using namespace cv;


static cv::Scalar getColorFromTrackerId(int id){
    std::vector<int> colors = {2,2,2,2,1,1,1,1,0,0,0,0};
    auto b = 127*colors[(id*3)%12];
    auto g = 127*colors[(id*5)%12];
    auto r = 127*colors[(id*7)%12];
    return cv::Scalar(b,g,r);
}


int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = false;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;
    VideoWriter video_out;

    VideoCapture cap;
    cap.open(VIDEOFILE);



	map<int, Rect> gt;
/*    gt[207] = cv::Rect(238,371,15,20);
    gt[1890] = cv::Rect(208,370,15,20);
    gt[2949] = cv::Rect(223,372,15,20);
    gt[3445] = cv::Rect(348,373,15,15);
    gt[4010] = cv::Rect(400,374,15,15);
*/
    gt[1] = cv::Rect(871,342,30,25);


    Scalar color = getColorFromTrackerId(1);
    bool initialized=false;
    for(int i=0; ; i++){
        cout << "i: " << i << endl;
        cap >> frame;
        if(!frame.data) {
            break;
        }
        if(i==0) {
            video_out.open(OUTVIDEOFILE, CV_FOURCC('D', 'I', 'V', 'X'), 25, frame.size());
        }
        if(!initialized && gt[i].area()<=0) {
            continue;
        }
        if(gt[i].area()>0){ //DA
            initialized=false;
        }

        if(!initialized) {
            color = getColorFromTrackerId(i+1);
            tracker.init( gt.at(i), frame );
            rectangle( frame, gt.at(i), Scalar( 0, 255, 255 ), 1, 8 );
            initialized=true;
            continue;
        }

		result = tracker.update(frame);
		rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), color, 2, 8 );

		imshow("Image", frame);
        video_out << frame;
		waitKey(1);
	}
}
