#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

//#include "/home/vmachado/OpenCV3.3.1/include/opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


#define VIDEOFILE "/home/vmachado/Videos/edgybees/sample.mp4"
///#define VIDEOFILE "/home/vmachado/Videos/edgybees/Clip_1.mov"
//#define VIDEOFILE "/home/vince/Videos/space/1Drone_2.mp4"
//#define VIDEOFILE "/home/vmachado/Videos/edgybees/DTB70/Girl2/video.mp4"
//#define ANNOTATIONSFILE "/home/vmachado/Videos/edgybees/DTB70/Girl2/groundtruth_rect.txt"
#define ANNOTATIONSOUTFILE "/tmp/out.struck.txt"
#define DATAASSOCIATIONPERIOD 25000
#define OUTVIDEOFILE "/tmp/aaa.avi"
//#define OUTVIDEOFILE "/home/vince/Videos/space/test_kfc_movement_hog_false.avi"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static cv::Scalar getColorFromTrackerId(int id){
    std::vector<int> colors = {2,2,2,2,1,1,1,1,0,0,0,0};
    auto b = 127*colors[(id*3)%12];
    auto g = 127*colors[(id*5)%12];
    auto r = 127*colors[(id*7)%12];
    return cv::Scalar(b,g,r);
}


static void readDTB79dataset(string filename, map<int, Rect> & detections){
    fstream in(filename);
    cout << filename << endl;
    int i;
    float x, y, w, h;
    while(in >> x >> y >> w >> h) {
        detections[i++] = cv::Rect2f(x, y, w, h);
    }
}



static Mat _calcHomography(Mat &_a, Mat &_b, Point2d centre, int radius){

    //TODO validate radius (check if circle is inside the image)

    Mat a, b;

    auto scale_x = 0.25;
    auto scale_y = 0.25;
    cv::resize(_a, a, cv::Size(0,0), scale_x, scale_y);
    cv::resize(_b, b, cv::Size(0,0), scale_x, scale_y);
    centre.x *= scale_x;
    centre.y *= scale_y;
    radius *= scale_x; ////// TODO

    Ptr<SURF> detector = SURF::create(100);
//    Ptr<SURF> extractor;

    cv::BFMatcher matcher(cv::NORM_L2,true);
    std::vector<KeyPoint> keypoints_a, keypoints_b;
    Mat descriptors_a, descriptors_b;
    std::vector<DMatch> matches;


    Mat mask = Mat::ones(a.size(), CV_8U);  // type of mask is CV_8U
    circle(mask, centre, radius, Scalar::all(0), -1);


    ///auto t1 = cv::getTickCount();
    detector->detectAndCompute(a, mask, keypoints_a, descriptors_a);
    ///auto t2 = cv::getTickCount();
    ///cout << "time: for a detectAndCompute: " << (t2-t1) / cv::getTickFrequency() << endl;
    detector->detectAndCompute(b, mask, keypoints_b, descriptors_b);

//    extractor->compute(a, keypoints_a, descriptors_a);
//    extractor->compute(b, keypoints_b, descriptors_b);

    ///auto t3 = cv::getTickCount();
    matcher.match(descriptors_a, descriptors_b, matches);
    ///auto t4 = cv::getTickCount();
    ///cout << "time: for a matcher.match: " << (t4-t3) / cv::getTickFrequency() << endl;


    cout << "matches size: " << matches.size() << endl;

    if(matches.size() > 100) {
        std::sort(matches.begin(), matches.end());
        // 100 best matches
        matches.erase(matches.begin() + 100, matches.end());
    }

//    Mat img_matches;
//    drawMatches(a, keypoints_a, b, keypoints_b,
//                matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//    cv::imshow("matchs", img_matches);
//    cv::waitKey();

    // Extract location of good matches
    std::vector<Point2f> points1, points2;

    for( size_t i = 0; i < 20 /*matches.size()*/; i++ )
    {
        points1.push_back( keypoints_a[ matches[i].queryIdx ].pt );
        points2.push_back( keypoints_b[ matches[i].trainIdx ].pt );
    }

    // Find homography
    Mat H = findHomography( points1, points2, RANSAC );

    Mat S = Mat::eye(3,3,CV_64F);
    S.at<double>(0,0) = 1/scale_x;
    S.at<double>(1,1) = 1/scale_y;
    Mat H2 = S* H * S.inv();

    return H2;
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
    Mat prevFrame;

	// Tracker results
	Rect result;
    VideoWriter video_out;

    VideoCapture cap;
    cap.open(VIDEOFILE);



	map<int, Rect> gt;
//    gt[214] = cv::Rect(705,195,45,45);
    gt[811] = cv::Rect(477,474,32,25);
//    gt[544] = cv::Rect(63,440,70,28);


//    gt[83] = cv::Rect(983,215,25,25);


//    readDTB79dataset(ANNOTATIONSFILE, gt);


    Scalar color = getColorFromTrackerId(1);
    bool initialized=false;
    for(int i=0; ; i++){
        cout << "i: " << i << endl;
        /// cout << gt[i] << endl;
        cap >> frame;
        if(i==0) prevFrame = frame;


        if(!frame.data) {
            break;
        }
        //if(i==0) {
        //    video_out.open(OUTVIDEOFILE, CV_FOURCC('D', 'I', 'V', 'X'), 25, frame.size());
        //}
        if(!initialized && gt[i].area()<=0) {
            prevFrame = frame.clone();
            continue;
        }


//        if(gt[i].area()>0){ //DA
//            initialized=false;
//        }

        if(!initialized) {
            color = getColorFromTrackerId(i+1);
            tracker.init( gt.at(i), frame );
            rectangle( frame, gt.at(i), Scalar( 0, 255, 255 ), 1, 8 );
            initialized=true;
            prevFrame = frame.clone();
            ///result = gt.at(i);
            continue;
        }


        ////////////
        auto H = _calcHomography(
                prevFrame,
                frame,
                Point2d(result.x+0.5*result.width, result.y+0.5*result.height),
                MAX(result.width, result.height));

        prevFrame = frame.clone();


        std::vector<Point2f> a_corners(4), b_corners(4);

        a_corners[0] = cvPoint(result.x, result.y);
        a_corners[1] = cvPoint(result.x+result.width, result.y);
        a_corners[2] = cvPoint(result.x, result.y+result.height);
        a_corners[3] = cvPoint(result.x+result.width, result.y+result.height);


        perspectiveTransform(a_corners, b_corners, H);
        cv::Rect_<float> b_rect;
        b_rect.x = b_corners[0].x;
        b_rect.y = b_corners[0].y;
        b_rect.width  = b_corners[3].x - b_corners[0].x;
        b_rect.height = b_corners[3].y - b_corners[0].y;

        //rectangle( frame, b_rect, cv::Scalar(255,0,0), 2, 8 );
        ////////////


        if(b_rect.x > frame.cols || b_rect.y > frame.rows || b_rect.x < 0 || b_rect.y < 0) {
            //initialized=false;
            //continue;
            b_rect = result;
            cout << "b_rect out of image" << b_rect << endl;
        }
        if(b_rect.width < 0 || b_rect.height <0){
            b_rect = result;
            cout << "b_rect out of image II" << endl;
        }

        result = tracker.update(frame, b_rect); // passes new roi projected to H
        rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), color,2, 8);

		imshow("Image", frame);
        //video_out << frame;
		waitKey(0);
	}
}
