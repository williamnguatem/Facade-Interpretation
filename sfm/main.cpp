#include <vector>
#include <iostream>

#include "feauture_matcher.h"

//#include "SiftMatch.h"


int main()
{
	//sfm::FeatureMatcher a("E:\\Haus35\\sfmtest", ".JPG");
	sfm::FeatureMatcher a("C:\\williamnguatem\\fi\\sfm\\data",".JPG");
	a.initMatcher(MATCHER0_GPU);
	unsigned int numkypnts = a.matchtKeyPoints();
//	SiftGPU  *sift = new SiftGPU;
//	SiftMatchGPU *matcher = new SiftMatchGPU(4096);
//
//	std::vector<float > descriptors1(1), descriptors2(1);
//	std::vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);    
//	int num1 = 0, num2 = 0;
//
//	//process parameters
//	//The following parameters are default in V340
//	//-m,       up to 2 orientations for each feature (change to single orientation by using -m 1)
//	//-s        enable subpixel subscale (disable by using -s 0)
//
//
//	char * argv[] = {"-fo", "-1",  "-v", "1"};//
//	//-fo -1    staring from -1 octave 
//	//-v 1      only print out # feature and overall time
//	//-loweo    add a (.5, .5) offset
//	//-tc <num> set a soft limit to number of detected features
//
//	//NEW:  parameters for  GPU-selection
//	//1. CUDA.                   Use parameter "-cuda", "[device_id]"
//	//2. OpenGL.				 Use "-Display", "display_name" to select monitor/GPU (XLIB/GLUT)
//	//   		                 on windows the display name would be something like \\.\DISPLAY4
//
//	//////////////////////////////////////////////////////////////////////////////////////
//	//You use CUDA for nVidia graphic cards by specifying
//	//-cuda   : cuda implementation (fastest for smaller images)
//	//          CUDA-implementation allows you to create multiple instances for multiple threads
//	//          Checkout src\TestWin\MultiThreadSIFT
//	/////////////////////////////////////////////////////////////////////////////////////
//
//	//////////////////////////////////////////////////////////////////////////////////////
//	////////////////////////Two Important Parameters///////////////////////////
//	// First, texture reallocation happens when image size increases, and too many 
//	// reallocation may lead to allocatoin failure.  You should be careful when using 
//	// siftgpu on a set of images with VARYING imag sizes. It is recommended that you 
//	// preset the allocation size to the largest width and largest height by using function
//	// AllocationPyramid or prameter '-p' (e.g. "-p", "1024x768").
//
//	// Second, there is a parameter you may not be aware of: the allowed maximum working
//	// dimension. All the SIFT octaves that needs a larger texture size will be skipped.
//	// The default prameter is 2560 for the unpacked implementation and 3200 for the packed.
//	// Those two default parameter is tuned to for 768MB of graphic memory. You should adjust
//	// it for your own GPU memory. You can also use this to keep/skip the small featuers.
//	// To change this, call function SetMaxDimension or use parameter "-maxd".
//	//
//	// NEW: by default SiftGPU will try to fit the cap of GPU memory, and reduce the working 
//	// dimension so as to not allocate too much. This feature can be disabled by -nomc
//	//////////////////////////////////////////////////////////////////////////////////////
//
//
//	int argc = sizeof(argv)/sizeof(char*);
//	sift->ParseParam(argc, argv);
//
//	///////////////////////////////////////////////////////////////////////
//	//Only the following parameters can be changed after initialization (by calling ParseParam). 
//	//-dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
//	//to change other parameters at runtime, you need to first unload the dynamically loaded libaray
//	//reload the libarary, then create a new siftgpu instance
//
//
//	//Create a context for computation, and SiftGPU will be initialized automatically 
//	//The same context can be used by SiftMatchGPU
//	if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;
//
//	if(sift->RunSIFT("data/001.jpg"))
//	{
//		//Call SaveSIFT to save result to file, the format is the same as Lowe's
//		//sift->SaveSIFT("../data/800-1.sift"); //Note that saving ASCII format is slow
//
//		//get feature count
//		num1 = sift->GetFeatureNum();
//
//		//allocate memory
//		keys1.resize(num1);    descriptors1.resize(128*num1);
//
//		//reading back feature vectors is faster than writing files
//		//if you dont need keys or descriptors, just put NULLs here
//		sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
//		//this can be used to write your own sift file.            
//	}
//
//	//You can have at most one OpenGL-based SiftGPU (per process).
//	//Normally, you should just create one, and reuse on all images. 
//	if(sift->RunSIFT("data/002.jpg"))
//	{
//		num2 = sift->GetFeatureNum();
//		keys2.resize(num2);    descriptors2.resize(128*num2);
//		sift->GetFeatureVector(&keys2[0], &descriptors2[0]);
//	}
//
//	//Testing code to check how it works when image size varies
//	//sift->RunSIFT("../data/256.jpg");sift->SaveSIFT("../data/256.sift.1");
//	//sift->RunSIFT("../data/1024.jpg"); //this will result in pyramid reallocation
//	//sift->RunSIFT("../data/256.jpg"); sift->SaveSIFT("../data/256.sift.2");
//	//two sets of features for 256.jpg may have different order due to implementation
//
//	//*************************************************************************
//	/////compute descriptors for user-specified keypoints (with or without orientations)
//
//	//Method1, set new keypoints for the image you've just processed with siftgpu
//	//say vector<SiftGPU::SiftKeypoint> mykeys;
//	//sift->RunSIFT(mykeys.size(), &mykeys[0]); 
//	//sift->RunSIFT(num2, &keys2[0], 1);         sift->SaveSIFT("../data/640-1.sift.2");
//	//sift->RunSIFT(num2, &keys2[0], 0);        sift->SaveSIFT("../data/640-1.sift.3");
//
//	//Method2, set keypoints for the next coming image
//	//The difference of with method 1 is that method 1 skips gaussian filtering
//	//SiftGPU::SiftKeypoint mykeys[100];
//	//for(int i = 0; i < 100; ++i){
//	//    mykeys[i].s = 1.0f;mykeys[i].o = 0.0f;
//	//    mykeys[i].x = (i%10)*10.0f+50.0f;
//	//    mykeys[i].y = (i/10)*10.0f+50.0f;
//	//}
//	//sift->SetKeypointList(100, mykeys, 0);
//	//sift->RunSIFT("../data/800-1.jpg");                    sift->SaveSIFT("../data/800-1.sift.2");
//	//### for comparing with method1: 
//	//sift->RunSIFT("../data/800-1.jpg"); 
//	//sift->RunSIFT(100, mykeys, 0);                          sift->SaveSIFT("../data/800-1.sift.3");
//	//*********************************************************************************
//
//
//	//**********************GPU SIFT MATCHING*********************************
//	//**************************select shader language*************************
//	//SiftMatchGPU will use the same shader lanaguage as SiftGPU by default
//	//Before initialization, you can choose between glsl, and CUDA(if compiled). 
//	//matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA); // +i for the (i+1)-th device
//
//	//Verify current OpenGL Context and initialize the Matcher;
//	//If you don't have an OpenGL Context, call matcher->CreateContextGL instead;
//	matcher->VerifyContextGL(); //must call once
//
//	//Set descriptors to match, the first argument must be either 0 or 1
//	//if you want to use more than 4096 or less than 4096
//	//call matcher->SetMaxSift() to change the limit before calling setdescriptor
//	matcher->SetDescriptors(0, num1, &descriptors1[0]); //image 1
//	matcher->SetDescriptors(1, num2, &descriptors2[0]); //image 2
//
//	//match and get result.    
//	int (*match_buf)[2] = new int[num1][2];
//	//use the default thresholds. Check the declaration in SiftGPU.h
//	int num_match = matcher->GetSiftMatch(num1, match_buf);
//	std::cout<< num_match << " sift matches were found;\n";
//
//	//enumerate all the feature matches
//	for(int i  = 0; i < num_match; ++i)
//	{
//		//How to get the feature matches: 
//		SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];
//		SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];
//		//key1 in the first image matches with key2 in the second image
//	}
//
//	//*****************GPU Guided SIFT MATCHING***************
//	//example: define a homography, and use default threshold 32 to search in a 64x64 window
//	//float h[3][3] = {{0.8f, 0, 0}, {0, 0.8f, 0}, {0, 0, 1.0f}};
//	//matcher->SetFeatureLocation(0, &keys1[0]); //SetFeatureLocaiton after SetDescriptors
//	//matcher->SetFeatureLocation(1, &keys2[0]);
//	//num_match = matcher->GetGuidedSiftMatch(num1, match_buf, h, NULL);
//	//std::cout << num_match << " guided sift matches were found;\n";
//	//if you can want to use a Fundamental matrix, check the function definition
//
//	// clean up..
//	delete[] match_buf;
//#ifdef REMOTE_SIFTGPU
//	delete combo;
//#else
//	delete sift;
//	delete matcher;
//#endif
//
//#ifdef SIFTGPU_DLL_RUNTIME
//	FREE_MYLIB(hsiftgpu);
//#endif
	return 1;
}

//
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//#include <stdio.h>
//#include <string.h>
//#include <ctype.h>
//
//using namespace cv;
//using namespace std;
//
//void help()
//{
//	printf(
//		"\nDemonstrate the use of the HoG descriptor using\n"
//		"  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
//		"Usage:\n"
//		"./peopledetect (<image_filename> | <image_list>.txt)\n\n");
//}
//
//int main(int argc, char** argv)
//{
//	Mat img;
//	FILE* f = 0;
//	char _filename[1024];
//
//	//if( argc == 1 )
//	//{
//	//	printf("Usage: peopledetect (<image_filename> | <image_list>.txt)\n");
//	//	return 0;
//	//}
//	//img = imread(argv[1]);
//
//	//if( img.data )
//	//{
//	//	strcpy(_filename, argv[1]);
//	//}
//	//else
//	//{
//	//	f = fopen(argv[1], "rt");
//	//	if(!f)
//	//	{
//	//		fprintf( stderr, "ERROR: the specified file could not be loaded\n");
//	//		return -1;
//	//	}
//	//}
//
//
//
//	HOGDescriptor hog;
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//	namedWindow("people detector", 1);
//
//	for(;;)
//	{
//		/*char* filename = "C:\\williamnguatem\\fi\\sfm\\data\\32nd.jpg";*/ 
//			//char* filename = "C:\\williamnguatem\\fi\\sfm\\data\\39fb.jpg";
//		/*char* filename = "C:\\williamnguatem\\fi\\sfm\\data\\38fb.jpg";*/ 
//			char* filename = "C:\\williamnguatem\\fi\\sfm\\data\\harry_weber_photography_project_people_street_7.jpg";
//		if(f)
//		{
//			if(!fgets(filename, (int)sizeof(_filename)-2, f))
//				break;
//			//while(*filename && isspace(*filename))
//			//	++filename;
//			if(filename[0] == '#')
//				continue;
//			int l = (int)strlen(filename);
//			while(l > 0 && isspace(filename[l-1]))
//				--l;
//			filename[l] = '\0';
//			img = imread(filename);
//		}
//		printf("%s:\n", filename);
//		img = imread(filename);
//		if(!img.data)
//			continue;
//
//
//		fflush(stdout);
//		vector<Rect> found, found_filtered;
//		double t = (double)getTickCount();
//		// run the detector with default parameters. to get a higher hit-rate
//		// (and more false alarms, respectively), decrease the hitThreshold and
//		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
//		hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
//		t = (double)getTickCount() - t;
//		printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
//		size_t i, j;
//		for( i = 0; i < found.size(); i++ )
//		{
//			Rect r = found[i];
//			for( j = 0; j < found.size(); j++ )
//				if( j != i && (r & found[j]) == r)
//					break;
//			if( j == found.size() )
//				found_filtered.push_back(r);
//		}
//		for( i = 0; i < found_filtered.size(); i++ )
//		{
//			Rect r = found_filtered[i];
//			// the HOG detector returns slightly larger rectangles than the real objects.
//			// so we slightly shrink the rectangles to get a nicer output.
//			r.x += cvRound(r.width*0.1);
//			r.width = cvRound(r.width*0.8);
//			r.y += cvRound(r.height*0.07);
//			r.height = cvRound(r.height*0.8);
//			rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
//		}
//		imshow("people detector", img);
//		int c = waitKey(0) & 255;
//		if( c == 'q' || c == 'Q' || !f)
//			break;
//	}
//	if(f)
//		fclose(f);
//	return 0;
//}
