#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>


#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "feauture_matcher.h"
#include <Eigen/SVD>





sfm::FeatureMatcher::FeatureMatcher()
{

}

sfm::FeatureMatcher::FeatureMatcher(const std::string &input_image_folder, const std::string &image_extension)
	:m_image_data_dir(input_image_folder), m_image_extension(image_extension)
{

}


bool sfm::FeatureMatcher::initMatcher(const MatcherType &matching_type)
{
	//check system properties to meet the matching type of choice ??
	m_matching_type = matching_type;

	return getImageFileNames(m_image_filenames);
}



bool sfm::FeatureMatcher::getImageFileNames(std::vector<std::string> &images_filenames)
{
	if ( !boost::filesystem::exists( m_image_data_dir ) )
	{
		std::cout<<"directory not found!"<<std::endl;
		return false;
	}

	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	//boost::regex mapping_filter("3D2Dtable");//used to matched the 3D2Dtable	

	for ( boost::filesystem::directory_iterator itr( m_image_data_dir );itr != end_itr; ++itr )
	{
		if ( !boost::filesystem::is_regular_file( itr->status() ) )
			continue; // Skip if not a file 
		if (! (itr->path().extension().string() == m_image_extension)) 
			continue; // Skip if no match
		images_filenames.push_back(itr->path().string());
	}

	if (images_filenames.size() < 2)
	{
		std::cout<<"At least not 2 images are needed for matching !"<<std::endl;
		return false;
	}

	return true;
}

sfm::FeatureMatcher::~FeatureMatcher()
{
}


unsigned int sfm::FeatureMatcher::matchtKeyPoints()
{

	switch(m_matching_type)
	{
	case MATCHER0_CPU: 
		std::cout << "Variable hat den Wert 0." << std::endl;
		break;

	case MATCHER1_CPU: 
		std::cout << "Variable hat den Wert 1." << std::endl;
		break;

	case MATCHER2_CPU: 
		std::cout << "Variable hat den Wert CPU2" << std::endl;
		break;

	case MATCHER3_CPU: 
		std::cout << "Variable hat den Wert 3" << std::endl;
		break;

	case MATCHER4_CPU:
		std::cout << "Variable hat den Wert 10." << std::endl;
		break;

	case MATCHER0_GPU:
		{
			gpumatcher0();
		}
		break;

	default: std::cout << "default cpu lasy matcher selected." << std::endl;
	}
	return 2;
}

int sfm::FeatureMatcher::gpumatcher0()
{
	unsigned int num_images = m_image_filenames.size();
	std::cout<<num_images<<" of images to process"<<std::endl;

	SiftGPU  *sift = new SiftGPU;
			int max_matching_num = 2048;
			//int max_matching_num = 4096;
	SiftMatchGPU *matcher = new SiftMatchGPU(max_matching_num);


	/*int match_buf[4096][2];*/
	//process parameters
	//The following parameters are default in V340
	//-m,       up to 2 orientations for each feature (change to single orientation by using -m 1)
	//-s        enable subpixel subscale (disable by using -s 0)


	char * argv[] = {"-fo", "-1",  "-v", "1"};//
	//-fo -1    staring from -1 octave 
	//-v 1      only print out # feature and overall time
	//-loweo    add a (.5, .5) offset
	//-tc <num> set a soft limit to number of detected features

	//NEW:  parameters for  GPU-selection
	//1. CUDA.                   Use parameter "-cuda", "[device_id]"
	//2. OpenGL.				 Use "-Display", "display_name" to select monitor/GPU (XLIB/GLUT)
	//   		                 on windows the display name would be something like \\.\DISPLAY4

	//////////////////////////////////////////////////////////////////////////////////////
	//You use CUDA for nVidia graphic cards by specifying
	//-cuda   : cuda implementation (fastest for smaller images)
	//          CUDA-implementation allows you to create multiple instances for multiple threads
	//          Checkout src\TestWin\MultiThreadSIFT
	/////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////
	////////////////////////Two Important Parameters///////////////////////////
	// First, texture reallocation happens when image size increases, and too many 
	// reallocation may lead to allocatoin failure.  You should be careful when using 
	// siftgpu on a set of images with VARYING imag sizes. It is recommended that you 
	// preset the allocation size to the largest width and largest height by using function
	// AllocationPyramid or prameter '-p' (e.g. "-p", "1024x768").

	// Second, there is a parameter you may not be aware of: the allowed maximum working
	// dimension. All the SIFT octaves that needs a larger texture size will be skipped.
	// The default prameter is 2560 for the unpacked implementation and 3200 for the packed.
	// Those two default parameter is tuned to for 768MB of graphic memory. You should adjust
	// it for your own GPU memory. You can also use this to keep/skip the small featuers.
	// To change this, call function SetMaxDimension or use parameter "-maxd".
	//
	// NEW: by default SiftGPU will try to fit the cap of GPU memory, and reduce the working 
	// dimension so as to not allocate too much. This feature can be disabled by -nomc
	//////////////////////////////////////////////////////////////////////////////////////


	int argc = sizeof(argv)/sizeof(char*);
	sift->ParseParam(argc, argv);

	///////////////////////////////////////////////////////////////////////
	//Only the following parameters can be changed after initialization (by calling ParseParam). 
	//-dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
	//to change other parameters at runtime, you need to first unload the dynamically loaded libaray
	//reload the libarary, then create a new siftgpu instance


	//Create a context for computation, and SiftGPU will be initialized automatically 
	//The same context can be used by SiftMatchGPU
	if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;


	std::vector<SiftKeysPtr> sift_keys(num_images);
	std::vector<SiftDescriptorsPtr> sift_descriptors(num_images);

	for ( unsigned int i = 0; i < num_images; i++)
	{
		std::vector<float > descriptors1(1);
		std::vector<SiftGPU::SiftKeypoint> keys1(1);
		int num1 = 0;
		if(sift->RunSIFT(m_image_filenames[i].c_str()))
		{
			//Call SaveSIFT to save result to file, the format is the same as Lowe's
			//sift->SaveSIFT("../data/800-1.sift"); //Note that saving ASCII format is slow

			//get feature count
			num1 = sift->GetFeatureNum();

			//allocate memory
			keys1.resize(num1);    descriptors1.resize(128*num1);

			//reading back feature vectors is faster than writing files
			//if you dont need keys or descriptors, just put NULLs here
			sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
			//this can be used to write your own sift file.     


			SiftKeysPtr keys(new SiftKeys(keys1));
			sift_keys[i] = keys;
			SiftDescriptorsPtr descri(new SiftDescriptors(descriptors1));
			sift_descriptors[i] = descri;

		}else
		{
			//all is null
		}

	}


	//**********************GPU SIFT MATCHING*********************************
	//**************************select shader language*************************
	//SiftMatchGPU will use the same shader lanaguage as SiftGPU by default
	//Before initialization, you can choose between glsl, and CUDA(if compiled). 
	//matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA); // +i for the (i+1)-th device

	//Verify current OpenGL Context and initialize the Matcher;
	//If you don't have an OpenGL Context, call matcher->CreateContextGL instead;
	matcher->VerifyContextGL(); //must call once


	for (unsigned int i = 0; i < num_images; i++)
	{
		std::vector<float > descriptors_a = *sift_descriptors.at(i);   //ToDo: Make faster
		std::vector<SiftGPU::SiftKeypoint> keys_a = *sift_keys.at(i);
		int num1 = descriptors_a.size();

		for (unsigned int j = i+1; j < num_images; j++)
		{
			std::vector<float > descriptors_b = *sift_descriptors.at(j);   //ToDo: Make faster
			std::vector<SiftGPU::SiftKeypoint> keys_b = *sift_keys.at(j);
			int num2 = descriptors_b.size();

			//Testing code to check how it works when image size varies
			//sift->RunSIFT("../data/256.jpg");sift->SaveSIFT("../data/256.sift.1");
			//sift->RunSIFT("../data/1024.jpg"); //this will result in pyramid reallocation
			//sift->RunSIFT("../data/256.jpg"); sift->SaveSIFT("../data/256.sift.2");
			//two sets of features for 256.jpg may have different order due to implementation

			//*************************************************************************
			/////compute descriptors for user-specified keypoints (with or without orientations)

			//Method1, set new keypoints for the image you've just processed with siftgpu
			//say vector<SiftGPU::SiftKeypoint> mykeys;
			//sift->RunSIFT(mykeys.size(), &mykeys[0]); 
			//sift->RunSIFT(num2, &keys2[0], 1);         sift->SaveSIFT("../data/640-1.sift.2");
			//sift->RunSIFT(num2, &keys2[0], 0);        sift->SaveSIFT("../data/640-1.sift.3");

			//Method2, set keypoints for the next coming image
			//The difference of with method 1 is that method 1 skips gaussian filtering
			//SiftGPU::SiftKeypoint mykeys[100];
			//for(int i = 0; i < 100; ++i){
			//    mykeys[i].s = 1.0f;mykeys[i].o = 0.0f;
			//    mykeys[i].x = (i%10)*10.0f+50.0f;
			//    mykeys[i].y = (i/10)*10.0f+50.0f;
			//}
			//sift->SetKeypointList(100, mykeys, 0);
			//sift->RunSIFT("../data/800-1.jpg");                    sift->SaveSIFT("../data/800-1.sift.2");
			//### for comparing with method1: 
			//sift->RunSIFT("../data/800-1.jpg"); 
			//sift->RunSIFT(100, mykeys, 0);                          sift->SaveSIFT("../data/800-1.sift.3");
			//*********************************************************************************

			//Set descriptors to match, the first argument must be either 0 or 1
			//if you want to use more than 4096 or less than 4096
			//call matcher->SetMaxSift() to change the limit before calling setdescriptor

			if (num1 > num2)
				max_matching_num = 2*num1;
			else
				max_matching_num = 2*num2;


			//matcher->SetMaxSift(20000);

			matcher->SetDescriptors(0, num1, &descriptors_a[0]); //image 1
			matcher->SetDescriptors(1, num2, &descriptors_b[0]); //image 2

			//match and get result.    
			int (*match_buf)[2] = new int[max_matching_num][2];
			//use the default thresholds. Check the declaration in SiftGPU.h
			int num_match = matcher->GetSiftMatch(max_matching_num, match_buf);
			std::cout<< num_match << " sift matches were found;\n";

			//throw away very few matches
			if (num_match < MIN_NUM_MATCHES)
				continue;

			//enumerate all the feature matches
			std::vector<cv::Point2f> points1(num_match);
			std::vector<cv::Point2f> points2(num_match);
			
			for(int h  = 0; h < num_match; ++h)
			{
				//How to get the feature matches: 
				//key1 in the first image matches with key2 in the second image
				SiftGPU::SiftKeypoint & key1 = keys_a[match_buf[h][0]];
				SiftGPU::SiftKeypoint & key2 = keys_b[match_buf[h][1]];

				points1[h].x = key1.x;
				points1[h].y = key1.y;

				points2[h].x = key2.x;
				points2[h].y = key2.y;
			}

			//select the type of fm_matrix computation scheme

			//compute the fundamental matrix using ransac
			Eigen::Matrix3d fundamental_matrix;
			computeFundamentalMatrixCPU0(points1, points2, fundamental_matrix);


			delete [] match_buf;
		}
	}

	std::cout<<"Finnished Matching all image pairs"<<std::endl;
}


//computes the second camera's matrix assuming the first camera P1 is identity matrix
void robustPose2(const Eigen::Matrix3d &essential_matrix, Eigen::Matrix4d &pose_matrix2)
{

	/*Eigen::JacobiSVD<Eigen::JacobianType> svd;*/
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(essential_matrix);	

	// Only specify ComputeFullU/V if you need them
	// (default is to only compute singular values)
	/*svd.compute(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);*/
	svd.compute(essential_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV );
	
	std::cout<<svd.singularValues()<<std::endl;
	Eigen::Vector3d sv = svd.singularValues();
	Eigen::Matrix3d um = svd.matrixU();
	Eigen::Matrix3d vm = svd.matrixV();

	std::cout << "Orig matrix:\n" << essential_matrix << std::endl;
	std::cout << "Singular values:\n" << sv << std::endl;
	std::cout << "U matrix:\n" << um << std::endl;
	std::cout << "V matrix:\n" << vm << std::endl;

	//double ddd = vm.dot(um);

	//make sure that the matrix is rank 2
	//std::cout<<"dot products: "<< um.dot(vm)<<std::endl;

	//std::cout<<"Remultiply USV: "<<um *sv*vm<<std::endl;


	//// Answer
	//// Set to zero to solve a homogeneous equation system
	////VectorXf b = VectorXf::Zero(26);
	////std::cout << "Vector b is:" << std::endl << std::endl << b << std::endl << std::endl;

	//Eigen::SVD<Eigen::Mat4d> svdOfA(essential_matrix);
	//svdOfA.solve(b, &x);

	//const Eigen::MatrixXf U = svdOfA.matrixU();
	//const Eigen::MatrixXf V = svdOfA.matrixV();
	//const Eigen::VectorXf S = svdOfA.singularValues();

	//std::cout << "Matrix U is:" << std::endl << std::endl << U << std::endl << std::endl;
	//std::cout << "Matrix V is:" << std::endl << std::endl << V << std::endl << std::endl;
	//std::cout << "Matrix S is:" << std::endl << std::endl << S << std::endl << std::endl;
	//std::cout << "Vector x is:" << std::endl << std::endl << x << std::endl << std::endl;
	//std::cout << "Vector b is:" << std::endl << std::endl << b << std::endl << std::endl;


	////Eigen::Matrix3f m3;
	////m3  svd(m3);

	////const Eigen::Matrix3f U = svd.matrixU();
	////const Eigen::Matrix3f V = svd.matrixV();
	////const Eigen::VectorXf S = svd.singularValues();
	//


	//
	//	//make sure essential_matrix is rank 2
	//const Eigen::Matrix4d mat_u = essential_matrix.matrixFunction()


	//	// The following code solves Ax = b, where b is 0, thus making this a homoegeneous linear equation system

	//	// *** LEFT SIDE ***
	//	// Linear equation system is expressed in the form of a matrix of coefficents
	//	// rows-by-cols, m-by-n, 26-by-12, i.e. overdetermined linear equation system
	//	MatrixXf A = MatrixXf::Random(26,12);
	//std::cout << "Matrix A is:" << std::endl << std::endl << A << std::endl << std::endl;

	//// Unknowns
	//VectorXf x(12);
	//std::cout << "Vector x is:" << std::endl << std::endl << x << std::endl << std::endl;

	//// *** RIGHT SIDE ***
	//// Answer
	//// Set to zero to solve a homogeneous equation system
	//VectorXf b = VectorXf::Zero(26);
	//std::cout << "Vector b is:" << std::endl << std::endl << b << std::endl << std::endl;

	//SVD<MatrixXf> svdOfA(A);
	//svdOfA.solve(b, &x);

	//const Eigen::MatrixXf U = svdOfA.matrixU();
	//const Eigen::MatrixXf V = svdOfA.matrixV();
	//const Eigen::VectorXf S = svdOfA.singularValues();

	//std::cout << "Matrix U is:" << std::endl << std::endl << U << std::endl << std::endl;
	//std::cout << "Matrix V is:" << std::endl << std::endl << V << std::endl << std::endl;
	//std::cout << "Matrix S is:" << std::endl << std::endl << S << std::endl << std::endl;
	//std::cout << "Vector x is:" << std::endl << std::endl << x << std::endl << std::endl;
	//std::cout << "Vector b is:" << std::endl << std::endl << b << std::endl << std::endl;

	

}



void sfm::FeatureMatcher::computeFundamentalMatrixCPU0(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, Eigen::Matrix3d &fundamental_matrix)
{
	cv::Mat f_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99);
	std::cout<<f_matrix<<std::endl;

	//m.at<Vec2f>( row , col )[0]

	float t00 = f_matrix.at<double>(0,0);
	float t01 = f_matrix.at<double>(0,1);
	float t02 = f_matrix.at<double>(0,2);
	fundamental_matrix<< f_matrix.at<double>(0,0), f_matrix.at<double>(0,1), f_matrix.at<double>(0,2),
		f_matrix.at<double>(1,0), f_matrix.at<double>(1,1), f_matrix.at<double>(1,2),
		f_matrix.at<double>(2,0), f_matrix.at<double>(2,1), f_matrix.at<double>(2,2);
	std::cout<<fundamental_matrix;

	Eigen::Matrix4d pose_matrix2;

	robustPose2(fundamental_matrix, pose_matrix2);
}

//
//void sfm::FeatureMatcher::robustTriangulationCPU0(const Eigen::Matrix3d &fundamental_matrix, const <Eigen::Matrix3d> &cam_intrinsics, const std::vector<Eigen::Vector3d> &points_3d, Eigen::Matrix3d &fundamental_matrix)
//{
//
//
//
//}
//


