#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>


#include "vp_detection_cloud.h"
//#include "VanishingPointDetection/VPDetection.h"
//#include "ModelSegmentation/ModelSegmentation.h"
#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp> 
#include <boost/tuple/tuple_io.hpp> 

//#include "CuboidDetection/CuboidDetection.h" // Remove this and re-implement the ppplane segmentation


//void PlaneModel::setModelParameter(const SegModelParamsPtr &_modelParams)
//{
//	
//
//
//}

void fi::VPDetectionCloud::ComputeVPDetection()
{
	//Collect all params first!
	unsigned int nValidationRounds = m_ModelParams->fNumberOfValidationRounds;
	
	//Check if Down Sampling required using a Voxel grid filter
	// Create the filtering object: down sample the dataset using a leaf size of 1cm
	float aLeaf = m_ModelParams->fVoxGridSize(0);
	float bLeaf = m_ModelParams->fVoxGridSize(1);
	float cLeaf = m_ModelParams->fVoxGridSize(2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZ>);

	if (!((aLeaf == 0.0f) && (bLeaf == 0.0f) &&(cLeaf == 0.0f)) )
	{
		pcl::VoxelGrid<pcl::PointXYZ> vg;
		vg.setInputCloud (m_InputCloud);
		vg.setLeafSize (aLeaf, bLeaf, cLeaf); //fachwerkhaus
		//vg.setLeafSize (0.04f, 0.04f, 0.04f);  //Schloss Etlingen
		//vg.setLeafSize (0.1f, 0.1f, 0.1f); //unichurch
		vg.filter (*cloudFiltered);
		std::cout << "PointCloud before filtering has: " << m_InputCloud->points.size ()  << " data points." << std::endl; //*
		std::cout << "PointCloud after filtering has: " << cloudFiltered->points.size ()  << " data points." << std::endl; //*
	}
	else
	{
		//copying is not necessary here since the size of the cloud would not be reducing!
		cloudFiltered = m_InputCloud;
	}

	for (unsigned int i = 0; i < nValidationRounds; i++)
	{
		Eigen::VectorXf fRobustEstimatedVP;
		if (m_ModelParams->fRadiusSearched != 0.0f)
		{
			//ClusterRDirectionNeighbourhoods(cloudFiltered, fRobustEstimatedVP);
		}
		else
		{
			ClusterKDirectionPatches(cloudFiltered, fRobustEstimatedVP);
			m_EstimatedVPs.push_back(fRobustEstimatedVP);
		}
	}
			m_vgFilteredCloud = cloudFiltered;
}



void fi::VPDetectionCloud::ClusterKDirectionPatches(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_InCloud, Eigen::VectorXf &fRobustEstimatedVP)
{

	unsigned int fNumOfNormals = _InCloud->points.size() * m_ModelParams->fPercentageOfNormals / 100;
	unsigned int fNumOfCrossProducts = m_ModelParams->fNumOfCrossProducts;
	unsigned  int fNumberOfRansacIterations = m_ModelParams->fNumOfIterations;
	unsigned int m_k = m_ModelParams->m_k;

	if ( m_k == 0)
	{
		m_k = 5;
	}

	double fEPS = m_ModelParams->fEPSInDegrees;
	Eigen::Vector4f fEstimatedVP1;


	//// Create the normal estimation class, and pass the input dataset to it
	//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
	//normal_estimation.setInputCloud (cloud);

	//// Create an empty kdtree representation, and pass it to the normal estimation object.
	//// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	//normal_estimation.setSearchMethod (tree);

	//// Output datasets
	//pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

	//PtCloudDataPtr InPclDataPtr1(new PtCloudData());
	///*pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(pclData);*/
	//boost::shared_ptr<PtCloudData> InPclDataPtr(new PtCloudData(pclData));
	int nPointCandidates = _InCloud->points.size();

		// Consider using more precise timers such as gettimeofday on *nix or
		// GetTickCount/timeGetTime/QueryPerformanceCounter on Windows.
		boost::mt19937 randGen(std::time(0));

		// Now we set up a distribution. Boost provides a bunch of these as well.
		// This is the preferred way to generate numbers in a certain range.
		// initialize a uniform distribution between 0 and the max=nPointCandidates

		boost::uniform_int<> uInt8Dist(0, nPointCandidates);

		// Finally, declare a variate_generator which maps the random number
		// generator and the distribution together. This variate_generator
		// is usable like a function call.
		boost::variate_generator< boost::mt19937&, boost::uniform_int<> > 
			GetRand(randGen, uInt8Dist);

		//sample random points and compute their normals
		pcl::IndicesPtr indices(new  std::vector<int>());
		for (unsigned int i = 0; i < fNumOfNormals; i++)
		{
			indices->push_back(GetRand() % nPointCandidates);      
		}

		/*pcl::NormalEstimationOMP fd;*/ // 6-8 times faster ?
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud (_InCloud);
		ne.setSearchMethod (pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
		ne.setKSearch (m_k);
		//ne.setRadiusSearch (fRadius); //////toDo Set Cloud and radius hier correctly!
		//ne.setSearchSurface(source.surface);
		ne.setIndices(indices);

		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
		ne.compute (*normals);

		//typedef boost::scoped_ptr<Eigen::Vector4f> fXProds;

		std::vector<boost::shared_ptr<Eigen::Vector4f> > fVPHypothesis;
		std::vector<boost::shared_ptr<Eigen::Vector4f> > fVPHypothesisInliers;

		//sample random points and compute their normals
		for (unsigned int i = 0; i < fNumOfCrossProducts; i++)
		{
			unsigned int fIndexA = GetRand() % fNumOfNormals;
			unsigned int fIndexB = GetRand() % fNumOfNormals;

			pcl::Normal &fnormalA = normals->points.at(fIndexA);
			pcl::Normal &fnormalB = normals->points.at(fIndexB);

			Eigen::Vector4f fA(fnormalA.normal_x, fnormalA.normal_y, fnormalA.normal_z, 0); 
			Eigen::Vector4f fB(fnormalB.normal_x, fnormalB.normal_y, fnormalB.normal_z, 0);
			Eigen::Vector4f fC = fA.cross3(fB);

			boost::shared_ptr<Eigen::Vector4f> ftmpRes(new Eigen::Vector4f(fC));
			/*fXProds faVPHypothesis(new Eigen::Vector4f(fC));*/
			fVPHypothesis.push_back(ftmpRes);
			//tmpdistances = sqrt ((line_pt - fPlaneB->points[j].getVector4fMap ()).cross3 (line_dir).squaredNorm ());
		}

		//The Cross products are then the direction we are looking. Find the majority w.r.t a ref Direction!
		Eigen::Vector4f fRefDir (1.0, 0.0, 0.0, 0);

		//Validate the best using RANSAC w.r.t refDir
		unsigned int counterMax = 0;
		unsigned int iBest;
		for (unsigned int g = 0; g < fNumberOfRansacIterations; g++)
		{

			unsigned int gIndex = GetRand() % fNumOfCrossProducts;

			Eigen::Vector4f &fu1 = *fVPHypothesis.at(gIndex);
			double fAnglesInRadians = pcl::getAngle3D(fu1, fRefDir);
			double fAngleInDegrees = pcl::rad2deg(fAnglesInRadians);

			//score the selected hypothesis
			unsigned int sCounter = 0 ;
			for (unsigned int l = 0; l < fNumOfCrossProducts; l++)
			{
				//const float sAnglesInRadianstmp = Math3d::Angle(*((*crosProds)[l]), sRefDir);
				//const float sAnglesInDegreestmp = sAnglesInRadianstmp * 180.0f / CV_PI;

				Eigen::Vector4f &fu1tmp = *fVPHypothesis.at(l);
				double fAnglesInRadianstmp = pcl::getAngle3D(fu1tmp, fRefDir);
				double fAngleInDegreestmp = pcl::rad2deg(fAnglesInRadianstmp);

				if (fabsf(fAngleInDegreestmp - fAngleInDegrees) <= fEPS) //Tolerance of 2 Degress!!!
				{
					sCounter++;

				}
			}
			if (sCounter > counterMax)
			{	
				counterMax = sCounter;
				iBest = gIndex;
			}
		}

		//Insert the Estimated VP!
		//fEstimatedVP1 = *fVPHypothesis.at(l);
		/*delete m_pclDataPtr;*/
		//collect all the inliers and do WLeastSquaresFit
		Eigen::Vector4f &fBestModel = *fVPHypothesis.at(iBest);
		double fAnglesInRadiansBest = pcl::getAngle3D(fBestModel, fRefDir);
		double fAngleInDegreesBest = pcl::rad2deg(fAnglesInRadiansBest);

		pcl::PointCloud<pcl::PointXYZ>::Ptr fVPCandidates(new pcl::PointCloud<pcl::PointXYZ>);
		fVPCandidates->width = counterMax;
		fVPCandidates->height = 1;
		fVPCandidates->resize(fVPCandidates->width * fVPCandidates->height);
		unsigned int ii = 0;

		for (unsigned int l = 0; l < fNumOfCrossProducts; l++)
		{
			Eigen::Vector4f &fu1tmp = *fVPHypothesis.at(l);
			double fAnglesInRadianstmp = pcl::getAngle3D(fu1tmp, fRefDir);
			double fAngleInDegreestmp = pcl::rad2deg(fAnglesInRadianstmp);

			if (fabsf(fAngleInDegreestmp - fAngleInDegreesBest) <= fEPS) //Tolerance of 2 Degrees!!!
			{
				fVPCandidates->points[ii].x = fu1tmp(0);
				fVPCandidates->points[ii].y = fu1tmp(1);
				fVPCandidates->points[ii].z = fu1tmp(2);

				/*boost::scoped_ptr<Eigen::Vector4f> ftmpInliers(new Eigen::Vector4f(fu1tmp));
				fVPHypothesisInliers.push_back(ftmpInliers);*/

				if (ii > counterMax)
				{
					std::cout<<" Something went really wrong!..."<<std::endl;
					return;
				}
				ii++;
			}
		}

		//Robust fitting the inliers using WleastSquears fit
		//Eigen::VectorXf optimized_vp_coefficients;
		LSLineFitting(fVPCandidates, fRobustEstimatedVP);
	   //collect data for 
}



//ClusterRDirectionNeighbourhoods(cloudFiltered, fRobustEstimatedVP);





void fi::VPDetectionCloud::LSLineFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &fInClouds, Eigen::VectorXf &optimized_coefficients)
{
	// Needs a valid set of model coefficients
	//if (!isModelValid (model_coefficients))
	//{
	//	optimized_coefficients = model_coefficients;
	//	return;
	//}

	//// Need at least 2 points to estimate a line
	//if (inliers.size () <= 2)
	//{
	//	PCL_ERROR ("[pcl::SampleConsensusModelLine::optimizeModelCoefficients] Not enough inliers found to support a model (%lu)! Returning the same coefficients.\n", (unsigned long)inliers.size ());
	//	optimized_coefficients = model_coefficients;
	//	return;
	//}

	//int nMatchCandidates = fInClouds->points.size();

	//const std::vector<float> fTLineCoefs;
	//pcl::PointXYZ fPntC;
	//fPntC.x = 0.0f;
	//fPntC.y = 0.0f;
	//fPntC.z = 0.0f;


	//model_coefficients.resize (6);
	//model_coefficients[0] = input_->points[samples[0]].x;
	//model_coefficients[1] = input_->points[samples[0]].y;
	//model_coefficients[2] = input_->points[samples[0]].z;

	//model_coefficients[3] = input_->points[samples[1]].x - model_coefficients[0];
	//model_coefficients[4] = input_->points[samples[1]].y - model_coefficients[1];
	//model_coefficients[5] = input_->points[samples[1]].z - model_coefficients[2];

	//model_coefficients.template tail<3> ().normalize ();






	//for(int i = 0; i < nMatchCandidates; i++)
	//{
	//	//identify 2 different points
	//	const int nFirstIndex = rand() % nMatchCandidates;
	//	int nTempIndex;
	//	
	//	do { nTempIndex = rand() % nMatchCandidates; } while (nTempIndex == nFirstIndex);
	//	
	//	pcl::PointXYZ &fPntA = fInClouds->points[nFirstIndex];
	//	pcl::PointXYZ &fPntB = fInClouds->points[nTempIndex];
	//			
	//	fPntC.x += fPntA.x - fPntB.x;
	//	fPntC.y += fPntA.y - fPntB.y;
	//	fPntC.z += fPntA.z - fPntB.z;
	//}

	//fPntC.x = fPntC.x/nMatchCandidates;
	//fPntC.y = fPntC.y/nMatchCandidates;
	//fPntC.z = fPntC.z/nMatchCandidates;

	//Eigen::VectorXf unnormalizedCoeffs(6);
	//unnormalizedCoeffs[3] = fPntC.x/nMatchCandidates; 
	//unnormalizedCoeffs[4] = fPntC.y/nMatchCandidates;
	//unnormalizedCoeffs[5] = fPntC.z/nMatchCandidates;
	//unnormalizedCoeffs.template tail<3> ().normalize ();

	//std::vector<float> fTmpCoefs;
	//fTmpCoefs.push_back(fPntC.x);
	//fTmpCoefs.push_back(fPntC.y);
	//fTmpCoefs.push_back(fPntC.z);
	//fTmpCoefs.push_back(0.0f);

	//GetPlaneCoefficients1(fInClouds, fTmpCoefs);

	//Eigen::VectorXf model_coefficients;
	//model_coefficients.resize (6);
	//model_coefficients[0] = fTmpCoefs.at(0);
	//model_coefficients[1] = fTmpCoefs.at(1);
	//model_coefficients[2] = fTmpCoefs.at(2);

	//model_coefficients[3] = fInClouds->points[0].x - model_coefficients[0];
	//model_coefficients[4] = fInClouds->points[0].y - model_coefficients[1];
	//model_coefficients[5] = fInClouds->points[0].z - model_coefficients[2];

	optimized_coefficients.resize (6);

	// Compute the 3x3 covariance matrix
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid (*fInClouds, centroid);


	Eigen::Matrix3f covariance_matrix;
	//pcl::computeCovarianceMatrix(*input_, inliers, centroid, covariance_matrix);
	pcl::computeCovarianceMatrix(*fInClouds, centroid, covariance_matrix);
	optimized_coefficients[0] = centroid[0];
	optimized_coefficients[1] = centroid[1];
	optimized_coefficients[2] = centroid[2];

	// Extract the eigenvalues and eigenvectors
	EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
	EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

	optimized_coefficients.tail<3> () = eigen_vectors.col (2).normalized ();
}
















