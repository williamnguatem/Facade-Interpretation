#include <pcl/common/centroid.h>
#include <pcl/common/angles.h>
#include "pcl/common/eigen.h"
#include "vp_detection.h"


fi::VPDetection::~VPDetection(){}

void fi::VPDetection::_setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_point_cloud)
{
	m_InputCloud = _point_cloud;
}


void fi::VPDetection::_setInputImg(const cv::Mat &_inputImg)
{
	m_InputImg = _inputImg;
}


void fi::VPDetection::_setModelParameter(const VPModelParamsPtr &_modelParams)
{
	m_ModelParams = _modelParams;
}


//void VPDetection::setMaxNumOfModelsToSegment(const unsigned  int &_maxNumModels )
//{
//	/*m_MaxNumModels = _maxNumModels;*/
//	m_ModelParams->fMaxNumModels = _maxNumModels;
//}

//
//void VPDetection::setMinModelInliers(const unsigned  int &_minModelInliers )
//{
//	/*m_MinModelInliers = _minModelInliers;*/
//	m_ModelParams->fMinNumInliers = _minModelInliers;
//}
//
//unsigned int VPDetection::getMaxNumOfModelsToSegment()
//{
//	return m_ModelParams->fMaxNumModels;
//}
//
//
//unsigned int VPDetection::getMinModelInliers()
//{
//	return m_ModelParams->fMinNumInliers;
//}


void fi::VPDetection::_validateResults(bool fValRes)
{		
	m_ValidateResults = fValRes;
}

void fi::VPDetection::_getvoxelgridedFilteredCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &vgFilteredCloud)
{
	vgFilteredCloud = m_vgFilteredCloud;
}


//bool VPDetection::getSegmentAllModels()
//{
//	if (m_ModelParams->fMaxNumModels > 1)
//	{
//		return true;
//	}
//}


void fi::VPDetection::_extractVPs(Eigen::VectorXf &outSVPs)
{	
	pcl::PointCloud<pcl::PointXYZ>::Ptr fCVPs(new pcl::PointCloud<pcl::PointXYZ>);
	if (m_ValidateResults == false)
	{
		//unsigned int fIndx = rand() % outSVPs.size()/RAND_MAX;
		outSVPs = m_EstimatedVPs.at(0);
	}
	else
	{
		//Do a robust fitting here! ToDo
		outSVPs = m_EstimatedVPs.at(0);
	//_robustFittingBestHypothesis(fCVPs, outSVPs);
	}

	////Overwrite the centroid int order to avoid transformations
	//outSVPs[3] = outSVPs[0];
	//outSVPs[4] = outSVPs[1];
	//outSVPs[5] = outSVPs[2];


	Eigen::Vector4f fCentroid;
	pcl::compute3DCentroid(*m_InputCloud,fCentroid);
	outSVPs[0] = fCentroid[0];
	outSVPs[1] = fCentroid[1];
	outSVPs[2] = fCentroid[2];
	//outSVPs[3] = fCVPs->points[4].x;
	//outSVPs[4] = fCVPs->points[4].y;
	//outSVPs[5] = fCVPs->points[4].z;
}


void fi::VPDetection::_robustFittingBestHypothesis(pcl::PointCloud<pcl::PointXYZ>::Ptr &fVPs, Eigen::VectorXf &optimized_coefficients)
{
	//The Best hypothesis are used to find the majority w.r.t a ref Direction!
		Eigen::Vector4f fRefDir (1.0, 0.0, 0.0, 0);
		unsigned int nValRounds = m_EstimatedVPs.size();
			double fEPS = 2;

		unsigned int ntrials = nValRounds/2;
		
		
		//Validate the best using RANSAC w.r.t refDir
		unsigned int counterMax = 0;
		unsigned int iBest;
		for (unsigned int g = 0; g < ntrials; g++)
		{
			unsigned int gIndex = rand() % nValRounds;
			Eigen::VectorXf &fSelectHypo = m_EstimatedVPs.at(gIndex);
			Eigen::Vector4f fu1(fSelectHypo[3], fSelectHypo[4], fSelectHypo[5], 0); 
			double fAnglesInRadians = pcl::getAngle3D(fu1, fRefDir);
			double fAngleInDegrees = pcl::rad2deg(fAnglesInRadians);

			//score the selected hypothesis
			unsigned int sCounter = 0 ;
			for (unsigned int l = 0; l < nValRounds; l++)
			{
				Eigen::VectorXf &ftmpHypo = m_EstimatedVPs.at(l);
				Eigen::Vector4f fu1tmp(ftmpHypo[3], ftmpHypo[4], ftmpHypo[5], 0);
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
				if(counterMax == nValRounds)
				break; //needless to continue
			}
		}
			
		Eigen::VectorXf &fBestHypo = m_EstimatedVPs.at(iBest);
		Eigen::Vector4f fBestModel(fBestHypo[3], fBestHypo[4], fBestHypo[5], 0);
		double fAnglesInRadiansBest = pcl::getAngle3D(fBestModel, fRefDir);
		double fAngleInDegreesBest = pcl::rad2deg(fAnglesInRadiansBest);

		pcl::PointCloud<pcl::PointXYZ>::Ptr fVPCandidates(new pcl::PointCloud<pcl::PointXYZ>);
		fVPCandidates->width = counterMax;
		fVPCandidates->height = 1;
		fVPCandidates->resize(fVPCandidates->width * fVPCandidates->height);
		unsigned int ii = 0;

		for (unsigned int l = 0; l < nValRounds; l++)
		{
			Eigen::VectorXf &fSelect = m_EstimatedVPs.at(l);
			Eigen::Vector4f fu1tmp(fSelect[3], fSelect[4], fSelect[5], 0);
			double fAnglesInRadianstmp = pcl::getAngle3D(fu1tmp, fRefDir);
			double fAngleInDegreestmp = pcl::rad2deg(fAnglesInRadianstmp);

			if (fabsf(fAngleInDegreestmp - fAngleInDegreesBest) <= fEPS) //Tolerance of 2 Degrees!!!
			{
				fVPCandidates->points[ii].x = fSelect(3);
				fVPCandidates->points[ii].y = fSelect(4);
				fVPCandidates->points[ii].z = fSelect(5);
				//std::cout<<fSelect<<std::endl;
				//std::cout<<fVPCandidates->points[ii].x<<fVPCandidates->points[ii].y<<fVPCandidates->points[ii].z<<std::endl;

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
		//Final Robust fitting
		_OLSFitting(fVPCandidates, optimized_coefficients );
		fVPs = fVPCandidates;
}


void fi::VPDetection::_OLSFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &fInClouds, Eigen::VectorXf &optimized_coefficients1)
{
	for (unsigned int ii = 0; ii < fInClouds->points.size(); ii++)
	{
	std::cout<<fInClouds->points[ii].x<<fInClouds->points[ii].y<<fInClouds->points[ii].z<<std::endl;
	}

	optimized_coefficients1.resize (6);

	// Compute the 3x3 covariance matrix
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid (*fInClouds, centroid);
	Eigen::Matrix3f covariance_matrix;
	//Eigen::Matrix3d covariance_matrix;

	//pcl::computeCovarianceMatrix(*input_, inliers, centroid, covariance_matrix);
	pcl::computeCovarianceMatrix(*fInClouds, centroid, covariance_matrix);
	std::cout<<covariance_matrix<<std::endl;
	optimized_coefficients1[0] = centroid[0];
	optimized_coefficients1[1] = centroid[1];
	optimized_coefficients1[2] = centroid[2];

	//// Extract the eigenvalues and eigenvectors
	//EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
	//EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
	Eigen::Vector3f eigen_values;
	Eigen::Matrix3f eigen_vectors;

	//Eigen::Vector3d eigen_values;
	//Eigen::Matrix3d eigen_vectors;
	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

optimized_coefficients1.tail<3> () = eigen_vectors.col (2).normalized ();
std::cout<<optimized_coefficients1<<std::endl;







	////// Needs a valid set of model coefficients
	////if (model_coefficients.size () != 4)
	////{
	////	PCL_ERROR ("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] Invalid number of model coefficients given (%lu)!\n", (unsigned long)model_coefficients.size ());
	////	optimized_coefficients = model_coefficients;
	////	return;
	////}

	////// Need at least 3 points to estimate a plane
	////if (inliers.size () < 4)
	////{
	////	PCL_ERROR ("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] Not enough inliers found to support a model (%lu)! Returning the same coefficients.\n", (unsigned long)inliers.size ());
	////	optimized_coefficients = model_coefficients;
	////	return;
	////}

	//if(fInClouds->points.size() < 4)
	//{
	//	std::cout<<"Need at least 3 points to estimate a plane"<<std::endl;
	//	return;
	//}

	//Eigen::Vector4f plane_parameters;

	//// Use ordinary Least-Squares to fit the plane through all the given sample points and find out its coefficients
	//EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
	//Eigen::Vector4f xyz_centroid;

	//// Estimate the XYZ centroid
	//pcl::compute3DCentroid (*fInClouds, xyz_centroid);
	//xyz_centroid[3] = 0;

	//// Compute the 3x3 covariance matrix
	//pcl::computeCovarianceMatrix (*fInClouds, xyz_centroid, covariance_matrix);

	//// Compute the model coefficients
	//EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
	//EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
	//pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);

	//// Hessian form (D = nc . p_plane (centroid here) + p)
	//optimized_coefficients.resize (4);
	//optimized_coefficients[0] = eigen_vectors (0, 0);
	//optimized_coefficients[1] = eigen_vectors (1, 0);
	//optimized_coefficients[2] = eigen_vectors (2, 0);
	//optimized_coefficients[3] = 0;
	//optimized_coefficients[3] = -1 * optimized_coefficients.dot (xyz_centroid);
}

