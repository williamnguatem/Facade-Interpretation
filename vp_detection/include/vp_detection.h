/******************************************************************************
\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
\file		VPDetection.h
\author		William Nguatem
\note		Copyright (C) 
\note		Bundeswehr University Munich
\note		Institute of Applied Computer Science
\note		Chair of Photogrammetry and Remote Sensing
\note		Neubiberg, Germany
\since		2012/02/07 
******************************************************************************/

//#pragma once;

#ifndef _VP_Detection_H_
#define _VP_Detection_H_
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


/*!
\brief 		Base class for all vanishing point detection routines
\ingroup 	VPDetection
*/

namespace fi
{	
	struct VPModelParam 
	{
		/*unsigned int fNumOfNormals;*/
		unsigned int fPercentageOfNormals;
		unsigned int fNumOfCrossProducts;
		unsigned int fNumberOfValidationRounds;
		unsigned int fNumOfIterations;
		unsigned int m_k; //for kdtree search
		double fRadiusSearched ;
		double fEPSInDegrees;
		Eigen::Vector4f fVoxGridSize;//Size of the sub sampling filters leafs
		bool paramValidateResults;
		bool fMulticoreSoppurt;
		bool fGPUSupport;
	};

	typedef VPModelParam VPModelParams;

	typedef boost::shared_ptr<VPModelParams> VPModelParamsPtr;

	class VPDetection
	{
	public:

		virtual ~VPDetection();

		/** \brief Set the name of the input cloud
		* \param[in] Input Cloud
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		void _setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_point_cloud);

		/** \brief Set the name of the input image 
		* \param[in] Input image
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		void _setInputImg(const cv::Mat &_inputImg); //change this and include opencv image and add the correct headers

		/** \brief Set the desired model parameters
		* \param[in] Parameters defining the model
		* \param[out] :ToDo: 
		* \return void 
		*/
		void _setModelParameter(const VPModelParamsPtr &_modelParams); 

		/** \brief Decide if we should segment all models
		* \param[in] true = segment all models else chose only the best
		* \param[out] :ToDo: 
		* \return void 
		*/
		//void setSegmentAllModels(const bool &_segmentAll); 
		//bool getSegmentAllModels(); 

		///** \brief Additional criteria if many models are to be segmented
		//  * \param[in] The maximum number of models to be segmented from the cloud
		//  * \param[out] :ToDo: 
		//  * \return void 
		//  */
		//void setMaxNumOfModelsToSegment(const unsigned  int &_maxNumModels = 1);  //Default always the best
		//unsigned int getMaxNumOfModelsToSegment();  //Default always the best, i.e. = 1

		///** \brief Additional criteria if many models are to be segmented
		//  * \param[in] set the min number of per model
		//  * \param[out] :ToDo: 
		//  * \return void 
		//  */
		//void setMinModelInliers(const unsigned  int &_minModelInliers = 2); //default is 2 for the case that the model is a line
		//unsigned int getMinModelInliers(); 

		/** \brief Determine if the output should be printed to a file
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		void setOutFile(const bool &_f = true );

		/** \brief Extract the segmentation results into the 3 tuple
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] Model Segments in the form |inliers|ModelCoefficients|Indices
		* \return void 
		*/
		void _extractVPs(Eigen::VectorXf &outSVPs);



		/** \brief Determine wherther a second robust statistical check should be run over the results or not
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] Model Segments in the form |inliers|ModelCoefficients|Indices
		* \return void 
		*/
		void _validateResults(bool fValRes = true);


		/** \brief Extracts the vg filtered Cloud
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] Model Segments in the form |inliers|ModelCoefficients|Indices
		* \return void 
		*/
		void _getvoxelgridedFilteredCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &vgFilteredCloud);



		/** \brief Do the conversion by inspecting the file extension
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		virtual void ComputeVPDetection() = 0;

	protected:

		cv::Mat m_InputImg;
		pcl::PointCloud<pcl::PointXYZ>::Ptr  m_InputCloud;
		pcl::PointCloud<pcl::PointXYZ>::Ptr  m_vgFilteredCloud;
		VPModelParamsPtr m_ModelParams; 
		std::string m_InputFile;
		std::vector<Eigen::VectorXf> m_EstimatedVPs;
		bool m_ValidateResults;
		/*unsigned int m_MaxNumModels;*/
		/*unsigned int m_MinModelInliers;*/
	private:
		//ToDo: set this to private maybe
		void _robustFittingBestHypothesis(pcl::PointCloud<pcl::PointXYZ>::Ptr &fVPs, Eigen::VectorXf &optimized_coefficients);
		void _OLSFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &fInClouds, Eigen::VectorXf &optimized_coefficients1);		     
	};

}

#endif// _VPDetection_H_
