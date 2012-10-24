/******************************************************************************
\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
\file		model_segmentation.h
\author		William Nguatem
\note		Copyright (C) 
\note		Bundeswehr University Munich
\note		Institute of Applied Computer Science
\note		Chair of Visual Computing
\note		Neubiberg, Germany
\since		2012/02/07 
******************************************************************************/
#ifndef _MODEL_SEGMENTATION_H_
#define _MODEL_SEGMENTATION_H_

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <boost/tuple/tuple.hpp> 
#include <boost/tuple/tuple_io.hpp> 


/*!
\brief 		Base class for all model segmentation classes
\ingroup 	ModelSegmentation
*/

namespace fi{

	//ToDo: Aligne the data
	struct SegModelParam 
	{
		unsigned int num_of_iterations;
		unsigned int min_num_inliers;
		unsigned int max_num_of_models;
		float ransac_thresh;
		Eigen::Vector3f vanishing_point; //Direction perpendicular to plane normal //ToDo: aligne using eigen align
		Eigen::Vector3f voxel_grid_size;//Size of the sub sampling filters leafs
		bool multicore_support;
		bool gpu_support;
	};

	typedef SegModelParam SegModelParams;
	typedef boost::shared_ptr<SegModelParams> SegModelParamsPtr;

	//Results are placed in the form:     |ModelCoefs|InlierIndices of model
	typedef boost::tuple< pcl::ModelCoefficients::Ptr, pcl::PointIndices::Ptr > SegmentationResult;
	typedef boost::shared_ptr <SegmentationResult> SegmentationResultPtr;

	class ModelSegmentation
	{
	public:
		//ModelSegmentation();

		virtual ~ModelSegmentation();

		/** \brief Set the name of the input cloud
		* \param[in] Input Cloud
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		void _setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cloud);

		/** \brief Set the desired model parameters
		* \param[in] Parameters defining the model
		* \param[out] :ToDo: 
		* \return void 
		*/
		void _setModelParameter(const SegModelParamsPtr &_modelParams); 


		/** \brief Decide if we should segment all models
		* \param[in] true = segment all models else chose only the best
		* \param[out] :ToDo: 
		* \return void 
		*/
		void setSegmentAllModels(const bool &_segmentAll); 

		bool getSegmentAllModels(); 

		/** \brief Additional criteria if many models are to be segmented
		* \param[in] The maximum number of models to be segmented from the cloud
		* \param[out] :ToDo: 
		* \return void 
		*/
		void setMaxNumOfModelsToSegment(const unsigned  int &_maxNumModels = 1);  //Default always the best
		unsigned int getMaxNumOfModelsToSegment();  //Default always the best, i.e. = 1

		/** \brief Additional criteria if many models are to be segmented
		* \param[in] set the min number of per model
		* \param[out] :ToDo: 
		* \return void 
		*/
		void setMinModelInliers(const unsigned  int &_minModelInliers = 2); //default is 2 for the case that the model is a line
		unsigned int getMinModelInliers(); 

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
		void _extractSegments(std::vector<SegmentationResultPtr> &outSegments);

		/** \brief Do the conversion by inspecting the file extension
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		virtual void ExecuteSegmentation() = 0;

	protected:

		std::vector<SegmentationResultPtr> m_ExtractedSegments;
		pcl::PointCloud<pcl::PointXYZ>::Ptr m_input_cloud;
		SegModelParamsPtr m_ModelParams; //This is just enough since the plane model is the minimum model and has the minimum params i.e equal to the params of a line too!
		std::string m_InputFile;
		/*bool m_SegmentAll;*/
		/*unsigned int m_MaxNumModels;*/
		/*unsigned int m_MinModelInliers;*/

	};
}

#endif// _MODEL_SEGMENTATION_H_
