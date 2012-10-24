//******************************************************************************
//\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
//\file		plane_model.h
//\author		William Nguatem
//\note		Copyright (C) 
//\note		Bundeswehr University Munich
//\note		Institute of Applied Computer Science
//\note		Chair of Photogrammetry and Remote Sensing
//\note		Neubiberg, Germany
//\since		2012/02/07 
//******************************************************************************/

//#pragma once;

#ifndef _PLANE_MODEL_H_
#define _PLANE_MODEL_H_

#include <string>
//#include "FacadeDataTypes/FacadeDataTypes.h"
#include "model_segmentation.h"


//*!
//\brief 		Base class for all model segmentation classes
//\ingroup 	ModelSegmentation
//*/

namespace fi{

	class PlaneModel :public ModelSegmentation
	{
	public:

		//PlaneModel();
		/*virtual ~PlaneModel();*/

		/** \brief Set the name of the input cloud
		* \param[in] Input Cloud
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		/*void setInputCloud(const PtCloudDataPtr &_inputCloud);*/


		///** \brief Set the desired model parameters
		//  * \param[in] Parameters defining the model
		//  * \param[out] :ToDo: 
		//  * \return void 
		//  */

		//void setModelParameter(const SegModelParamsPtr &_modelParams); 

		/** \brief Determine if the output should be printed to a file
		* \param[in] true if the output should be printed to a file els retain the output in the system for further processing
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		void ExecuteSegmentation();

		void ExecuteSegmentationOld();


		/** \brief project all inliers on to the plane with coefficients ax + by + cz + d = 0
		* \param[in] 
		* \param[out]
		* \return void 
		*/
		void projectPointsOnPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_inliers, const Eigen::Vector4f &plane_coefficents, pcl::PointCloud<pcl::PointXYZ>::Ptr &projected_inliers);



		/** \brief Had to re-implement this my self since the pcl method has some issues
		* \param[in] 
		* \param[in] 
		* \param[in] 
		* \param[in] 
		* \param[in] 
		* \param[out] :ToDo: If needed specify the name of the output file
		* \param[out] :ToDo: If needed specify the name of the output file
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		void PlanesPerpendicularToDirection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cloud, 
			pcl::PointCloud<pcl::PointXYZ>::Ptr &inliers_cloud,  
			pcl::PointIndices::Ptr &inlierIndices,  
			pcl::ModelCoefficients::Ptr &coefficients, 
			pcl::PointCloud<pcl::PointXYZ>::Ptr &remainder_cloud,
			const Eigen::Vector3f &vertical_direction,
			const float ransac_threshold, 
			unsigned int num_iterations = 1000
			);


		void PlanesPerpendicularToDirection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cloud, 
			const pcl::PointIndices::Ptr &indices_to_consider, 
			pcl::PointIndices::Ptr &inlier_indices, 
			pcl::PointIndices::Ptr &remainder_indices,  
			pcl::ModelCoefficients::Ptr &model_coefficients, 
			const Eigen::Vector3f &vanishing_point, 
			const float ransac_threshold, 
			unsigned int num_iterations
			);


		void LSPlaneFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::ModelCoefficients::Ptr &model_coefficients);


		//unsigned int computeMeanAndCovarianceMatrix (const pcl::PointCloud<pcl::PointXYZ> &cloud,
		//	const std::vector<int> &indices,
		//	Eigen::Matrix3d &covariance_matrix,
		//	Eigen::Vector4d &centroid);

		//unsigned int computeMeanAndCovarianceMatrix (const pcl::PointCloud<pcl::PointXYZ> &cloud,
		//	Eigen::Matrix3d &covariance_matrix,
		//	Eigen::Vector4d &centroid);



		//private:
		//
		//	PtCloudDataPtr m_InputCloud;
		//	SegModelParamsPtr  m_ModelParams;
		//	std::string m_InputFile;
		//	bool m_f;

	};

}

#endif// _PLANE_MODEL_H_