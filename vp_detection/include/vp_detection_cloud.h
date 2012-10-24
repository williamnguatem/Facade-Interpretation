/******************************************************************************
\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
\file		VPDetectionCloud.h
\author		William Nguatem
\note		Copyright (C) 
\note		Bundeswehr University Munich
\note		Institute of Applied Computer Science
\note		Chair of Photogrammetry and Remote Sensing
\note		Neubiberg, Germany
\since		2012/02/07 
******************************************************************************/

//#pragma once;

#ifndef _VPDetectionCloud_H_
#define _VPDetectionCloud_H_

#include <string>
#include "vp_detection.h"


/*!
\brief 		Vanishing Point Detection on a point cloud containing scaned buildings
\ingroup 	VanishingPointDetection
*/

namespace fi
{

	class VPDetectionCloud :public VPDetection
	{
	public:

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
		void ClusterKDirectionPatches(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_InCloud, Eigen::VectorXf &fRobustEstimatedVP);



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
		void ComputeVPDetection();

		void LSLineFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &fInClouds, Eigen::VectorXf &optimized_coefficients);


		//private:
		//
		//	PtCloudDataPtr m_InputCloud;
		//	SegModelParamsPtr  m_ModelParams;
		//	std::string m_InputFile;
		//	bool m_f;

	};

}

#endif// _VPDetectionCloud_H_