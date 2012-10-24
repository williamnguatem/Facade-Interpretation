/******************************************************************************
\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
\file		VPDetectionContext.h
\author		William Nguatem
\note		Copyright (C) 
\note		Bundeswehr University Munich
\note		Institute of Applied Computer Science
\note		Chair of Photogrammetry and Remote Sensing
\note		Neubiberg, Germany
\since		2012/02/07 
******************************************************************************/

//#pragma once;

#ifndef _VPDetectionContext_H_
#define _VPDetectionContext_H_

#include "vp_detection.h"
#include <string>
/*!
\brief 		Defines the various segmentation context such that the base clase execute will select the correct model to execute segmentation
\ingroup 	ModelSegmentation
*/

namespace fi
{

	class VPDetectionContext
	{
	public:

		/** \brief Set the name of the input cloud
		* \param[in] Input Cloud
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		explicit VPDetectionContext(VPDetection *_fVPDetectionStrategy);

		/** \brief Set the desired model parameters
		* \param[in] Parameters defining the model
		* \param[out] :ToDo: 
		* \return void 
		*/

		void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_inputCloud);

		void setInputImg(const cv::Mat &_inputImg);

		void setModelParams(const VPModelParamsPtr &_mode);

		void validateResults(bool fValRes = true);

		void extractVPs(Eigen::VectorXf &_outSVPs);

		void getvoxelgridedFilteredCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr &vgFilteredCloud);

		void setVPStrategy (VPDetection *_fVPDetectionStrategy);


		/** \brief Get the strategy type: plane, cube, cone, ....
		* \param[in] 
		* \param[out] :ToDo: 
		* \return Pointer to the type of model segmented 
		*/
		VPDetection* getVPStrategy();

		/** \brief Call the true segmentation according to the input model
		* \param[in] Parameters defining the model
		* \param[out] :ToDo: 
		* \return void 
		*/

		void ComputeVPDetection();


	private:
		VPDetection *m_VPDetectionStrategy;
	};

}

#endif// _VPDetectionContext_H_