//******************************************************************************
//\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
//\file		ModelContext.h
//\author		William Nguatem
//\note		Copyright (C) 
//\note		Bundeswehr University Munich
//\note		Institute of Applied Computer Science
//\note		Chair of Photogrammetry and Remote Sensing
//\note		Neubiberg, Germany
//\since		2012/02/07 
//******************************************************************************/

//#pragma once;

#ifndef _MODEL_CONTEXT_H_
#define _MODEL_CONTEXT_H_

#include <string>
#include "model_segmentation.h"

//*!
//\brief 		Defines the various segmentation context such that the base clase execute will select the correct model to execute segmentation
//\ingroup 	ModelSegmentation
//*/

namespace fi
{

	class ModelContext
	{
	public:

		/** \brief Set the name of the input cloud
		* \param[in] Input Cloud
		* \param[out] :ToDo: If needed specify the name of the output file
		* \return void 
		*/
		explicit ModelContext(fi::ModelSegmentation *_model_segmentation_strategy);

		/** \brief Set the desired model parameters
		* \param[in] Parameters defining the model
		* \param[out] :ToDo: 
		* \return void 
		*/

		void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_m_InputCloud);

		void setModelParams(const SegModelParamsPtr &_mode);

		void extractSegments(std::vector<SegmentationResultPtr> &outSegments);

		void setSegmentationStrategy (fi::ModelSegmentation *_model_segmentation_strategy);


		/** \brief Get the strategy type: plane, cube, cone, ....
		* \param[in] 
		* \param[out] :ToDo: 
		* \return Pointer to the type of model segmented 
		*/
		fi::ModelSegmentation* getSegStrategy();

		/** \brief Call the true segmentation according to the input model
		* \param[in] Parameters defining the model
		* \param[out] :ToDo: 
		* \return void 
		*/
		void ExecuteSegmentation();


	private:
		fi::ModelSegmentation *m_ModelSegmentationStrategy;
	};

}

#endif// _ModelSegmentation_H_