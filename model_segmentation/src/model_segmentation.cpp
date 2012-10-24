#include "model_segmentation.h"


//fi::ModelSegmentation::ModelSegmentation(){}

fi::ModelSegmentation::~ModelSegmentation(){}


void fi::ModelSegmentation::_setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cloud)
{
	m_input_cloud = _cloud;
}


void fi::ModelSegmentation::_setModelParameter(const SegModelParamsPtr &_model_params)
{
	m_ModelParams = _model_params;
}


void fi::ModelSegmentation::setMaxNumOfModelsToSegment(const unsigned  int &_max_number_of_models )
{
	/*m_MaxNumModels = _maxNumModels;*/
	m_ModelParams->max_num_of_models = _max_number_of_models;
}


void fi::ModelSegmentation::setMinModelInliers(const unsigned  int &_min_num_inliers )
{
	/*m_MinModelInliers = _minModelInliers;*/
	m_ModelParams->min_num_inliers = _min_num_inliers;
}

unsigned int fi::ModelSegmentation::getMaxNumOfModelsToSegment()
{
	return m_ModelParams->max_num_of_models;
}


unsigned int fi::ModelSegmentation::getMinModelInliers()
{
	return m_ModelParams->min_num_inliers;
}


bool fi::ModelSegmentation::getSegmentAllModels()
{
	if (m_ModelParams->max_num_of_models > 1)
	{
		return true;
	}
}


void fi::ModelSegmentation::_extractSegments(std::vector<SegmentationResultPtr> &_out_segments)
{
	 _out_segments = m_ExtractedSegments;
}