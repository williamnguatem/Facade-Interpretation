#include "model_context.h"

fi::ModelContext::ModelContext(fi::ModelSegmentation *_model_segmentation_strategy): m_ModelSegmentationStrategy(_model_segmentation_strategy)
{

}


void fi::ModelContext::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_m_InputCloud)
{
	m_ModelSegmentationStrategy->_setInputCloud(_m_InputCloud);
}


void fi::ModelContext::setModelParams(const SegModelParamsPtr &_modelParams)
{
	m_ModelSegmentationStrategy->_setModelParameter(_modelParams);
}


void fi::ModelContext::setSegmentationStrategy(fi::ModelSegmentation *_fModelSegmentationStrategy)
{
	m_ModelSegmentationStrategy = _fModelSegmentationStrategy;
}

fi::ModelSegmentation* fi::ModelContext::getSegStrategy()
{
	return m_ModelSegmentationStrategy;
}


void fi::ModelContext::ExecuteSegmentation()
{
	m_ModelSegmentationStrategy->ExecuteSegmentation();
}

void fi::ModelContext::extractSegments(std::vector<SegmentationResultPtr> &_outSegments)
{
	m_ModelSegmentationStrategy->_extractSegments(_outSegments);
}