#include "vp_detection_context.h"

fi::VPDetectionContext::VPDetectionContext(VPDetection *_fVPDetectionStrategy): m_VPDetectionStrategy(_fVPDetectionStrategy)
{

}

void fi::VPDetectionContext::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_inputCloud)
{
	m_VPDetectionStrategy->_setInputCloud( _inputCloud);
}


void fi::VPDetectionContext::setInputImg(const cv::Mat &_inputImg)
{
	m_VPDetectionStrategy->_setInputImg(_inputImg);
}


void fi::VPDetectionContext::setModelParams(const VPModelParamsPtr &_modelParams)
{
	m_VPDetectionStrategy->_setModelParameter(_modelParams);
}


void fi::VPDetectionContext::setVPStrategy(fi::VPDetection *_fVPDetectionStrategy)
{
	m_VPDetectionStrategy = _fVPDetectionStrategy;
}

fi::VPDetection* fi::VPDetectionContext::getVPStrategy()
{
	return m_VPDetectionStrategy;
}

void fi::VPDetectionContext::validateResults(bool fValRes)
{
	m_VPDetectionStrategy->_validateResults(fValRes);
}

void fi::VPDetectionContext::getvoxelgridedFilteredCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &vgFilteredCloud)
{
	m_VPDetectionStrategy->_getvoxelgridedFilteredCloud(vgFilteredCloud);
}

void fi::VPDetectionContext::ComputeVPDetection()
{
	m_VPDetectionStrategy->ComputeVPDetection();
}

void fi::VPDetectionContext::extractVPs(Eigen::VectorXf &_outSVPs)
{
	//check first if the results are to be validated then re-run a robust fitting routine.
	// otherwise select one from the list and give out as the VP!
	m_VPDetectionStrategy->_extractVPs(_outSVPs);
}

