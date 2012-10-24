#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/angles.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/common.h>
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"


#include "plane_model.h"
#include "model_segmentation.h"
#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp> 
#include <boost/tuple/tuple_io.hpp> 


void fi::PlaneModel::ExecuteSegmentationOld()
{

	//Check if Down Sampling required using a Voxel grid filter
	// Create the filtering object: down sample the dataset using a leaf size of 1cm
	float aLeaf = m_ModelParams->voxel_grid_size(0);
	float bLeaf = m_ModelParams->voxel_grid_size(1);
	float cLeaf = m_ModelParams->voxel_grid_size(2);


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZ>);

	if (!((aLeaf == 0.0f) && (bLeaf == 0.0f) &&(cLeaf == 0.0f)) )
	{
		pcl::VoxelGrid<pcl::PointXYZ> vg;
		vg.setInputCloud (m_input_cloud);
		vg.setLeafSize (aLeaf, bLeaf, cLeaf); //fachwerkhaus
		//vg.setLeafSize (0.04f, 0.04f, 0.04f);  //Schloss Etlingen
		//vg.setLeafSize (0.1f, 0.1f, 0.1f); //unichurch
		vg.filter (*cloudFiltered);
		std::cout << "PointCloud before filtering has: " << m_input_cloud->points.size ()  << " data points." << std::endl; //*
		std::cout << "PointCloud after filtering has: " << cloudFiltered->points.size ()  << " data points." << std::endl; //*
	}
	else
	{
		//this is necessary since the size will be reducing by successive execution of the segmentation routine 
		pcl::copyPointCloud(*m_input_cloud, *cloudFiltered);
	}

	std::vector<SegmentationResultPtr> computedSegments;

	//check if oriented plane segmentation is required
	float xPpnormal = m_ModelParams->vanishing_point(0);
	float yPpnormal = m_ModelParams->vanishing_point(1);
	float zPpnormal = m_ModelParams->vanishing_point(2);

	if (!((xPpnormal == 0.0f) && (yPpnormal == 0.0f) &&(zPpnormal == 0.0f)) )
	{
		//Do oriented plane segmentation
		for(unsigned int i = 0; i < m_ModelParams->max_num_of_models; i++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Inliers (new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remainders (new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);
			pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
			//Vec3d dVec = {xPpnormal, yPpnormal, zPpnormal};
			//Vec3dPtr RefDirectionVector(new Vec3d (dVec));
			PlanesPerpendicularToDirection(cloudFiltered, cloud_Inliers, inlierIndices, coefficients, cloud_remainders, m_ModelParams->vanishing_point, m_ModelParams->ransac_thresh, m_ModelParams->num_of_iterations);

			if (inlierIndices->indices.size () == 0)
			{
				PCL_ERROR ("Could not estimate a planar model for the given dataset.");
				break;
			}

			if (inlierIndices->indices.size() < m_ModelParams->min_num_inliers)
			{
				PCL_ERROR ("Number of suporting inliers smaller than the min allowable inliers for the param set.");
				break;
			}

			//Sorry folks if this line is too many boost stuff, I didn't want to write two lines just for this ;)
			//fIntersectorsPtr fInitToNull(new fIntersectors(NULL));
			pcl::PointCloud<pcl::PointXYZ>::Ptr projected_inliers(new pcl::PointCloud<pcl::PointXYZ>);

			Eigen::Vector4f plane_coefficients(coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
			plane_coefficients.head<3>().normalize();

			projectPointsOnPlane(cloud_Inliers, plane_coefficients, projected_inliers);
			
		/*	SegmentationResultPtr tmpRes(new SegmentationResult(boost::make_tuple(boost::ref(cloud_Inliers), boost::ref(coefficients), boost::ref(inlierIndices), boost::ref(projected_inliers))));
			computedSegments.push_back(tmpRes);
			cloudFiltered = cloud_remainders;*/
		}

		//copy result for the output
		if (computedSegments.size() == 0)
		{
			PCL_ERROR ("Segmentation Failed!");
		}
		else
		{
			m_ExtractedSegments = computedSegments;
		}
		
	}
	else
	{
		//Do simple straight forward plane segmentation

		for(unsigned int i = 0; i < m_ModelParams->max_num_of_models; i++)
		{

			pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			pcl::PointIndices::Ptr inlierIndices (new pcl::PointIndices);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers (new pcl::PointCloud<pcl::PointXYZ>), 
				cloudRemaining (new pcl::PointCloud<pcl::PointXYZ>);

			// Create the segmentation object
			boost::scoped_ptr<pcl::SACSegmentation<pcl::PointXYZ>> seg(new pcl::SACSegmentation<pcl::PointXYZ>);
			// Optional
			seg->setOptimizeCoefficients (true);
			// Mandatory
			seg->setModelType (pcl::SACMODEL_PLANE);
			seg->setMethodType (pcl::SAC_RANSAC);
			seg->setDistanceThreshold (m_ModelParams->ransac_thresh);

			seg->setInputCloud (cloudFiltered);
			seg->segment (*inlierIndices, *coefficients);

			if (inlierIndices->indices.size () == 0)
			{
				PCL_ERROR ("Could not estimate a planar model for the given dataset.");
				break;
			}

			if (inlierIndices->indices.size() < m_ModelParams->min_num_inliers)
			{
				PCL_ERROR ("Number of suporting inliers smaller than the min allowable inliers for the param set.");
				break;
			}

			// Extract the planar inliers from the input cloud
			boost::scoped_ptr<pcl::ExtractIndices<pcl::PointXYZ>> extract(new pcl::ExtractIndices<pcl::PointXYZ>);
			extract->setInputCloud (cloudFiltered);
			extract->setIndices (inlierIndices);
			extract->setNegative (false);

			// Write the planar inliers to disk
			extract->filter (*cloudInliers);
			std::cout << "PointCloud representing the planar component: " << cloudInliers->points.size () << " data points." << std::endl;

			// Remove the planar inliers, extract the rest
			extract->setNegative (true);
			extract->filter (*cloudRemaining);
			cloudFiltered = cloudRemaining;

			pcl::PointCloud<pcl::PointXYZ>::Ptr projected_inliers (new pcl::PointCloud<pcl::PointXYZ>);
			Eigen::Vector4f plane_coefficients(coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
			plane_coefficients.head<3>().normalize();

			projectPointsOnPlane(cloudInliers, plane_coefficients, projected_inliers);

			/*SegmentationResultPtr tmpRes(new SegmentationResult(boost::make_tuple(boost::ref(cloudInliers), boost::ref(coefficients), boost::ref(inlierIndices), boost::ref(projected_inliers))));
			computedSegments.push_back(tmpRes);*/
		}

		//copy result for the output
		if (computedSegments.size() == 0)
		{
			PCL_ERROR ("Segmentation Failed!");
			return;
		}
		else
		{
			m_ExtractedSegments = computedSegments;
		}
	}
}



void fi::PlaneModel::ExecuteSegmentation()
{

	//Check if Down Sampling required using a Voxel grid filter
	// Create the filtering object: down sample the dataset using a leaf size of 1cm
	float aLeaf = m_ModelParams->voxel_grid_size(0);
	float bLeaf = m_ModelParams->voxel_grid_size(1);
	float cLeaf = m_ModelParams->voxel_grid_size(2);


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZ>);

	if (!((aLeaf == 0.0f) && (bLeaf == 0.0f) &&(cLeaf == 0.0f)) )
	{
		pcl::VoxelGrid<pcl::PointXYZ> vg;
		vg.setInputCloud (m_input_cloud);
		vg.setLeafSize (aLeaf, bLeaf, cLeaf); //fachwerkhaus
		//vg.setLeafSize (0.04f, 0.04f, 0.04f);  //Schloss Etlingen
		//vg.setLeafSize (0.1f, 0.1f, 0.1f); //unichurch
		vg.filter (*cloudFiltered);
		std::cout << "PointCloud before filtering has: " << m_input_cloud->points.size ()  << " data points." << std::endl; //*
		std::cout << "PointCloud after filtering has: " << cloudFiltered->points.size ()  << " data points." << std::endl; //*
	}
	else
	{
		//this is necessary since the size will be reducing by successive execution of the segmentation routine 
		pcl::copyPointCloud(*m_input_cloud, *cloudFiltered);
	}

	std::vector<SegmentationResultPtr> computedSegments;

	//check if oriented plane segmentation is required
	float xPpnormal = m_ModelParams->vanishing_point(0);
	float yPpnormal = m_ModelParams->vanishing_point(1);
	float zPpnormal = m_ModelParams->vanishing_point(2);

	//porpulate the indices vectors
	pcl::PointIndices::Ptr indices_to_consider(new pcl::PointIndices);
	indices_to_consider->indices.resize(cloudFiltered->points.size());
	for (unsigned int i = 0; i < cloudFiltered->points.size(); i++)
	{
		indices_to_consider->indices[i] = i;
	}

	if (!((xPpnormal == 0.0f) && (yPpnormal == 0.0f) &&(zPpnormal == 0.0f)) )
	{

		//Do oriented plane segmentation
		for(unsigned int i = 0; i < m_ModelParams->max_num_of_models; i++)
		{
			pcl::PointIndices::Ptr inliers_indices(new pcl::PointIndices);
			pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
			pcl::PointIndices::Ptr remainder_indices(new pcl::PointIndices);

			PlanesPerpendicularToDirection(cloudFiltered, indices_to_consider, inliers_indices, remainder_indices, coefficients, m_ModelParams->vanishing_point, m_ModelParams->ransac_thresh, m_ModelParams->num_of_iterations);

			if (inliers_indices->indices.size () == 0)
			{
				PCL_ERROR ("Could not estimate a planar model for the given dataset.");
				break;
			}

			if (inliers_indices->indices.size() < m_ModelParams->min_num_inliers)
			{
				PCL_ERROR ("Number of suporting inliers smaller than the min allowable inliers for the param set.");
				break;
			}

			//Sorry folks if this line is too many boost stuff, I didn't want to write two lines just for this ;)
			//fIntersectorsPtr fInitToNull(new fIntersectors(NULL));
			pcl::PointCloud<pcl::PointXYZ>::Ptr projected_inliers(new pcl::PointCloud<pcl::PointXYZ>);

		/*	Eigen::Vector4f plane_coefficients(coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
			plane_coefficients.head<3>().normalize();*/

		//	projectPointsOnPlane(cloud_Inliers, plane_coefficients, projected_inliers);
			
			SegmentationResultPtr tmpRes(new SegmentationResult(boost::make_tuple( boost::ref(coefficients), boost::ref(inliers_indices))));
			computedSegments.push_back(tmpRes);
			indices_to_consider = remainder_indices;
		}

		//copy result for the output
		if (computedSegments.size() == 0)
		{
			PCL_ERROR ("Segmentation Failed!");
		}
		else
		{
			m_ExtractedSegments = computedSegments;
		}
		
	}
	else
	{
		////Do simple straight forward plane segmentation

		//for(unsigned int i = 0; i < m_ModelParams->max_num_of_models; i++)
		//{

		//	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		//	pcl::PointIndices::Ptr inlierIndices (new pcl::PointIndices);
		//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers (new pcl::PointCloud<pcl::PointXYZ>), 
		//		cloudRemaining (new pcl::PointCloud<pcl::PointXYZ>);

		//	// Create the segmentation object
		//	boost::scoped_ptr<pcl::SACSegmentation<pcl::PointXYZ>> seg(new pcl::SACSegmentation<pcl::PointXYZ>);
		//	// Optional
		//	seg->setOptimizeCoefficients (true);
		//	// Mandatory
		//	seg->setModelType (pcl::SACMODEL_PLANE);
		//	seg->setMethodType (pcl::SAC_RANSAC);
		//	seg->setDistanceThreshold (m_ModelParams->ransac_thresh);

		//	seg->setInputCloud (cloudFiltered);
		//	seg->segment (*inlierIndices, *coefficients);

		//	if (inlierIndices->indices.size () == 0)
		//	{
		//		PCL_ERROR ("Could not estimate a planar model for the given dataset.");
		//		break;
		//	}

		//	if (inlierIndices->indices.size() < m_ModelParams->min_num_inliers)
		//	{
		//		PCL_ERROR ("Number of suporting inliers smaller than the min allowable inliers for the param set.");
		//		break;
		//	}

		//	// Extract the planar inliers from the input cloud
		//	boost::scoped_ptr<pcl::ExtractIndices<pcl::PointXYZ>> extract(new pcl::ExtractIndices<pcl::PointXYZ>);
		//	extract->setInputCloud (cloudFiltered);
		//	extract->setIndices (inlierIndices);
		//	extract->setNegative (false);

		//	// Write the planar inliers to disk
		//	extract->filter (*cloudInliers);
		//	std::cout << "PointCloud representing the planar component: " << cloudInliers->points.size () << " data points." << std::endl;

		//	// Remove the planar inliers, extract the rest
		//	extract->setNegative (true);
		//	extract->filter (*cloudRemaining);
		//	cloudFiltered = cloudRemaining;

		//	pcl::PointCloud<pcl::PointXYZ>::Ptr projected_inliers (new pcl::PointCloud<pcl::PointXYZ>);
		//	Eigen::Vector4f plane_coefficients(coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
		//	plane_coefficients.head<3>().normalize();

		//	projectPointsOnPlane(cloudInliers, plane_coefficients, projected_inliers);

		//	//SegmentationResultPtr tmpRes(new SegmentationResult(boost::make_tuple(boost::ref(cloudInliers), boost::ref(coefficients), boost::ref(inlierIndices), boost::ref(projected_inliers))));
		//	//computedSegments.push_back(tmpRes);
		//}

		////copy result for the output
		//if (computedSegments.size() == 0)
		//{
		//	PCL_ERROR ("Segmentation Failed!");
		//	return;
		//}
		//else
		//{
		//	m_ExtractedSegments = computedSegments;
		//}
	}
}









void fi::PlaneModel::projectPointsOnPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud, const Eigen::Vector4f &plane_coefficents, pcl::PointCloud<pcl::PointXYZ>::Ptr &projected_cloud)
{
	unsigned int n_points = plane_cloud->points.size();
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_out_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	tmp_out_cloud->width = n_points;
	tmp_out_cloud->height = 1;
	tmp_out_cloud->resize(tmp_out_cloud->width * tmp_out_cloud->height);

	for (unsigned int i = 0 ; i < n_points; i++)
	{
		pcl::projectPoint(plane_cloud->points[i], plane_coefficents, tmp_out_cloud->points[i]);
	}
	projected_cloud = tmp_out_cloud;
}



void fi::PlaneModel::PlanesPerpendicularToDirection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &inliers_cloud,  pcl::PointIndices::Ptr &inlier_indices,  pcl::ModelCoefficients::Ptr &model_coefficients, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_remainders,  const Eigen::Vector3f &vanishing_point, const float ransac_threshold, unsigned int num_iterations)
{

	/*const Vec3dPtr &tmpVPDirection = RefDirectionVector;*/
	float wRANSACThreshold = ransac_threshold;

	int nPointCandidates = input_cloud->points.size();
	typedef std::vector<int > Inx;
	std::vector<Inx> InliersInices, sDataIndex;
	int i, maxSupport = 0;
	float best_c = 0.0f;
	Eigen::Vector3f best_n;
	best_n(0) = 0.0f;
	best_n(1) = 0.0f;
	best_n(2) = 0.0f;
/*
	InInxsPtr AllDataIndxs(new InIndxs);

	InInxsPtr maxSupportIndxs(new InIndxs);

	CVec3dArrayPtr wmaxInliers(new CVec3dArray);*/


	// Initialize a random number generator.
	// Boost provides a bunch of these, note that some of them are not meant
	// for direct user usage and you should instead use a specialization (for 
	// example, don't use linear_congruential and use minstd_rand or 
	// minstd_rand0 instead)

	// This constructor seeds the generator with the current time.
	// As mentioned in Boost's sample program, time(0) is not a great seed,
	// but you can probably get away with it for most situations.
	// Consider using more precise timers such as gettimeofday on *nix or
	// GetTickCount/timeGetTime/QueryPerformanceCounter on Windows.
	boost::mt19937 randGen(std::time(0));

	// Now we set up a distribution. Boost provides a bunch of these as well.
	// This is the preferred way to generate numbers in a certain range.
	// initialize a uniform distribution between 0 and the max=nPointCandidates

	boost::uniform_int<> uInt8Dist(0, nPointCandidates);

	// Finally, declare a variate_generator which maps the random number
	// generator and the distribution together. This variate_generator
	// is usable like a function call.
	boost::variate_generator< boost::mt19937&, boost::uniform_int<> > 
		GetRand(randGen, uInt8Dist);

	// Generate a random number
	int aRandomNumber = GetRand();


	for (i = 0; i < num_iterations; i++)
	{
		// identify 2 different points
		const int nFirstIndex = GetRand() % nPointCandidates;
		int nTempIndex;

		//do { nTempIndex = rand() % nPointCandidates; } while (nTempIndex == nFirstIndex);
		//const int nSecondIndex = nTempIndex;

		do { nTempIndex = GetRand() % nPointCandidates; } while (nTempIndex == nFirstIndex);
		const int nSecondIndex = nTempIndex;

		//do { nTempIndex = rand() % nPointCandidates; } while (nTempIndex == nFirstIndex || nTempIndex == nSecondIndex);
		Eigen::Vector3f p1, p2;

		p1(0) = input_cloud->points[nFirstIndex].x;
		p1(1) = input_cloud->points[nFirstIndex].y;
		p1(2) = input_cloud->points[nFirstIndex].z;
		p2(0) = input_cloud->points[nSecondIndex].x;
		p2(1) = input_cloud->points[nSecondIndex].y;
		p2(2) = input_cloud->points[nSecondIndex].z;
		/*const Vec3d &p2 = inData[nSecondIndex];*/
		//const Vec3d &p3 = pointCandidates[nTempIndex];
		Eigen::Vector3f sNormalVec1, sNormalVec2;

		//compute perpendicular plane to the first plane given two points p1 and p2
		const float sAPerpendicularPlane = vanishing_point(2)*(p2(1) - p1(1)) - vanishing_point(1)*(p2(2) - p1(2));
		const float sBPerpendicularPlane = (-1) * (vanishing_point(2)*(p2(0) - p1(0)) - vanishing_point(0)*(p2(2) - p1(2)));
		const float sCPerpendicularPlane = vanishing_point(1)*(p2(0) - p1(0)) - vanishing_point(0)*(p2(1) - p1(1));
		const float sDPerpendicularPlane = -p1(0)*sAPerpendicularPlane - p1(1)*sBPerpendicularPlane - p1(2)*sCPerpendicularPlane;


		sNormalVec2(0) = sAPerpendicularPlane;
		sNormalVec2(1) = sBPerpendicularPlane;
		sNormalVec2(2) = sCPerpendicularPlane;

		sNormalVec2 = sNormalVec2.normalized();

		////unit testing
		//float sAngle = Math3d::Angle(*tmpVPDirection, sNormalVec2);
		//float sAngleDegree = sAngle*180/CV_PI;
		//std::cerr <<"Angle in degrees between both planes is: ";
		//std::cerr <<sAngleDegree<<std::endl;

		/*const float sp1 = sNormalVec1.x * sNormalVec2.x + sNormalVec1.y * sNormalVec2.y + sNormalVec1.z * sNormalVec2.z;*/

		const float c = sNormalVec2.dot(p1);

		// count support
		int nSupport = 0;
		//InInxsPtr tmpInliersIndices(new InIndxs);
		//CVec3dArrayPtr tmpwMaxInliers(new CVec3dArray);


		for (int j = 0; j < nPointCandidates; j++)
		{
			Eigen::Vector3f tmPoint;
			tmPoint(0) = input_cloud->points[j].x;
			tmPoint(1) = input_cloud->points[j].y;
			tmPoint(2) = input_cloud->points[j].z;
		
			if (fabsf(sNormalVec2.dot(tmPoint) - c) <= ransac_threshold)
			{
				nSupport++; //one can actually get the inliers here too and be shifting then now and then to see which is the best ?
				//tmpInliersIndices->push_back(j);
				//tmpwMaxInliers->AddElement(tmPoint); // this step could be saved and ran only once to save processing time <==ToDo
			}
		}
/*
		InInxsPtr tmpSupportIndxs(new InIndxs(*tmpInliersIndices));*/

		// store if it is the current maximum
		if (nSupport > maxSupport)
		{
			maxSupport = nSupport;
			//Math3d::SetVec(best_n, sNormalVec2);
			best_n = sNormalVec2;
			best_c = c;
			//maxSupportIndxs = tmpSupportIndxs;
			//wmaxInliers = tmpwMaxInliers;
		}

	}//_nIterations


	//since we now know the exact number of inliers for the best plane
	pcl::PointIndices::Ptr _inlierIndices(new pcl::PointIndices);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _inliers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _remainder_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	for (int j = 0; j < nPointCandidates; j++)
	{
		Eigen::Vector3f tmPoint;
		tmPoint(0) = input_cloud->points[j].x;
		tmPoint(1) = input_cloud->points[j].y;
		tmPoint(2) = input_cloud->points[j].z;
		if (fabsf(best_n.dot(tmPoint) - best_c) <= ransac_threshold)
		{
			_inlierIndices->indices.push_back(j);
		}
	}

	// Extract the planar inliers from the input cloud
	boost::scoped_ptr<pcl::ExtractIndices<pcl::PointXYZ>> extract(new pcl::ExtractIndices<pcl::PointXYZ>);
	extract->setInputCloud (input_cloud);
	extract->setIndices (_inlierIndices);
	extract->setNegative (false);

	// Write the planar inliers to disk
	extract->filter (*_inliers_cloud);
	std::cout << "PointCloud representing the planar component: " << _inliers_cloud->points.size () << " data points." << std::endl;

	// Remove the planar inliers, extract the rest
	extract->setNegative (true);
	extract->filter (*_remainder_cloud);

	//weighted ls optimization
	pcl::ModelCoefficients::Ptr _modelCoefficients(new pcl::ModelCoefficients());
	LSPlaneFitting(_inliers_cloud, _modelCoefficients);

	//Copy all to output
	inliers_cloud = _inliers_cloud;
	cloud_remainders = _remainder_cloud;
	model_coefficients =_modelCoefficients;
	inlier_indices = _inlierIndices;
}


void fi::PlaneModel::PlanesPerpendicularToDirection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, 
	const pcl::PointIndices::Ptr &indices_to_consider, 
	pcl::PointIndices::Ptr &inlier_indices, 
	pcl::PointIndices::Ptr &remainder_indices,  
	pcl::ModelCoefficients::Ptr &model_coefficients, 
	const Eigen::Vector3f &vanishing_point, 
	const float ransac_threshold, 
	unsigned int num_iterations
	)
{

	/*const Vec3dPtr &tmpVPDirection = RefDirectionVector;*/
	float wRANSACThreshold = ransac_threshold;

	unsigned int total_num_points = input_cloud->points.size();
	unsigned int num_candidates = indices_to_consider->indices.size();

	//sanity check
	if (total_num_points < num_candidates)
	{
		PCL_ERROR("Number of points to consider has to be less than the number of points in the cloud");
		return;
	}


	typedef std::vector<int > Inx;
	std::vector<Inx> InliersInices, sDataIndex;
	int i, maxSupport = 0;
	float best_c = 0.0f;
	Eigen::Vector3f best_n;
	best_n(0) = 0.0f;
	best_n(1) = 0.0f;
	best_n(2) = 0.0f;
/*
	InInxsPtr AllDataIndxs(new InIndxs);

	InInxsPtr maxSupportIndxs(new InIndxs);

	CVec3dArrayPtr wmaxInliers(new CVec3dArray);*/


	// Initialize a random number generator.
	// Boost provides a bunch of these, note that some of them are not meant
	// for direct user usage and you should instead use a specialization (for 
	// example, don't use linear_congruential and use minstd_rand or 
	// minstd_rand0 instead)

	// This constructor seeds the generator with the current time.
	// As mentioned in Boost's sample program, time(0) is not a great seed,
	// but you can probably get away with it for most situations.
	// Consider using more precise timers such as gettimeofday on *nix or
	// GetTickCount/timeGetTime/QueryPerformanceCounter on Windows.
	boost::mt19937 randGen(std::time(0));

	// Now we set up a distribution. Boost provides a bunch of these as well.
	// This is the preferred way to generate numbers in a certain range.
	// initialize a uniform distribution between 0 and the max=nPointCandidates

	boost::uniform_int<> uInt8Dist(0, num_candidates);

	// Finally, declare a variate_generator which maps the random number
	// generator and the distribution together. This variate_generator
	// is usable like a function call.
	boost::variate_generator< boost::mt19937&, boost::uniform_int<> > 
		GetRand(randGen, uInt8Dist);

	// Generate a random number
	int aRandomNumber = GetRand();


	for (i = 0; i < num_iterations; i++)
	{
		// identify 2 different points
		const int nFirstIndex = GetRand() % num_candidates;
		int nTempIndex;

		//do { nTempIndex = rand() % nPointCandidates; } while (nTempIndex == nFirstIndex);
		//const int nSecondIndex = nTempIndex;

		do { nTempIndex = GetRand() % num_candidates; } while (nTempIndex == nFirstIndex);
		const int nSecondIndex = nTempIndex;

		//do { nTempIndex = rand() % nPointCandidates; } while (nTempIndex == nFirstIndex || nTempIndex == nSecondIndex);
		Eigen::Vector3f p1, p2;

		p1(0) = input_cloud->points[indices_to_consider->indices[nFirstIndex]].x;
		p1(1) = input_cloud->points[indices_to_consider->indices[nFirstIndex]].y;
		p1(2) = input_cloud->points[indices_to_consider->indices[nFirstIndex]].z;
		p2(0) = input_cloud->points[indices_to_consider->indices[nSecondIndex]].x;
		p2(1) = input_cloud->points[indices_to_consider->indices[nSecondIndex]].y;
		p2(2) = input_cloud->points[indices_to_consider->indices[nSecondIndex]].z;
		/*const Vec3d &p2 = inData[nSecondIndex];*/
		//const Vec3d &p3 = pointCandidates[nTempIndex];
		Eigen::Vector3f sNormalVec1, sNormalVec2;

		//compute perpendicular plane to the first plane given two points p1 and p2
		const float sAPerpendicularPlane = vanishing_point(2)*(p2(1) - p1(1)) - vanishing_point(1)*(p2(2) - p1(2));
		const float sBPerpendicularPlane = (-1) * (vanishing_point(2)*(p2(0) - p1(0)) - vanishing_point(0)*(p2(2) - p1(2)));
		const float sCPerpendicularPlane = vanishing_point(1)*(p2(0) - p1(0)) - vanishing_point(0)*(p2(1) - p1(1));
		const float sDPerpendicularPlane = -p1(0)*sAPerpendicularPlane - p1(1)*sBPerpendicularPlane - p1(2)*sCPerpendicularPlane;

		sNormalVec2(0) = sAPerpendicularPlane;
		sNormalVec2(1) = sBPerpendicularPlane;
		sNormalVec2(2) = sCPerpendicularPlane;

		sNormalVec2 = sNormalVec2.normalized();

		////unit testing
		//float sAngle = Math3d::Angle(*tmpVPDirection, sNormalVec2);
		//float sAngleDegree = sAngle*180/CV_PI;
		//std::cerr <<"Angle in degrees between both planes is: ";
		//std::cerr <<sAngleDegree<<std::endl;

		/*const float sp1 = sNormalVec1.x * sNormalVec2.x + sNormalVec1.y * sNormalVec2.y + sNormalVec1.z * sNormalVec2.z;*/

		const float c = sNormalVec2.dot(p1);

		// count support
		int nSupport = 0;
		//InInxsPtr tmpInliersIndices(new InIndxs);
		//CVec3dArrayPtr tmpwMaxInliers(new CVec3dArray);


		for (int j = 0; j < num_candidates; j++)
		{
			Eigen::Vector3f tmPoint;
			tmPoint(0) = input_cloud->points[indices_to_consider->indices[j]].x;
			tmPoint(1) = input_cloud->points[indices_to_consider->indices[j]].y;
			tmPoint(2) = input_cloud->points[indices_to_consider->indices[j]].z;
		
			if (fabsf(sNormalVec2.dot(tmPoint) - c) <= ransac_threshold)
			{
				nSupport++; //one can actually get the inliers here too and be shifting then now and then to see which is the best ?
				//tmpInliersIndices->push_back(j);
				//tmpwMaxInliers->AddElement(tmPoint); // this step could be saved and ran only once to save processing time <==ToDo
			}
		}
/*
		InInxsPtr tmpSupportIndxs(new InIndxs(*tmpInliersIndices));*/

		// store if it is the current maximum
		if (nSupport > maxSupport)
		{
			maxSupport = nSupport;
			//Math3d::SetVec(best_n, sNormalVec2);
			best_n = sNormalVec2;
			best_c = c;
			//maxSupportIndxs = tmpSupportIndxs;
			//wmaxInliers = tmpwMaxInliers;
		}

	}//_nIterations


	//since we now know the exact number of inliers for the best plane
	pcl::PointIndices::Ptr _inlierIndices(new pcl::PointIndices);
	pcl::PointIndices::Ptr _remainder_indices(new pcl::PointIndices);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _inliers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr _remainder_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	for (int j = 0; j < num_candidates; j++)
	{
		Eigen::Vector3f tmPoint;
		tmPoint(0) = input_cloud->points[indices_to_consider->indices[j]].x;
		tmPoint(1) = input_cloud->points[indices_to_consider->indices[j]].y;
		tmPoint(2) = input_cloud->points[indices_to_consider->indices[j]].z;
		
		if (fabsf(best_n.dot(tmPoint) - best_c) <= ransac_threshold)
		{
			_inlierIndices->indices.push_back(indices_to_consider->indices[j]);
		}
		else
		{
			_remainder_indices->indices.push_back(indices_to_consider->indices[j]);
		}
	}


	// Extract the planar in_liers from the input cloud
	boost::scoped_ptr<pcl::ExtractIndices<pcl::PointXYZ> > extract(new pcl::ExtractIndices<pcl::PointXYZ>);
	extract->setInputCloud (input_cloud);
	extract->setIndices (_inlierIndices);
	extract->setNegative (false);

	// Write the planar inliers to disk
	extract->filter (*_inliers_cloud);
	std::cout << "PointCloud representing the planar component: " << _inliers_cloud->points.size () << " data points." << std::endl;

	//// Remove the planar inliers, extract the rest
	//extract->setNegative (true);
	//extract->filter (*_remainder_cloud);

	//weighted ls optimization
	pcl::ModelCoefficients::Ptr _modelCoefficients(new pcl::ModelCoefficients());
	LSPlaneFitting(_inliers_cloud, _modelCoefficients);

	////Copy all to output
	//inliers_cloud = _inliers_cloud;
	//cloud_remainders = _remainder_cloud;
	model_coefficients =_modelCoefficients;
	inlier_indices = _inlierIndices;
	remainder_indices = _remainder_indices;
}




void fi::PlaneModel::LSPlaneFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::ModelCoefficients::Ptr &model_coefficients)
{
	//// Needs a valid set of model coefficients
	//if (model_coefficients.size () != 4)
	//{
	//	PCL_ERROR ("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] Invalid number of model coefficients given (%lu)!\n", (unsigned long)model_coefficients.size ());
	//	optimized_coefficients = model_coefficients;
	//	return;
	//}

	//// Need at least 3 points to estimate a plane
	//if (inliers.size () < 4)
	//{
	//	PCL_ERROR ("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] Not enough inliers found to support a model (%lu)! Returning the same coefficients.\n", (unsigned long)inliers.size ());
	//	optimized_coefficients = model_coefficients;
	//	return;
	//}

	if(in_cloud->points.size() < 4)
	{
		std::cout<<"Need at least 3 points to estimate a plane"<<std::endl;
		return;
	}

	Eigen::Vector4d plane_parameters;

	// Use ordinary Least-Squares to fit the plane through all the given sample points and find out its coefficients
	EIGEN_ALIGN16 Eigen::Matrix3d covariance_matrix;
	Eigen::Vector4d xyz_centroid;

	// Estimate the XYZ centroid
	//pcl::compute3DCentroid (*in_cloud, xyz_centroid);
		
	xyz_centroid[3] = 0;

	// Compute the 3x3 covariance matrix
	//pcl::computeCovarianceMatrix (*in_cloud, xyz_centroid, covariance_matrix);
	pcl::computeMeanAndCovarianceMatrix(*in_cloud, covariance_matrix, xyz_centroid);


	// Compute the model coefficients
	EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
	EIGEN_ALIGN16 Eigen::Matrix3d eigen_vectors;
	pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);

	// Hessian form (D = nc . p_plane (centroid here) + p)
	Eigen::VectorXd optimized_coefficients;
	optimized_coefficients.resize (4);
	optimized_coefficients[0] = eigen_vectors (0, 0);
	optimized_coefficients[1] = eigen_vectors (1, 0);
	optimized_coefficients[2] = eigen_vectors (2, 0);
	optimized_coefficients[3] = 0;
	optimized_coefficients[3] = -1 * optimized_coefficients.dot (xyz_centroid);

	model_coefficients->values.push_back(optimized_coefficients(0));
	model_coefficients->values.push_back(optimized_coefficients(1));
	model_coefficients->values.push_back(optimized_coefficients(2));
	model_coefficients->values.push_back(optimized_coefficients(3));
}

//
// unsigned int fi::PlaneModel::computeMeanAndCovarianceMatrix (const pcl::PointCloud<pcl::PointXYZ> &cloud,
//	Eigen::Matrix3d &covariance_matrix,
//	Eigen::Vector4d &centroid)
//{
//	// create the buffer on the stack which is much faster than using cloud.points[indices[i]] and centroid as a buffer
//	Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<double, 1, 9, Eigen::RowMajor>::Zero ();
//	unsigned int point_count;
//	if (cloud.is_dense)
//	{
//		point_count = cloud.size ();
//		// For each point in the cloud
//		for (size_t i = 0; i < point_count; ++i)
//		{
//			accu [0] += cloud[i].x * cloud[i].x;
//			accu [1] += cloud[i].x * cloud[i].y;
//			accu [2] += cloud[i].x * cloud[i].z;
//			accu [3] += cloud[i].y * cloud[i].y;
//			accu [4] += cloud[i].y * cloud[i].z;
//			accu [5] += cloud[i].z * cloud[i].z;
//			accu [6] += cloud[i].x;
//			accu [7] += cloud[i].y;
//			accu [8] += cloud[i].z;
//		}
//	}
//	else
//	{
//		point_count = 0;
//		for (size_t i = 0; i < cloud.points.size (); ++i)
//		{
//			if (!isFinite (cloud[i]))
//				continue;
//
//			accu [0] += cloud[i].x * cloud[i].x;
//			accu [1] += cloud[i].x * cloud[i].y;
//			accu [2] += cloud[i].x * cloud[i].z;
//			accu [3] += cloud[i].y * cloud[i].y;
//			accu [4] += cloud[i].y * cloud[i].z;
//			accu [5] += cloud[i].z * cloud[i].z;
//			accu [6] += cloud[i].x;
//			accu [7] += cloud[i].y;
//			accu [8] += cloud[i].z;
//			++point_count;
//		}
//	}
//
//	if (point_count != 0)
//	{
//		accu /= static_cast<double> (point_count);
//		centroid.head<3> () = accu.tail<3> ();
//		centroid[3] = 0;
//		covariance_matrix.coeffRef (0) = accu [0] - accu [6] * accu [6];
//		covariance_matrix.coeffRef (1) = accu [1] - accu [6] * accu [7];
//		covariance_matrix.coeffRef (2) = accu [2] - accu [6] * accu [8];
//		covariance_matrix.coeffRef (4) = accu [3] - accu [7] * accu [7];
//		covariance_matrix.coeffRef (5) = accu [4] - accu [7] * accu [8];
//		covariance_matrix.coeffRef (8) = accu [5] - accu [8] * accu [8];
//		covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);
//		covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);
//		covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);
//	}
//	return (point_count);
//}
//
// unsigned int fi::PlaneModel::computeMeanAndCovarianceMatrix (const pcl::PointCloud<pcl::PointXYZ> &cloud,
//	const std::vector<int> &indices,
//	Eigen::Matrix3d &covariance_matrix,
//	Eigen::Vector4d &centroid)
//{
//	// create the buffer on the stack which is much faster than using cloud.points[indices[i]] and centroid as a buffer
//	Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<double, 1, 9, Eigen::RowMajor>::Zero ();
//	unsigned point_count;
//	if (cloud.is_dense)
//	{
//		point_count = indices.size ();
//		for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
//		{
//			//const PointT& point = cloud[*iIt];
//			accu [0] += cloud[*iIt].x * cloud[*iIt].x;
//			accu [1] += cloud[*iIt].x * cloud[*iIt].y;
//			accu [2] += cloud[*iIt].x * cloud[*iIt].z;
//			accu [3] += cloud[*iIt].y * cloud[*iIt].y;
//			accu [4] += cloud[*iIt].y * cloud[*iIt].z;
//			accu [5] += cloud[*iIt].z * cloud[*iIt].z;
//			accu [6] += cloud[*iIt].x;
//			accu [7] += cloud[*iIt].y;
//			accu [8] += cloud[*iIt].z;
//		}
//	}
//	else
//	{
//		point_count = 0;
//		for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
//		{
//			if (!isFinite (cloud[*iIt]))
//				continue;
//
//			++point_count;
//			accu [0] += cloud[*iIt].x * cloud[*iIt].x;
//			accu [1] += cloud[*iIt].x * cloud[*iIt].y;
//			accu [2] += cloud[*iIt].x * cloud[*iIt].z;
//			accu [3] += cloud[*iIt].y * cloud[*iIt].y; // 4
//			accu [4] += cloud[*iIt].y * cloud[*iIt].z; // 5
//			accu [5] += cloud[*iIt].z * cloud[*iIt].z; // 8
//			accu [6] += cloud[*iIt].x;
//			accu [7] += cloud[*iIt].y;
//			accu [8] += cloud[*iIt].z;
//		}
//	}
//
//	if (point_count != 0)
//	{
//		accu /= static_cast<double> (point_count);
//		Eigen::Vector3f vec = accu.tail<3> ();
//		centroid.head<3> () = vec;//= accu.tail<3> ();
//		centroid[3] = 0;
//		covariance_matrix.coeffRef (0) = accu [0] - accu [6] * accu [6];
//		covariance_matrix.coeffRef (1) = accu [1] - accu [6] * accu [7];
//		covariance_matrix.coeffRef (2) = accu [2] - accu [6] * accu [8];
//		covariance_matrix.coeffRef (4) = accu [3] - accu [7] * accu [7];
//		covariance_matrix.coeffRef (5) = accu [4] - accu [7] * accu [8];
//		covariance_matrix.coeffRef (8) = accu [5] - accu [8] * accu [8];
//		covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);
//		covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);
//		covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);
//	}
//	return (point_count);
//}
//