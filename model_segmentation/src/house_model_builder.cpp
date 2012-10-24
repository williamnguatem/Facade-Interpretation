#include "house_model_builder.h"

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/intersections.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/angles.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/common.h>
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>



#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/intersections.h>
#include <pcl/common/centroid.h>
#include <pcl/common/angles.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp> 

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp> 
#include <boost/tuple/tuple_io.hpp>

//demo only
#include <pcl/features/normal_3d.h>


#include <vtkQuad.h>


fi::HouseModelBuilder::HouseModelBuilder()
{

}


fi::HouseModelBuilder::HouseModelBuilder(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unibw_params_dir)
	:m_mshr_result_dir(mshr_result_dir), m_image_data_dir(image_data_dir), m_image_extension(image_extension), m_unibw_params_dir(unibw_params_dir)
{

}


//destructor
fi::HouseModelBuilder::~HouseModelBuilder()
{

}

void fi::HouseModelBuilder::initModel(const VPDetectionType &vp_detection_type)
{
	//parse input data
	MSHRFileIO reg_dat(m_mshr_result_dir, m_image_data_dir, m_image_extension, m_unibw_params_dir);
	reg_dat.get3DTo2DProjectionsUniBW();
	//MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unbw_params_dir);
	//std::vector <std::vector<Eigen::Vector2f> > pnt_projectionstion;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = reg_dat.getCloud();
	m_cloud = cloud;
	//std::vector <std::vector<Eigen::Vector2f> > image_points2D = reg_dat.getImagePoints();
	m_corresponding_images_filenames = reg_dat.getCorrespondingImageFileNames();
	m_mapping_table = reg_dat.getMappingTable();

	//cam calib
	CamCalibParser camera_calib(m_mshr_result_dir);
	Eigen::Matrix3f cam_calib;
	camera_calib.getCameraCalib(cam_calib);
	m_cam_calib = cam_calib;
	std::cout<<cam_calib<<std::endl;


	if (vp_detection_type == CLOUD)
	{
		Eigen::VectorXf fResults;
		//Extract Model Parameters from GUI Elements
		VPModelParamsPtr fModelParams(new fi::VPModelParams);
		fModelParams->fNumOfIterations = 2000;
		// collect Voxel grid params
		//	//ToDo: Check to see which controller is active before collecting voxel grid value
		fModelParams->fVoxGridSize(0) = 0.000; //doubleSpinBox_8->value();
		fModelParams->fVoxGridSize(1) = 0.000; //doubleSpinBox_9->value();
		fModelParams->fVoxGridSize(2) = 0.000; //doubleSpinBox_10->value();
		// collect Spatial partitioning params, default use KD-tree
		fModelParams->fRadiusSearched = 0.0f; //doubleSpinBox_3->value();//
		fModelParams->m_k = 10; //spinBox_5->value();
		//collect EPS
		fModelParams->fEPSInDegrees = 3;
		//collect percentage of the number of clouds to compute normals
		fModelParams->fPercentageOfNormals = 40;
		//collect the number of cross products
		fModelParams->fNumOfCrossProducts = 1000;
		//always validate results
		fModelParams->paramValidateResults = true;
		fModelParams->fNumberOfValidationRounds = 20;

		//Do the actual work here
		fi::VPDetectionCloud *vp = new fi::VPDetectionCloud;
		fi::VPDetectionContext vpContext(vp);
		vpContext.setInputCloud(m_cloud);
		vpContext.setModelParams(fModelParams);
		vpContext.validateResults(fModelParams->paramValidateResults);
		vpContext.ComputeVPDetection();
		vpContext.extractVPs(fResults);
		std::cout<<fResults<<std::endl;
		Eigen::VectorXf computed_cloud_vp;
		m_zenith_direction = fResults.tail<3>();

		delete vp;
	}
	else
	{
		VPDetectionWrapper test_vp_wrapper(m_corresponding_images_filenames, m_image_data_dir);
		std::vector<std::vector<Eigen::Vector2f> > sets_of_vanishing_point;
		test_vp_wrapper.collectVanishingPoints(sets_of_vanishing_point);
		Eigen::Vector3f final_robust_vp_x, final_robust_vp_y;
		test_vp_wrapper.validateVanishingPoint(sets_of_vanishing_point, cam_calib, final_robust_vp_x, final_robust_vp_y);

		m_zenith_direction = final_robust_vp_x;
		m_horizon_direction = final_robust_vp_y;
		//float angle_in_radiens = angleBetweenVectors(final_robust_vp_x, final_robust_vp_y);

		//m_horizon_direction = final_robust_vp_y;
		//m_zenith_direction = final_robust_vp_x;
		//float angle_value_degrees_cxy = pcl::rad2deg(angle_in_radiens);
		//std::cout <<" angle between vps: "<<angle_value_degrees_cxy<<std::endl;
	}
}


void fi::HouseModelBuilder::reconstructModel()
{
	//save cloud with normals too for visualizer
	///only for demo purposes normal computation
	////
	////
	////	   Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (m_cloud);
	n.setInputCloud (m_cloud);
	n.setSearchMethod (tree);
	n.setKSearch (10);
	n.compute (*normals);
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*m_cloud, *normals, *cloud_with_normals);

	//* cloud_with_normals = cloud + normals

	/*pcl::io::savePCDFile ("bun0-mls.pcd", *outPutf);*/
	pcl::PCDWriter writer;
	writer.write ("haus45_1_with_normals_k10.pcd", *cloud_with_normals, false);

	/* const unsigned int fTopPercent = 10;
	Eigen::Vector3f fVP(0,0,0);
	std::vector<int> fEdges;
	GetEdges(fTopPercent, fVP, fEdges);*/


	//visualize cloud
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer0(new pcl::visualization::PCLVisualizer("3D Viewer")) ;
	viewer0->setBackgroundColor (0, 0, 0);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fPerpendicularHanders (m_cloud, 255, 0, 0);
	std::string fCloudId = "Planes Perpendicular Cloud" + boost::lexical_cast<std::string>(1);
	viewer0->addPointCloud (m_cloud, fPerpendicularHanders, fCloudId );
	viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
	viewer0->addCoordinateSystem (1.0f);

	while (!viewer0->wasStopped ())
	{
		viewer0->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}



	//segment planes along the zenith
	std::cout <<"Reconstructing"<<std::endl;
	unsigned int num_iterations = 1000;
	fi::SegModelParamsPtr tmpModelParams(new fi::SegModelParams);
	tmpModelParams->num_of_iterations = num_iterations;
	tmpModelParams->min_num_inliers = 100;
	tmpModelParams->max_num_of_models = 4;//8;

	tmpModelParams->ransac_thresh = 0.03;
	Eigen::Vector3f tmp_voxel_grid(0.0f, 0.0f, 0.0f);
	tmpModelParams->voxel_grid_size = tmp_voxel_grid;

	//cp from image only
	//tmpModelParams->vanishing_point = vanishing_point_x;

	tmpModelParams->vanishing_point = m_zenith_direction;
	std::cout <<"zenith: "<< m_zenith_direction<<std::endl;


	//Check if voxel grid is needed otherwise do not down sample things!
	PlaneModel *p = new PlaneModel;
	ModelContext fContext(p);
	fContext.setInputCloud(m_cloud);
	fContext.setModelParams(tmpModelParams);
	fContext.ExecuteSegmentation();

	//	boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));
	//extract segmentation results
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_segments;
	std::vector<pcl::ModelCoefficients::Ptr> cloud_segment_coefficients;
	std::vector<fi::SegmentationResultPtr> out_segments;
	fContext.extractSegments(out_segments);

	delete p;

	for (unsigned int i = 0; i < out_segments.size(); i++)
	{
		pcl::PointIndices::Ptr _inlierIndices(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr _modelCoefficients(new pcl::ModelCoefficients());
		pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_segment(new pcl::PointCloud<pcl::PointXYZ>);
		fi::SegmentationResultPtr results(new fi::SegmentationResult);
		results = out_segments[i];
		_inlierIndices = results->get<1>();
		_modelCoefficients = results->get<0>();

		boost::scoped_ptr<pcl::ExtractIndices<pcl::PointXYZ> > extract(new pcl::ExtractIndices<pcl::PointXYZ>);
		extract->setInputCloud (m_cloud);
		extract->setIndices (_inlierIndices);
		extract->setNegative (false);
		extract->filter (*tmp_segment);
		std::cout << "PointCloud representing the planar component: " << tmp_segment->points.size () << " data points." << std::endl;

		//collect planes
		cloud_segment_coefficients.push_back(_modelCoefficients);
		cloud_segments.push_back(tmp_segment);
	}

	//collect intersection lines
	std::vector<Intersectors> test_polys;
	for (unsigned int i = 0; i < cloud_segments.size(); i++)
	{
		pcl::ModelCoefficients::Ptr plane_a(new pcl::ModelCoefficients);
		plane_a = cloud_segment_coefficients[i];

		for (unsigned int j = i + 1; j < cloud_segments.size(); j++)
		{
			pcl::ModelCoefficients::Ptr plane_b(new pcl::ModelCoefficients);
			plane_b = cloud_segment_coefficients[j];
			Eigen::VectorXf linesWi;
			linesWi.resize(6);
			bool do_intersect = getIntersectionLines(plane_a, plane_b, linesWi);

			if (do_intersect)
			{
				test_polys.push_back(boost::make_tuple( boost::ref(linesWi), boost::ref(i), boost::ref(j)));
				std::cout<<"intersecting planes"<<std::endl;
			}
		}
	}

	//ToDo:
	//project intersection lines to plane with normal = zenith and 
	//determine cycles from intersection lines

	// Estimate the XYZ centroid
	Eigen::Vector4f xyz_centroid1;
	pcl::compute3DCentroid (*cloud_segments[0], xyz_centroid1);
	Eigen::Vector4f xyz_centroid2;
	pcl::compute3DCentroid (*cloud_segments[1], xyz_centroid2);
	Eigen::Vector4f xyz_centroid;
	xyz_centroid = (xyz_centroid1 + xyz_centroid2)/2;

	//project centroid to lines along vp
	pcl::PointXYZ pPoint0;
	pcl::PointXYZ qPoint0;
	pPoint0.x = xyz_centroid(0);
	pPoint0.y = xyz_centroid(1);
	pPoint0.z = xyz_centroid(2);
	Intersectors a0 = test_polys.at(0);
	Eigen::VectorXf line_0 = a0.get<0>();
	projectPointOnLine(pPoint0, line_0, qPoint0);

	pcl::PointXYZ pPoint1;
	pcl::PointXYZ qPoint1;
	pPoint1.x = xyz_centroid(0);
	pPoint1.y = xyz_centroid(1);
	pPoint1.z = xyz_centroid(2);
	Intersectors a1 = test_polys.at(1);
	Eigen::VectorXf line_1 = a1.get<0>();
	projectPointOnLine(pPoint1, line_1, qPoint1);


	pcl::PointXYZ pPoint2;
	pcl::PointXYZ qPoint2;
	pPoint2.x = xyz_centroid(0);
	pPoint2.y = xyz_centroid(1);
	pPoint2.z = xyz_centroid(2);
	Intersectors a2 = test_polys.at(2);
	Eigen::VectorXf line_2 = a2.get<0>();
	projectPointOnLine(pPoint2, line_2, qPoint2);


	pcl::PointXYZ pPoint3;
	pcl::PointXYZ qPoint3;
	pPoint3.x = xyz_centroid(0);
	pPoint3.y = xyz_centroid(1);
	pPoint3.z = xyz_centroid(2);
	Intersectors a3 = test_polys.at(3);
	Eigen::VectorXf line_3 = a3.get<0>();
	projectPointOnLine(pPoint3, line_3, qPoint3);


	//collect outline segments mappings i.e plane, outlines indices
	std::vector<PlaneLines> plane_outlines;
	for (unsigned int i = 0; i < cloud_segment_coefficients.size(); i++)
	{
		std::vector<unsigned int> indeces_lines;
		for (unsigned int j = 0; j < test_polys.size(); j++)
		{
			Intersectors a_tmp = test_polys.at(j);
			Eigen::VectorXf line_0 = a_tmp.get<0>();
			Eigen::Vector3f pnt_on_line = line_0.head<3>();

			if(pointIsOnPlane(cloud_segment_coefficients[i], pnt_on_line))
			{
				indeces_lines.push_back(j);
				std::cout<<"indices: "<<j<<std::endl;
			}
		}
		plane_outlines.push_back(boost::make_tuple( i, boost::ref(indeces_lines)));
	}


	//extract quads and plot
	findQuads(m_cloud,
		out_segments, 
		test_polys, 
		plane_outlines, 
		m_corresponding_images_filenames,
		m_zenith_direction,
		m_horizon_direction,
		m_cam_calib,
		m_unibw_params_dir
		);
}


void fi::HouseModelBuilder::projectPointOnLine(const pcl::PointXYZ &pPoint, const Eigen::VectorXf &fOptimLinesCoefs, pcl::PointXYZ &qPoint)
{

	Eigen::Vector3f fPnt (pPoint.x, pPoint.y, pPoint.z);
	Eigen::Vector3f fLineCentroid (fOptimLinesCoefs[0], fOptimLinesCoefs[1], fOptimLinesCoefs[2]);
	Eigen::Vector3f fLineDir (fOptimLinesCoefs[3], fOptimLinesCoefs[4], fOptimLinesCoefs[5]);
	fLineDir = fLineDir.normalized();//create ray normal
	Eigen::Vector3f fRes1 = fLineCentroid - fPnt;

	float fDistance = fRes1.dot(fLineDir);// calculate dot
	Eigen::Vector3f fMult = fLineDir * fDistance;
	Eigen::Vector3f fRes2 = fLineCentroid + fMult;

	qPoint.x = fRes2(0);
	qPoint.y = fRes2(1);
	qPoint.z = fRes2(2);
}


void fi::HouseModelBuilder::findQuads(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
	const std::vector<fi::SegmentationResultPtr> &segments_results,  
	const	std::vector<Intersectors> &test_polys, 
	const std::vector<PlaneLines> &plane_outlines, 
	const std::vector<std::vector<std::string> > &corresponding_images_filenames,
	const Eigen::Vector3f &vanishing_point_x,
	const Eigen::Vector3f &vanishing_point_y,
	const Eigen::Matrix3f &cam_calib,
	const std::string &unibw_result_dir
	)
{
	// prepare to Extract the planar in liers from the input cloud
	boost::scoped_ptr<pcl::ExtractIndices<pcl::PointXYZ> > extract(new pcl::ExtractIndices<pcl::PointXYZ>);
	extract->setInputCloud (cloud);


	//visualize 
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer0(new pcl::visualization::PCLVisualizer("3D Viewer")) ;
	//viewer0->setBackgroundColor (0, 0, 0);

	unsigned int num_outlined_segments = plane_outlines.size();

	// Visualize the textured plane
	vtkSmartPointer<vtkRenderer> renderer =
		vtkSmartPointer<vtkRenderer>::New();

	//for (unsigned int i = 0; i < num_outlined_segments; i++)
	for (unsigned int i = 0; i < 1; i++)
	{
		//extract plane between two lines
		std::cout<<"processing outlined segment: "<<i<<std::endl;
		PlaneLines a = plane_outlines[i];
		unsigned int plane_of_interess_indx = a.get<0>();
		fi::SegmentationResultPtr seg_interess_res(new fi::SegmentationResult);
		seg_interess_res = segments_results[plane_of_interess_indx];
		pcl::PointIndices::Ptr interess_plane_model_indices = seg_interess_res->get<1>();
		pcl::ModelCoefficients::Ptr plane_model_coefficients = seg_interess_res->get<0>();

		std::vector<unsigned int> plane_of_interess_outlines_indx = a.get<1>();
		if ( plane_of_interess_outlines_indx.size() != 2)
		{
			PCL_ERROR(" more than two lines outlined plane!");
			return;
		}

		Intersectors ints_a = test_polys[plane_of_interess_outlines_indx[0]]; 
		Intersectors ints_b = test_polys[plane_of_interess_outlines_indx[1]];

		Eigen::VectorXf line_a = ints_a.get<0>();
		Eigen::VectorXf line_b = ints_b.get<0>();
		Eigen::Vector4f xyz_centroid;
		//pcl::compute3DCentroid(*cloud, *interess_plane_model_indices, xyz_centroid);
		pcl::compute3DCentroid(*cloud, xyz_centroid);

		//get top line perpendicular to both lines and plane normal
		Eigen::Vector3f plane_normal (plane_model_coefficients->values[0], plane_model_coefficients->values[1], plane_model_coefficients->values[2]);
		Eigen::Vector3f top_line_direction = plane_normal.cross(line_a.tail<3>()); 
		top_line_direction = top_line_direction.normalized();

		//project the centroid to the two lines
		pcl::PointXYZ pPoint0;
		pcl::PointXYZ lPoint0;
		pcl::PointXYZ rPoint0;
		pPoint0.x = xyz_centroid(0);
		pPoint0.y = xyz_centroid(1);
		pPoint0.z = xyz_centroid(2);
		projectPointOnLine(pPoint0, line_a, lPoint0);
		//projectPointOnLine(lPoint0, line_b, rPoint0);


		//define endings of the quad
		float fLineScale = 0.6f;
		pcl::PointXYZ fLineStart;
		fLineStart.x=lPoint0.x - 1.08*fLineScale*line_a(3);
		fLineStart.y=lPoint0.y - 1.08*fLineScale*line_a(4);
		fLineStart.z=lPoint0.z - 1.08*fLineScale*line_a(5);
		pcl::PointXYZ fLineEndPoint;

		fLineEndPoint.x = lPoint0.x + 0.08*fLineScale*line_a(3);
		fLineEndPoint.y = lPoint0.y + 0.08*fLineScale*line_a(4);
		fLineEndPoint.z = lPoint0.z + 0.08*fLineScale*line_a(5);

		pcl::PointXYZ fLineStart1;
		//fLineStart1.x=rPoint0.x - 0.02*fLineScale*line_b(3);
		//fLineStart1.y=rPoint0.y - 0.02*fLineScale*line_b(4);
		//fLineStart1.z=rPoint0.z - 0.02*fLineScale*line_b(5);
		pcl::PointXYZ fLineEndPoint1;

		//fLineEndPoint1.x = rPoint0.x + fLineScale*line_b(3);
		//fLineEndPoint1.y = rPoint0.y + fLineScale*line_b(4);
		//fLineEndPoint1.z = rPoint0.z + fLineScale*line_b(5);


		Eigen::Vector4f p_up, p_down;
		Eigen::VectorXf line_p;
		line_p.resize(6);
		line_p(0) = fLineStart.x;
		line_p(1) = fLineStart.y;
		line_p(2) = fLineStart.z;
		line_p.tail<3>() = top_line_direction;
		pcl::lineWithLineIntersection(line_p, line_b, p_up);
		fLineStart1.x = p_up(0);
		fLineStart1.y = p_up(1);
		fLineStart1.z = p_up(2);

		line_p(0) = fLineEndPoint.x;
		line_p(1) = fLineEndPoint.y;
		line_p(2) = fLineEndPoint.z;
		pcl::lineWithLineIntersection(line_p, line_b, p_down);
		fLineEndPoint1.x = p_down(0);
		fLineEndPoint1.y = p_down(1);
		fLineEndPoint1.z = p_down(2);

		//determine which images are best suited to project the cloud on
		pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		extract->setIndices (interess_plane_model_indices);
		extract->setNegative (false);
		// Write the planar inliers to disk
		extract->filter (*plane_cloud);

		//construct a kd tree to enable search for points around points of the centroid
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud (plane_cloud);

		pcl::PointXYZ searchPoint; //this should be the centroid of every quad found
		searchPoint.x = 1/4 * (fLineStart.x + fLineEndPoint.x + fLineStart1.x + fLineEndPoint1.x);
		searchPoint.y = 1/4 * (fLineStart.y + fLineEndPoint.y + fLineStart1.y + fLineEndPoint1.y);
		searchPoint.z = 1/4 * (fLineStart.z + fLineEndPoint.z + fLineStart1.z + fLineEndPoint1.z);

		// K nearest neighbor search
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);

		if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
		{
			for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
				std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
				<< " " << cloud->points[ pointIdxNKNSearch[i] ].y 
				<< " " << cloud->points[ pointIdxNKNSearch[i] ].z 
				<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
		}

		//get the position  of the point on the main cloud
		unsigned int img_at_inx = interess_plane_model_indices->indices[pointIdxNKNSearch[0]];

		std::cout<<"Index found: "<< img_at_inx<<std::endl;
		std::vector<std::string> images_concerned = corresponding_images_filenames[img_at_inx];
		for (unsigned int g = 0; g < images_concerned.size(); g++)
		{
			Eigen::Vector2f image_point_up, image_point_down, image_point_l, image_point_r;
			getProjectionPointOnImage(fLineStart, image_point_l,unibw_result_dir, images_concerned[g],cam_calib);
			getProjectionPointOnImage(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g], cam_calib);
			getProjectionPointOnImage(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g], cam_calib);
			getProjectionPointOnImage(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g], cam_calib);

			vtkSmartPointer<vtkActor> line_actor0 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor1 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor2 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor3 =
				vtkSmartPointer<vtkActor>::New();
			addLineToRenderer(fLineStart, fLineEndPoint, line_actor0);
			line_actor0->GetProperty()->SetLineWidth(4);
			line_actor0->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineStart, fLineStart1, line_actor1);
			line_actor1->GetProperty()->SetLineWidth(4);
			line_actor1->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineStart1, fLineEndPoint1, line_actor2);
			line_actor2->GetProperty()->SetLineWidth(4);
			line_actor2->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineEndPoint, fLineEndPoint1, line_actor3);
			line_actor3->GetProperty()->SetLineWidth(4);
			line_actor3->GetProperty()->SetColor(1.0, 0,0);
			//getProjectionPointOnImageDLR(fLineStart, image_point_l,unibw_result_dir, images_concerned[g] );
			//getProjectionPointOnImageDLR(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g]);
			//getProjectionPointOnImageDLR(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g]);
			//getProjectionPointOnImageDLR(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g]);

			cv::Mat image;
			image = cv::imread(images_concerned[g]);

			cv::Mat image2, image3;
			// the two images refer to the same data
			image.copyTo(image2); // a new copy is created

			cv::Point pt1, pt2, pt3, pt4;
			pt1.x = image_point_l(0);
			pt1.y = image_point_l(1);
			pt2.x = image_point_r(0);
			pt2.y = image_point_r(1);
			pt3.x = image_point_up(0);
			pt3.y = image_point_up(1);
			pt4.x = image_point_down(0);
			pt4.y = image_point_down(1);
			// draw a white line
			cv::line( image2, pt1, pt2, cv::Scalar(255), 4);
			cv::line( image2, pt3, pt4, cv::Scalar(255), 4);
			cv::line( image2, pt3, pt1, cv::Scalar(255), 4);
			cv::line( image2, pt4, pt2, cv::Scalar(255), 4);

			cv::circle(image2,pt1, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);

			cv::circle(image2,pt2, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			cv::circle(image2,pt3, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			cv::circle(image2,pt4, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			//get homography or perspective transform

			std::vector<cv::Point2f> src_vertices(4);
			src_vertices[0] = pt1;
			src_vertices[1] = pt3;
			src_vertices[2] = pt2;
			src_vertices[3] = pt4;

			//destination
			std::vector<cv::Point2f> dst_vertices(4);
			dst_vertices[0] = cv::Point(0, 0);
			dst_vertices[1] = cv::Point(image2.size().width-1, 0);
			dst_vertices[2] = cv::Point(0, image2.size().height-1);
			dst_vertices[3] = cv::Point(image2.size().width-1, image2.size().height-1);

			std::vector<cv::Point2f> src_quad_hull;
			std::vector<cv::Point2f> dst_quad_hull;
			cv::Mat image_hull_points;
			image_hull_points.resize(4);

			// Calculate convex hull of original points (which points positioned on the boundary)
			cv::convexHull(cv::Mat(src_vertices), src_quad_hull, false);
			cv::convexHull(cv::Mat(dst_vertices), dst_quad_hull, false);

			cv::Mat quad_homography = cv::getPerspectiveTransform(src_quad_hull, dst_quad_hull);

			cv::Mat rectified1(image2.size(), image2.type()); ;
			//cv::Size size(box.boundingRect().width, box.boundingRect().height);
			cv::Size img_size(image2.size().width, image2.size().height);
			/*cv::warpPerspective(image2, rotated, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);*/
			//cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
			cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
			cv::imwrite("rotated.jpg", rectified1);

			//boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));


			boost::filesystem::path tmp_path(images_concerned[g]);

			std::string result_name = "results_"+ boost::lexical_cast<std::string>(i)+ "_" +tmp_path.stem().string()+ ".jpg";
			std::cout<<"result_image_name: "<<result_name<<std::endl;
			//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

			cv::imwrite(result_name, image2);
			std::cout<< image_point_l<<std::endl;
			std::cout<< image_point_r <<std::endl;
			std::cout<< image_point_up<<std::endl;
			std::cout<< image_point_down<<std::endl;
			//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

			std::cout<<images_concerned[g]<<std::endl;


			// Read the image which will be the texture
			vtkSmartPointer<vtkJPEGReader> jPEGReader =
				vtkSmartPointer<vtkJPEGReader>::New();
			jPEGReader->SetFileName ( "rotated.jpg" );


			//reorder ending points for the opengl renderer
			pcl::PointCloud<pcl::PointXYZ>::Ptr quad_cloud(new pcl::PointCloud<pcl::PointXYZ>),
				ordered_hull_points(new pcl::PointCloud<pcl::PointXYZ>);
			quad_cloud->width = 4;
			quad_cloud->height = 1;
			quad_cloud->resize(quad_cloud->width * quad_cloud->height);
			quad_cloud->points[0] = fLineStart;
			quad_cloud->points[1] = fLineEndPoint;
			quad_cloud->points[2] = fLineStart1;
			quad_cloud->points[3] = fLineEndPoint1;
			getCloudHull(quad_cloud, ordered_hull_points);

			vtkSmartPointer<vtkActor> quad_actor = vtkSmartPointer<vtkActor>::New();
			addQuadActorToRenderer(ordered_hull_points, quad_actor);

			vtkSmartPointer<vtkActor> textured_quad_actor = vtkSmartPointer<vtkActor>::New();
			addTexturedQuadActorToRenderer(ordered_hull_points, "rotated.jpg",textured_quad_actor);

			vtkSmartPointer<vtkActor> cloud_actor =	vtkSmartPointer<vtkActor>::New();
			addCloudActorToRenderer(cloud, cloud_actor);
			cloud_actor->GetProperty()->SetColor(0.0,1.0, 0.0);
			cloud_actor->GetProperty()->SetPointSize(3);

			//// Visualize the textured plane
			//vtkSmartPointer<vtkRenderer> renderer =
			//	vtkSmartPointer<vtkRenderer>::New();
			//renderer->AddActor(texturedQuad);
			renderer->AddActor(textured_quad_actor);
			//renderer->AddActor(quad_actor);
			renderer->AddActor(cloud_actor);
			renderer->AddActor(line_actor0);
			renderer->AddActor(line_actor1);
			renderer->AddActor(line_actor2);
			renderer->AddActor(line_actor3);
			renderer->SetBackground(1,1,1); // Background color white
			renderer->ResetCamera();

			/*	vtkSmartPointer<vtkRenderWindow> renderWindow =
			vtkSmartPointer<vtkRenderWindow>::New();
			renderWindow->AddRenderer(renderer);

			vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
			vtkSmartPointer<vtkRenderWindowInteractor>::New();
			renderWindowInteractor->SetRenderWindow(renderWindow);

			renderWindow->Render();

			renderWindowInteractor->Start();*/


			//projectPointOnLine(fLineStart, line_b, fLineStart1);
			//projectPointOnLine(fLineEndPoint, line_b, fLineEndPoint1);

			//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fPerpendicularHanders (cloud, 255, 0, 0);
			//std::string fCloudId = "Planes Perpendicular Cloud" + boost::lexical_cast<std::string>(i);
			//viewer0->addPointCloud (cloud, fPerpendicularHanders, fCloudId );
			//viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
			//viewer0->addCoordinateSystem (1.0f);

			//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, "line 0"+i, 0); 
			//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart1, fLineEndPoint1, 0.0, 255.0, 0.0, "line 1"+i, 0);

			//extract a real quad using line -line-intersection, vp, and the normal to the plane



	for (unsigned int i = 3; i < 4; i++)
	{
		//extract plane between two lines
		std::cout<<"processing outlined segment: "<<i<<std::endl;
		PlaneLines a = plane_outlines[i];
		unsigned int plane_of_interess_indx = a.get<0>();
		fi::SegmentationResultPtr seg_interess_res(new fi::SegmentationResult);
		seg_interess_res = segments_results[plane_of_interess_indx];
		pcl::PointIndices::Ptr interess_plane_model_indices = seg_interess_res->get<1>();
		pcl::ModelCoefficients::Ptr plane_model_coefficients = seg_interess_res->get<0>();

		std::vector<unsigned int> plane_of_interess_outlines_indx = a.get<1>();
		if ( plane_of_interess_outlines_indx.size() != 2)
		{
			PCL_ERROR(" more than two lines outlined plane!");
			return;
		}

		Intersectors ints_a = test_polys[plane_of_interess_outlines_indx[0]]; 
		Intersectors ints_b = test_polys[plane_of_interess_outlines_indx[1]];

		Eigen::VectorXf line_a = ints_a.get<0>();
		Eigen::VectorXf line_b = ints_b.get<0>();
		Eigen::Vector4f xyz_centroid;
		//pcl::compute3DCentroid(*cloud, *interess_plane_model_indices, xyz_centroid);
		pcl::compute3DCentroid(*cloud, xyz_centroid);

		//get top line perpendicular to both lines and plane normal
		Eigen::Vector3f plane_normal (plane_model_coefficients->values[0], plane_model_coefficients->values[1], plane_model_coefficients->values[2]);
		Eigen::Vector3f top_line_direction = plane_normal.cross(line_a.tail<3>()); 
		top_line_direction = top_line_direction.normalized();

		//project the centroid to the two lines
		pcl::PointXYZ pPoint0;
		pcl::PointXYZ lPoint0;
		pcl::PointXYZ rPoint0;
		pPoint0.x = xyz_centroid(0);
		pPoint0.y = xyz_centroid(1);
		pPoint0.z = xyz_centroid(2);
		projectPointOnLine(pPoint0, line_a, lPoint0);
		//projectPointOnLine(lPoint0, line_b, rPoint0);


		//define endings of the quad
		float fLineScale = 0.6f;
		pcl::PointXYZ fLineStart;
		fLineStart.x=lPoint0.x - 0.08*fLineScale*line_a(3);
		fLineStart.y=lPoint0.y - 0.08*fLineScale*line_a(4);
		fLineStart.z=lPoint0.z - 0.08*fLineScale*line_a(5);
		pcl::PointXYZ fLineEndPoint;

		fLineEndPoint.x = lPoint0.x + 1.08*fLineScale*line_a(3);
		fLineEndPoint.y = lPoint0.y + 1.08*fLineScale*line_a(4);
		fLineEndPoint.z = lPoint0.z + 1.08*fLineScale*line_a(5);

		pcl::PointXYZ fLineStart1;
		//fLineStart1.x=rPoint0.x - 0.02*fLineScale*line_b(3);
		//fLineStart1.y=rPoint0.y - 0.02*fLineScale*line_b(4);
		//fLineStart1.z=rPoint0.z - 0.02*fLineScale*line_b(5);
		pcl::PointXYZ fLineEndPoint1;

		//fLineEndPoint1.x = rPoint0.x + fLineScale*line_b(3);
		//fLineEndPoint1.y = rPoint0.y + fLineScale*line_b(4);
		//fLineEndPoint1.z = rPoint0.z + fLineScale*line_b(5);


		Eigen::Vector4f p_up, p_down;
		Eigen::VectorXf line_p;
		line_p.resize(6);
		line_p(0) = fLineStart.x;
		line_p(1) = fLineStart.y;
		line_p(2) = fLineStart.z;
		line_p.tail<3>() = top_line_direction;
		pcl::lineWithLineIntersection(line_p, line_b, p_up);
		fLineStart1.x = p_up(0);
		fLineStart1.y = p_up(1);
		fLineStart1.z = p_up(2);

		line_p(0) = fLineEndPoint.x;
		line_p(1) = fLineEndPoint.y;
		line_p(2) = fLineEndPoint.z;
		pcl::lineWithLineIntersection(line_p, line_b, p_down);
		fLineEndPoint1.x = p_down(0);
		fLineEndPoint1.y = p_down(1);
		fLineEndPoint1.z = p_down(2);

		//determine which images are best suited to project the cloud on
		pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		extract->setIndices (interess_plane_model_indices);
		extract->setNegative (false);
		// Write the planar inliers to disk
		extract->filter (*plane_cloud);

		//construct a kd tree to enable search for points around points of the centroid
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud (plane_cloud);

		pcl::PointXYZ searchPoint; //this should be the centroid of every quad found
		searchPoint.x = 1/4 * (fLineStart.x + fLineEndPoint.x + fLineStart1.x + fLineEndPoint1.x);
		searchPoint.y = 1/4 * (fLineStart.y + fLineEndPoint.y + fLineStart1.y + fLineEndPoint1.y);
		searchPoint.z = 1/4 * (fLineStart.z + fLineEndPoint.z + fLineStart1.z + fLineEndPoint1.z);

		// K nearest neighbor search
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);

		if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
		{
			for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
				std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
				<< " " << cloud->points[ pointIdxNKNSearch[i] ].y 
				<< " " << cloud->points[ pointIdxNKNSearch[i] ].z 
				<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
		}

		//get the position  of the point on the main cloud
		unsigned int img_at_inx = interess_plane_model_indices->indices[pointIdxNKNSearch[0]];

		std::cout<<"Index found: "<< img_at_inx<<std::endl;
		std::vector<std::string> images_concerned = corresponding_images_filenames[img_at_inx];
		for (unsigned int g = 0; g < images_concerned.size(); g++)
		{
			Eigen::Vector2f image_point_up, image_point_down, image_point_l, image_point_r;
			getProjectionPointOnImage(fLineStart, image_point_l,unibw_result_dir, images_concerned[g],cam_calib);
			getProjectionPointOnImage(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g], cam_calib);
			getProjectionPointOnImage(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g], cam_calib);
			getProjectionPointOnImage(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g], cam_calib);

			vtkSmartPointer<vtkActor> line_actor0 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor1 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor2 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor3 =
				vtkSmartPointer<vtkActor>::New();
			addLineToRenderer(fLineStart, fLineEndPoint, line_actor0);
			line_actor0->GetProperty()->SetLineWidth(4);
			line_actor0->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineStart, fLineStart1, line_actor1);
			line_actor1->GetProperty()->SetLineWidth(4);
			line_actor1->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineStart1, fLineEndPoint1, line_actor2);
			line_actor2->GetProperty()->SetLineWidth(4);
			line_actor2->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineEndPoint, fLineEndPoint1, line_actor3);
			line_actor3->GetProperty()->SetLineWidth(4);
			line_actor3->GetProperty()->SetColor(1.0, 0,0);
			//getProjectionPointOnImageDLR(fLineStart, image_point_l,unibw_result_dir, images_concerned[g] );
			//getProjectionPointOnImageDLR(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g]);
			//getProjectionPointOnImageDLR(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g]);
			//getProjectionPointOnImageDLR(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g]);

			cv::Mat image;
			image = cv::imread(images_concerned[g]);

			cv::Mat image2, image3;
			// the two images refer to the same data
			image.copyTo(image2); // a new copy is created

			cv::Point pt1, pt2, pt3, pt4;
			pt1.x = image_point_l(0);
			pt1.y = image_point_l(1);
			pt2.x = image_point_r(0);
			pt2.y = image_point_r(1);
			pt3.x = image_point_up(0);
			pt3.y = image_point_up(1);
			pt4.x = image_point_down(0);
			pt4.y = image_point_down(1);
			// draw a white line
			cv::line( image2, pt1, pt2, cv::Scalar(255), 4);
			cv::line( image2, pt3, pt4, cv::Scalar(255), 4);
			cv::line( image2, pt3, pt1, cv::Scalar(255), 4);
			cv::line( image2, pt4, pt2, cv::Scalar(255), 4);

			cv::circle(image2,pt1, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);

			cv::circle(image2,pt2, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			cv::circle(image2,pt3, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			cv::circle(image2,pt4, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			//get homography or perspective transform

			std::vector<cv::Point2f> src_vertices(4);
			src_vertices[0] = pt1;
			src_vertices[1] = pt3;
			src_vertices[2] = pt2;
			src_vertices[3] = pt4;

			//destination
			std::vector<cv::Point2f> dst_vertices(4);
			dst_vertices[0] = cv::Point(0, 0);
			dst_vertices[1] = cv::Point(image2.size().width-1, 0);
			dst_vertices[2] = cv::Point(0, image2.size().height-1);
			dst_vertices[3] = cv::Point(image2.size().width-1, image2.size().height-1);

			std::vector<cv::Point2f> src_quad_hull;
			std::vector<cv::Point2f> dst_quad_hull;
			cv::Mat image_hull_points;
			image_hull_points.resize(4);

			// Calculate convex hull of original points (which points positioned on the boundary)
			cv::convexHull(cv::Mat(src_vertices), src_quad_hull, false);
			cv::convexHull(cv::Mat(dst_vertices), dst_quad_hull, false);

			cv::Mat quad_homography = cv::getPerspectiveTransform(src_quad_hull, dst_quad_hull);

			cv::Mat rectified1(image2.size(), image2.type()); ;
			//cv::Size size(box.boundingRect().width, box.boundingRect().height);
			cv::Size img_size(image2.size().width, image2.size().height);
			/*cv::warpPerspective(image2, rotated, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);*/
			//cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
			cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
			cv::imwrite("rotated2.jpg", rectified1);

			//boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));


			boost::filesystem::path tmp_path(images_concerned[g]);

			std::string result_name = "results_"+ boost::lexical_cast<std::string>(i)+ "_" +tmp_path.stem().string()+ ".jpg";
			std::cout<<"result_image_name: "<<result_name<<std::endl;
			//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

			cv::imwrite(result_name, image2);
			std::cout<< image_point_l<<std::endl;
			std::cout<< image_point_r <<std::endl;
			std::cout<< image_point_up<<std::endl;
			std::cout<< image_point_down<<std::endl;
			//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

			std::cout<<images_concerned[g]<<std::endl;


			// Read the image which will be the texture
			vtkSmartPointer<vtkJPEGReader> jPEGReader =
				vtkSmartPointer<vtkJPEGReader>::New();
			jPEGReader->SetFileName ( "rotated2.jpg" );


			//reorder ending points for the opengl renderer
			pcl::PointCloud<pcl::PointXYZ>::Ptr quad_cloud(new pcl::PointCloud<pcl::PointXYZ>),
				ordered_hull_points(new pcl::PointCloud<pcl::PointXYZ>);
			quad_cloud->width = 4;
			quad_cloud->height = 1;
			quad_cloud->resize(quad_cloud->width * quad_cloud->height);
			quad_cloud->points[0] = fLineStart;
			quad_cloud->points[1] = fLineEndPoint;
			quad_cloud->points[2] = fLineStart1;
			quad_cloud->points[3] = fLineEndPoint1;
			getCloudHull(quad_cloud, ordered_hull_points);

			vtkSmartPointer<vtkActor> quad_actor = vtkSmartPointer<vtkActor>::New();
			addQuadActorToRenderer(ordered_hull_points, quad_actor);

			vtkSmartPointer<vtkActor> textured_quad_actor = vtkSmartPointer<vtkActor>::New();
			addTexturedQuadActorToRendererR(ordered_hull_points, "rotated2.jpg",textured_quad_actor);

			vtkSmartPointer<vtkActor> cloud_actor =	vtkSmartPointer<vtkActor>::New();
			addCloudActorToRenderer(cloud, cloud_actor);
			cloud_actor->GetProperty()->SetColor(0.0,1.0, 0.0);
			cloud_actor->GetProperty()->SetPointSize(3);

			//// Visualize the textured plane
			//vtkSmartPointer<vtkRenderer> renderer =
			//	vtkSmartPointer<vtkRenderer>::New();
			//renderer->AddActor(texturedQuad);
			renderer->AddActor(textured_quad_actor);
			//renderer->AddActor(quad_actor);
			renderer->AddActor(cloud_actor);
			renderer->AddActor(line_actor0);
			renderer->AddActor(line_actor1);
			renderer->AddActor(line_actor2);
			renderer->AddActor(line_actor3);
			renderer->SetBackground(1,1,1); // Background color white
			renderer->ResetCamera();

			/*	vtkSmartPointer<vtkRenderWindow> renderWindow =
			vtkSmartPointer<vtkRenderWindow>::New();
			renderWindow->AddRenderer(renderer);

			vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
			vtkSmartPointer<vtkRenderWindowInteractor>::New();
			renderWindowInteractor->SetRenderWindow(renderWindow);

			renderWindow->Render();

			renderWindowInteractor->Start();*/


			//projectPointOnLine(fLineStart, line_b, fLineStart1);
			//projectPointOnLine(fLineEndPoint, line_b, fLineEndPoint1);

			//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fPerpendicularHanders (cloud, 255, 0, 0);
			//std::string fCloudId = "Planes Perpendicular Cloud" + boost::lexical_cast<std::string>(i);
			//viewer0->addPointCloud (cloud, fPerpendicularHanders, fCloudId );
			//viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
			//viewer0->addCoordinateSystem (1.0f);

			//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, "line 0"+i, 0); 
			//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart1, fLineEndPoint1, 0.0, 255.0, 0.0, "line 1"+i, 0);

			//extract a real quad using line -line-intersection, vp, and the normal to the plane



			for (unsigned int i = 2; i < 3; i++)
	{
		//extract plane between two lines
		std::cout<<"processing outlined segment: "<<i<<std::endl;
		PlaneLines a = plane_outlines[i];
		unsigned int plane_of_interess_indx = a.get<0>();
		fi::SegmentationResultPtr seg_interess_res(new fi::SegmentationResult);
		seg_interess_res = segments_results[plane_of_interess_indx];
		pcl::PointIndices::Ptr interess_plane_model_indices = seg_interess_res->get<1>();
		pcl::ModelCoefficients::Ptr plane_model_coefficients = seg_interess_res->get<0>();

		std::vector<unsigned int> plane_of_interess_outlines_indx = a.get<1>();
		if ( plane_of_interess_outlines_indx.size() != 2)
		{
			PCL_ERROR(" more than two lines outlined plane!");
			return;
		}

		Intersectors ints_a = test_polys[plane_of_interess_outlines_indx[0]]; 
		Intersectors ints_b = test_polys[plane_of_interess_outlines_indx[1]];

		Eigen::VectorXf line_a = ints_a.get<0>();
		Eigen::VectorXf line_b = ints_b.get<0>();
		Eigen::Vector4f xyz_centroid;
		//pcl::compute3DCentroid(*cloud, *interess_plane_model_indices, xyz_centroid);
		pcl::compute3DCentroid(*cloud, xyz_centroid);

		//get top line perpendicular to both lines and plane normal
		Eigen::Vector3f plane_normal (plane_model_coefficients->values[0], plane_model_coefficients->values[1], plane_model_coefficients->values[2]);
		Eigen::Vector3f top_line_direction = plane_normal.cross(line_a.tail<3>()); 
		top_line_direction = top_line_direction.normalized();

		//project the centroid to the two lines
		pcl::PointXYZ pPoint0;
		pcl::PointXYZ lPoint0;
		pcl::PointXYZ rPoint0;
		pPoint0.x = xyz_centroid(0);
		pPoint0.y = xyz_centroid(1);
		pPoint0.z = xyz_centroid(2);
		projectPointOnLine(pPoint0, line_a, lPoint0);
		//projectPointOnLine(lPoint0, line_b, rPoint0);


		//define endings of the quad
		float fLineScale = 0.6f;
		pcl::PointXYZ fLineStart;
		fLineStart.x=lPoint0.x - 1.08*fLineScale*line_a(3);
		fLineStart.y=lPoint0.y - 1.08*fLineScale*line_a(4);
		fLineStart.z=lPoint0.z - 1.08*fLineScale*line_a(5);
		pcl::PointXYZ fLineEndPoint;

		fLineEndPoint.x = lPoint0.x + 0.08*fLineScale*line_a(3);
		fLineEndPoint.y = lPoint0.y + 0.08*fLineScale*line_a(4);
		fLineEndPoint.z = lPoint0.z + 0.08*fLineScale*line_a(5);

		pcl::PointXYZ fLineStart1;
		//fLineStart1.x=rPoint0.x - 0.02*fLineScale*line_b(3);
		//fLineStart1.y=rPoint0.y - 0.02*fLineScale*line_b(4);
		//fLineStart1.z=rPoint0.z - 0.02*fLineScale*line_b(5);
		pcl::PointXYZ fLineEndPoint1;

		//fLineEndPoint1.x = rPoint0.x + fLineScale*line_b(3);
		//fLineEndPoint1.y = rPoint0.y + fLineScale*line_b(4);
		//fLineEndPoint1.z = rPoint0.z + fLineScale*line_b(5);


		Eigen::Vector4f p_up, p_down;
		Eigen::VectorXf line_p;
		line_p.resize(6);
		line_p(0) = fLineStart.x;
		line_p(1) = fLineStart.y;
		line_p(2) = fLineStart.z;
		line_p.tail<3>() = top_line_direction;
		pcl::lineWithLineIntersection(line_p, line_b, p_up);
		fLineStart1.x = p_up(0);
		fLineStart1.y = p_up(1);
		fLineStart1.z = p_up(2);

		line_p(0) = fLineEndPoint.x;
		line_p(1) = fLineEndPoint.y;
		line_p(2) = fLineEndPoint.z;
		pcl::lineWithLineIntersection(line_p, line_b, p_down);
		fLineEndPoint1.x = p_down(0);
		fLineEndPoint1.y = p_down(1);
		fLineEndPoint1.z = p_down(2);

		//determine which images are best suited to project the cloud on
		pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		extract->setIndices (interess_plane_model_indices);
		extract->setNegative (false);
		// Write the planar inliers to disk
		extract->filter (*plane_cloud);

		//construct a kd tree to enable search for points around points of the centroid
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud (plane_cloud);

		pcl::PointXYZ searchPoint; //this should be the centroid of every quad found
		searchPoint.x = 1/4 * (fLineStart.x + fLineEndPoint.x + fLineStart1.x + fLineEndPoint1.x);
		searchPoint.y = 1/4 * (fLineStart.y + fLineEndPoint.y + fLineStart1.y + fLineEndPoint1.y);
		searchPoint.z = 1/4 * (fLineStart.z + fLineEndPoint.z + fLineStart1.z + fLineEndPoint1.z);

		// K nearest neighbor search
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);

		if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
		{
			for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
				std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
				<< " " << cloud->points[ pointIdxNKNSearch[i] ].y 
				<< " " << cloud->points[ pointIdxNKNSearch[i] ].z 
				<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
		}

		//get the position  of the point on the main cloud
		unsigned int img_at_inx = interess_plane_model_indices->indices[pointIdxNKNSearch[0]];

		std::cout<<"Index found: "<< img_at_inx<<std::endl;
		std::vector<std::string> images_concerned = corresponding_images_filenames[img_at_inx];
		for (unsigned int g = 0; g < images_concerned.size(); g++)
		{
			Eigen::Vector2f image_point_up, image_point_down, image_point_l, image_point_r;
			getProjectionPointOnImage(fLineStart, image_point_l,unibw_result_dir, images_concerned[g],cam_calib);
			getProjectionPointOnImage(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g], cam_calib);
			getProjectionPointOnImage(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g], cam_calib);
			getProjectionPointOnImage(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g], cam_calib);

			vtkSmartPointer<vtkActor> line_actor0 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor1 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor2 =
				vtkSmartPointer<vtkActor>::New();
			vtkSmartPointer<vtkActor> line_actor3 =
				vtkSmartPointer<vtkActor>::New();
			addLineToRenderer(fLineStart, fLineEndPoint, line_actor0);
			line_actor0->GetProperty()->SetLineWidth(4);
			line_actor0->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineStart, fLineStart1, line_actor1);
			line_actor1->GetProperty()->SetLineWidth(4);
			line_actor1->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineStart1, fLineEndPoint1, line_actor2);
			line_actor2->GetProperty()->SetLineWidth(4);
			line_actor2->GetProperty()->SetColor(1.0, 0,0);
			addLineToRenderer(fLineEndPoint, fLineEndPoint1, line_actor3);
			line_actor3->GetProperty()->SetLineWidth(4);
			line_actor3->GetProperty()->SetColor(1.0, 0,0);
			//getProjectionPointOnImageDLR(fLineStart, image_point_l,unibw_result_dir, images_concerned[g] );
			//getProjectionPointOnImageDLR(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g]);
			//getProjectionPointOnImageDLR(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g]);
			//getProjectionPointOnImageDLR(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g]);

			cv::Mat image;
			image = cv::imread(images_concerned[g]);

			cv::Mat image2, image3;
			// the two images refer to the same data
			image.copyTo(image2); // a new copy is created

			cv::Point pt1, pt2, pt3, pt4;
			pt1.x = image_point_l(0);
			pt1.y = image_point_l(1);
			pt2.x = image_point_r(0);
			pt2.y = image_point_r(1);
			pt3.x = image_point_up(0);
			pt3.y = image_point_up(1);
			pt4.x = image_point_down(0);
			pt4.y = image_point_down(1);
			// draw a white line
			cv::line( image2, pt1, pt2, cv::Scalar(255), 4);
			cv::line( image2, pt3, pt4, cv::Scalar(255), 4);
			cv::line( image2, pt3, pt1, cv::Scalar(255), 4);
			cv::line( image2, pt4, pt2, cv::Scalar(255), 4);

			cv::circle(image2,pt1, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);

			cv::circle(image2,pt2, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			cv::circle(image2,pt3, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			cv::circle(image2,pt4, // circle centre
				8, // circle radius
				cv::Scalar(255), // color
				6);


			//get homography or perspective transform

			std::vector<cv::Point2f> src_vertices(4);
			src_vertices[0] = pt1;
			src_vertices[1] = pt3;
			src_vertices[2] = pt2;
			src_vertices[3] = pt4;

			//destination
			std::vector<cv::Point2f> dst_vertices(4);
			dst_vertices[0] = cv::Point(0, 0);
			dst_vertices[1] = cv::Point(image2.size().width-1, 0);
			dst_vertices[2] = cv::Point(0, image2.size().height-1);
			dst_vertices[3] = cv::Point(image2.size().width-1, image2.size().height-1);

			std::vector<cv::Point2f> src_quad_hull;
			std::vector<cv::Point2f> dst_quad_hull;
			cv::Mat image_hull_points;
			image_hull_points.resize(4);

			// Calculate convex hull of original points (which points positioned on the boundary)
			cv::convexHull(cv::Mat(src_vertices), src_quad_hull, false);
			cv::convexHull(cv::Mat(dst_vertices), dst_quad_hull, false);

			cv::Mat quad_homography = cv::getPerspectiveTransform(src_quad_hull, dst_quad_hull);

			cv::Mat rectified1(image2.size(), image2.type()); ;
			//cv::Size size(box.boundingRect().width, box.boundingRect().height);
			cv::Size img_size(image2.size().width, image2.size().height);
			/*cv::warpPerspective(image2, rotated, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);*/
			//cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
			cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
			cv::imwrite("rotated3.jpg", rectified1);

			//boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));


			boost::filesystem::path tmp_path(images_concerned[g]);

			std::string result_name = "results_"+ boost::lexical_cast<std::string>(i)+ "_" +tmp_path.stem().string()+ ".jpg";
			std::cout<<"result_image_name: "<<result_name<<std::endl;
			//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

			cv::imwrite(result_name, image2);
			std::cout<< image_point_l<<std::endl;
			std::cout<< image_point_r <<std::endl;
			std::cout<< image_point_up<<std::endl;
			std::cout<< image_point_down<<std::endl;
			//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

			std::cout<<images_concerned[g]<<std::endl;


			// Read the image which will be the texture
			vtkSmartPointer<vtkJPEGReader> jPEGReader =
				vtkSmartPointer<vtkJPEGReader>::New();
			jPEGReader->SetFileName ( "rotated3.jpg" );


			//reorder ending points for the opengl renderer
			pcl::PointCloud<pcl::PointXYZ>::Ptr quad_cloud(new pcl::PointCloud<pcl::PointXYZ>),
				ordered_hull_points(new pcl::PointCloud<pcl::PointXYZ>);
			quad_cloud->width = 4;
			quad_cloud->height = 1;
			quad_cloud->resize(quad_cloud->width * quad_cloud->height);
			quad_cloud->points[0] = fLineStart;
			quad_cloud->points[1] = fLineEndPoint;
			quad_cloud->points[2] = fLineStart1;
			quad_cloud->points[3] = fLineEndPoint1;
			getCloudHull(quad_cloud, ordered_hull_points);

			vtkSmartPointer<vtkActor> quad_actor = vtkSmartPointer<vtkActor>::New();
			addQuadActorToRenderer(ordered_hull_points, quad_actor);

			vtkSmartPointer<vtkActor> textured_quad_actor = vtkSmartPointer<vtkActor>::New();
			addTexturedQuadActorToRendererRR(ordered_hull_points, "rotated3.jpg",textured_quad_actor);

			vtkSmartPointer<vtkActor> cloud_actor =	vtkSmartPointer<vtkActor>::New();
			addCloudActorToRenderer(cloud, cloud_actor);
			cloud_actor->GetProperty()->SetColor(0.0,1.0, 0.0);
			cloud_actor->GetProperty()->SetPointSize(3);

			//// Visualize the textured plane
			//vtkSmartPointer<vtkRenderer> renderer =
			//	vtkSmartPointer<vtkRenderer>::New();
			//renderer->AddActor(texturedQuad);
			renderer->AddActor(textured_quad_actor);
			//renderer->AddActor(quad_actor);
			renderer->AddActor(cloud_actor);
			renderer->AddActor(line_actor0);
			renderer->AddActor(line_actor1);
			renderer->AddActor(line_actor2);
			renderer->AddActor(line_actor3);
			renderer->SetBackground(1,1,1); // Background color white
			renderer->ResetCamera();

			/*	vtkSmartPointer<vtkRenderWindow> renderWindow =
			vtkSmartPointer<vtkRenderWindow>::New();
			renderWindow->AddRenderer(renderer);

			vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
			vtkSmartPointer<vtkRenderWindowInteractor>::New();
			renderWindowInteractor->SetRenderWindow(renderWindow);

			renderWindow->Render();

			renderWindowInteractor->Start();*/


			//projectPointOnLine(fLineStart, line_b, fLineStart1);
			//projectPointOnLine(fLineEndPoint, line_b, fLineEndPoint1);

			//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fPerpendicularHanders (cloud, 255, 0, 0);
			//std::string fCloudId = "Planes Perpendicular Cloud" + boost::lexical_cast<std::string>(i);
			//viewer0->addPointCloud (cloud, fPerpendicularHanders, fCloudId );
			//viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
			//viewer0->addCoordinateSystem (1.0f);

			//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, "line 0"+i, 0); 
			//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart1, fLineEndPoint1, 0.0, 255.0, 0.0, "line 1"+i, 0);

			//extract a real quad using line -line-intersection, vp, and the normal to the plane



			for (unsigned int i = 1; i < 2; i++)
			{
				//extract plane between two lines
				std::cout<<"processing outlined segment: "<<i<<std::endl;
				PlaneLines a = plane_outlines[i];
				unsigned int plane_of_interess_indx = a.get<0>();
				fi::SegmentationResultPtr seg_interess_res(new fi::SegmentationResult);
				seg_interess_res = segments_results[plane_of_interess_indx];
				pcl::PointIndices::Ptr interess_plane_model_indices = seg_interess_res->get<1>();
				pcl::ModelCoefficients::Ptr plane_model_coefficients = seg_interess_res->get<0>();

				std::vector<unsigned int> plane_of_interess_outlines_indx = a.get<1>();
				if ( plane_of_interess_outlines_indx.size() != 2)
				{
					PCL_ERROR(" more than two lines outlined plane!");
					return;
				}

				Intersectors ints_a = test_polys[plane_of_interess_outlines_indx[0]]; 
				Intersectors ints_b = test_polys[plane_of_interess_outlines_indx[1]];

				Eigen::VectorXf line_a = ints_a.get<0>();
				Eigen::VectorXf line_b = ints_b.get<0>();
				Eigen::Vector4f xyz_centroid;
				//pcl::compute3DCentroid(*cloud, *interess_plane_model_indices, xyz_centroid);
				pcl::compute3DCentroid(*cloud, xyz_centroid);

				//get top line perpendicular to both lines and plane normal
				Eigen::Vector3f plane_normal (plane_model_coefficients->values[0], plane_model_coefficients->values[1], plane_model_coefficients->values[2]);
				Eigen::Vector3f top_line_direction = plane_normal.cross(line_a.tail<3>()); 
				top_line_direction = top_line_direction.normalized();

				//project the centroid to the two lines
				pcl::PointXYZ pPoint0;
				pcl::PointXYZ lPoint0;
				pcl::PointXYZ rPoint0;
				pPoint0.x = xyz_centroid(0);
				pPoint0.y = xyz_centroid(1);
				pPoint0.z = xyz_centroid(2);
				projectPointOnLine(pPoint0, line_a, lPoint0);
				//projectPointOnLine(lPoint0, line_b, rPoint0);


				//define endings of the quad
				float fLineScale = 0.6f;
				pcl::PointXYZ fLineStart;
				fLineStart.x=lPoint0.x - 1.08*fLineScale*line_a(3);
				fLineStart.y=lPoint0.y - 1.08*fLineScale*line_a(4);
				fLineStart.z=lPoint0.z - 1.08*fLineScale*line_a(5);
				pcl::PointXYZ fLineEndPoint;

				fLineEndPoint.x = lPoint0.x + 0.09*fLineScale*line_a(3);
				fLineEndPoint.y = lPoint0.y + 0.09*fLineScale*line_a(4);
				fLineEndPoint.z = lPoint0.z + 0.09*fLineScale*line_a(5);

				pcl::PointXYZ fLineStart1;
				//fLineStart1.x=rPoint0.x - 0.02*fLineScale*line_b(3);
				//fLineStart1.y=rPoint0.y - 0.02*fLineScale*line_b(4);
				//fLineStart1.z=rPoint0.z - 0.02*fLineScale*line_b(5);
				pcl::PointXYZ fLineEndPoint1;

				//fLineEndPoint1.x = rPoint0.x + fLineScale*line_b(3);
				//fLineEndPoint1.y = rPoint0.y + fLineScale*line_b(4);
				//fLineEndPoint1.z = rPoint0.z + fLineScale*line_b(5);


				Eigen::Vector4f p_up, p_down;
				Eigen::VectorXf line_p;
				line_p.resize(6);
				line_p(0) = fLineStart.x;
				line_p(1) = fLineStart.y;
				line_p(2) = fLineStart.z;
				line_p.tail<3>() = top_line_direction;
				pcl::lineWithLineIntersection(line_p, line_b, p_up);
				fLineStart1.x = p_up(0);
				fLineStart1.y = p_up(1);
				fLineStart1.z = p_up(2);

				line_p(0) = fLineEndPoint.x;
				line_p(1) = fLineEndPoint.y;
				line_p(2) = fLineEndPoint.z;
				pcl::lineWithLineIntersection(line_p, line_b, p_down);
				fLineEndPoint1.x = p_down(0);
				fLineEndPoint1.y = p_down(1);
				fLineEndPoint1.z = p_down(2);

				//determine which images are best suited to project the cloud on
				pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud (new pcl::PointCloud<pcl::PointXYZ>);
				extract->setIndices (interess_plane_model_indices);
				extract->setNegative (false);
				// Write the planar inliers to disk
				extract->filter (*plane_cloud);

				//construct a kd tree to enable search for points around points of the centroid
				pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
				kdtree.setInputCloud (plane_cloud);

				pcl::PointXYZ searchPoint; //this should be the centroid of every quad found
				searchPoint.x = 1/4 * (fLineStart.x + fLineEndPoint.x + fLineStart1.x + fLineEndPoint1.x);
				searchPoint.y = 1/4 * (fLineStart.y + fLineEndPoint.y + fLineStart1.y + fLineEndPoint1.y);
				searchPoint.z = 1/4 * (fLineStart.z + fLineEndPoint.z + fLineStart1.z + fLineEndPoint1.z);

				// K nearest neighbor search
				int K = 1;
				std::vector<int> pointIdxNKNSearch(K);
				std::vector<float> pointNKNSquaredDistance(K);

				if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
				{
					for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
						std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
						<< " " << cloud->points[ pointIdxNKNSearch[i] ].y 
						<< " " << cloud->points[ pointIdxNKNSearch[i] ].z 
						<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
				}

				//get the position  of the point on the main cloud
				unsigned int img_at_inx = interess_plane_model_indices->indices[pointIdxNKNSearch[0]];

				std::cout<<"Index found: "<< img_at_inx<<std::endl;
				std::vector<std::string> images_concerned = corresponding_images_filenames[img_at_inx];
				for (unsigned int g = 0; g < images_concerned.size(); g++)
				{
					Eigen::Vector2f image_point_up, image_point_down, image_point_l, image_point_r;
					getProjectionPointOnImage(fLineStart, image_point_l,unibw_result_dir, images_concerned[g],cam_calib);
					getProjectionPointOnImage(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g], cam_calib);
					getProjectionPointOnImage(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g], cam_calib);
					getProjectionPointOnImage(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g], cam_calib);

					vtkSmartPointer<vtkActor> line_actor0 =
						vtkSmartPointer<vtkActor>::New();
					vtkSmartPointer<vtkActor> line_actor1 =
						vtkSmartPointer<vtkActor>::New();
					vtkSmartPointer<vtkActor> line_actor2 =
						vtkSmartPointer<vtkActor>::New();
					vtkSmartPointer<vtkActor> line_actor3 =
						vtkSmartPointer<vtkActor>::New();
					addLineToRenderer(fLineStart, fLineEndPoint, line_actor0);
					line_actor0->GetProperty()->SetLineWidth(4);
					line_actor0->GetProperty()->SetColor(1.0, 0,0);
					addLineToRenderer(fLineStart, fLineStart1, line_actor1);
					line_actor1->GetProperty()->SetLineWidth(4);
					line_actor1->GetProperty()->SetColor(1.0, 0,0);
					addLineToRenderer(fLineStart1, fLineEndPoint1, line_actor2);
					line_actor2->GetProperty()->SetLineWidth(4);
					line_actor2->GetProperty()->SetColor(1.0, 0,0);
					addLineToRenderer(fLineEndPoint, fLineEndPoint1, line_actor3);
					line_actor3->GetProperty()->SetLineWidth(4);
					line_actor3->GetProperty()->SetColor(1.0, 0,0);
					//getProjectionPointOnImageDLR(fLineStart, image_point_l,unibw_result_dir, images_concerned[g] );
					//getProjectionPointOnImageDLR(fLineEndPoint, image_point_r,unibw_result_dir, images_concerned[g]);
					//getProjectionPointOnImageDLR(fLineStart1, image_point_up, unibw_result_dir, images_concerned[g]);
					//getProjectionPointOnImageDLR(fLineEndPoint1, image_point_down, unibw_result_dir, images_concerned[g]);

					cv::Mat image;
					image = cv::imread(images_concerned[g]);

					cv::Mat image2, image3;
					// the two images refer to the same data
					image.copyTo(image2); // a new copy is created

					cv::Point pt1, pt2, pt3, pt4;
					pt1.x = image_point_l(0);
					pt1.y = image_point_l(1);
					pt2.x = image_point_r(0);
					pt2.y = image_point_r(1);
					pt3.x = image_point_up(0);
					pt3.y = image_point_up(1);
					pt4.x = image_point_down(0);
					pt4.y = image_point_down(1);
					// draw a white line
					cv::line( image2, pt1, pt2, cv::Scalar(255), 4);
					cv::line( image2, pt3, pt4, cv::Scalar(255), 4);
					cv::line( image2, pt3, pt1, cv::Scalar(255), 4);
					cv::line( image2, pt4, pt2, cv::Scalar(255), 4);

					cv::circle(image2,pt1, // circle centre
						8, // circle radius
						cv::Scalar(255), // color
						6);

					cv::circle(image2,pt2, // circle centre
						8, // circle radius
						cv::Scalar(255), // color
						6);


					cv::circle(image2,pt3, // circle centre
						8, // circle radius
						cv::Scalar(255), // color
						6);


					cv::circle(image2,pt4, // circle centre
						8, // circle radius
						cv::Scalar(255), // color
						6);


					//get homography or perspective transform

					std::vector<cv::Point2f> src_vertices(4);
					src_vertices[0] = pt1;
					src_vertices[1] = pt3;
					src_vertices[2] = pt2;
					src_vertices[3] = pt4;

					//destination
					std::vector<cv::Point2f> dst_vertices(4);
					dst_vertices[0] = cv::Point(0, 0);
					dst_vertices[1] = cv::Point(image2.size().width-1, 0);
					dst_vertices[2] = cv::Point(0, image2.size().height-1);
					dst_vertices[3] = cv::Point(image2.size().width-1, image2.size().height-1);

					std::vector<cv::Point2f> src_quad_hull;
					std::vector<cv::Point2f> dst_quad_hull;
					cv::Mat image_hull_points;
					image_hull_points.resize(4);

					// Calculate convex hull of original points (which points positioned on the boundary)
					cv::convexHull(cv::Mat(src_vertices), src_quad_hull, false);
					cv::convexHull(cv::Mat(dst_vertices), dst_quad_hull, false);

					cv::Mat quad_homography = cv::getPerspectiveTransform(src_quad_hull, dst_quad_hull);

					cv::Mat rectified1(image2.size(), image2.type()); ;
					//cv::Size size(box.boundingRect().width, box.boundingRect().height);
					cv::Size img_size(image2.size().width, image2.size().height);
					/*cv::warpPerspective(image2, rotated, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);*/
					//cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
					cv::warpPerspective(image2, rectified1, quad_homography, img_size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);//cv::BORDER_TRANSPARENT);//, cv::INTER_LINEAR);
					cv::imwrite("rotated1.jpg", rectified1);

					//boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));


					boost::filesystem::path tmp_path(images_concerned[g]);

					std::string result_name = "results_"+ boost::lexical_cast<std::string>(i)+ "_" +tmp_path.stem().string()+ ".jpg";
					std::cout<<"result_image_name: "<<result_name<<std::endl;
					//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

					cv::imwrite(result_name, image2);
					std::cout<< image_point_l<<std::endl;
					std::cout<< image_point_r <<std::endl;
					std::cout<< image_point_up<<std::endl;
					std::cout<< image_point_down<<std::endl;
					//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

					std::cout<<images_concerned[g]<<std::endl;


					// Read the image which will be the texture
					vtkSmartPointer<vtkJPEGReader> jPEGReader =
						vtkSmartPointer<vtkJPEGReader>::New();
					jPEGReader->SetFileName ( "rotated1.jpg" );


					//reorder ending points for the opengl renderer
					pcl::PointCloud<pcl::PointXYZ>::Ptr quad_cloud(new pcl::PointCloud<pcl::PointXYZ>),
						ordered_hull_points(new pcl::PointCloud<pcl::PointXYZ>);
					quad_cloud->width = 4;
					quad_cloud->height = 1;
					quad_cloud->resize(quad_cloud->width * quad_cloud->height);
					quad_cloud->points[0] = fLineStart;
					quad_cloud->points[1] = fLineEndPoint;
					quad_cloud->points[2] = fLineStart1;
					quad_cloud->points[3] = fLineEndPoint1;
					getCloudHull(quad_cloud, ordered_hull_points);

					vtkSmartPointer<vtkActor> quad_actor = vtkSmartPointer<vtkActor>::New();
					addQuadActorToRenderer(ordered_hull_points, quad_actor);

					vtkSmartPointer<vtkActor> textured_quad_actor = vtkSmartPointer<vtkActor>::New();
					addTexturedQuadActorToRendererR(ordered_hull_points, "rotated1.jpg",textured_quad_actor);

					vtkSmartPointer<vtkActor> cloud_actor =	vtkSmartPointer<vtkActor>::New();
					addCloudActorToRenderer(cloud, cloud_actor);
					cloud_actor->GetProperty()->SetColor(0.0,0.0, 0.0);
					cloud_actor->GetProperty()->SetPointSize(3);

					//renderer->AddActor(texturedQuad);
					renderer->AddActor(textured_quad_actor);
					//renderer->AddActor(quad_actor);
					//renderer->AddActor(cloud_actor);
					renderer->AddActor(line_actor0);
					renderer->AddActor(line_actor1);
					renderer->AddActor(line_actor2);
					renderer->AddActor(line_actor3);
					//projectPointOnLine(fLineStart, line_b, fLineStart1);
					//projectPointOnLine(fLineEndPoint, line_b, fLineEndPoint1);

					//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fPerpendicularHanders (cloud, 255, 0, 0);
					//std::string fCloudId = "Planes Perpendicular Cloud" + boost::lexical_cast<std::string>(i);
					//viewer0->addPointCloud (cloud, fPerpendicularHanders, fCloudId );
					//viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
					//viewer0->addCoordinateSystem (1.0f);

					//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, "line 0"+i, 0); 
					//viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart1, fLineEndPoint1, 0.0, 255.0, 0.0, "line 1"+i, 0);

					//extract a real quad using line -line-intersection, vp, and the normal to the plane

					renderer->SetBackground(1,1,1); // Background color white
					renderer->ResetCamera();

					vtkSmartPointer<vtkRenderWindow> renderWindow =
						vtkSmartPointer<vtkRenderWindow>::New();
					renderWindow->AddRenderer(renderer);

					vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
						vtkSmartPointer<vtkRenderWindowInteractor>::New();
					renderWindowInteractor->SetRenderWindow(renderWindow);

					renderWindow->Render();

					renderWindowInteractor->Start();
					}//_a quad
				}//_a quad
					}//_a quad
										}//_a quad
			}
			}
			}
	}


	//while (!viewer0->wasStopped ())
	//{
	//	viewer0->spinOnce (100);
	//	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//}

}



bool fi::HouseModelBuilder::pointIsOnPlane(const pcl::ModelCoefficients::Ptr &model_coefficients_a, const Eigen::Vector3f &pPoint )
{
	float tolerance = 0.001;

	float dist = model_coefficients_a->values[0] * pPoint(0) + model_coefficients_a->values[1] * pPoint(1) + model_coefficients_a->values[2] * pPoint(2)  + model_coefficients_a->values[3];
	if (fabsf(dist) < tolerance)
	{
		return true;
	}
	return false;
}


bool fi::HouseModelBuilder::getIntersectionLines(const pcl::ModelCoefficients::Ptr &model_coefficients_a, const pcl::ModelCoefficients::Ptr &model_coefficients_b, Eigen::VectorXf &intersection_line )
{
	Eigen::Vector4f plane_a;
	plane_a(0) = model_coefficients_a->values[0];
	plane_a(1) = model_coefficients_a->values[1];
	plane_a(2) = model_coefficients_a->values[2];
	plane_a(3) = model_coefficients_a->values[3];
	Eigen::Vector4f plane_b;
	plane_b(0) = model_coefficients_b->values[0];
	plane_b(1) = model_coefficients_b->values[1];
	plane_b(2) = model_coefficients_b->values[2];
	plane_b(3) = model_coefficients_b->values[3];

	return pcl::planeWithPlaneIntersection(plane_a, plane_b, intersection_line);
}


void fi::HouseModelBuilder::getProjectionPointOnImage(const pcl::PointXYZ &reconstructed_3D_point,	
	Eigen::Vector2f &image_point, 
	const std::string &unibw_result_dir,
	const std::string &image_file_name,
	const Eigen::Matrix3f &cam_calib)
{
	boost::filesystem::path p0(image_file_name);
	std::string file_name = p0.stem().string() + "_param.txt" ;

	std::string param_path(unibw_result_dir + "\\" + file_name);

	std::cout <<"params filename: "<<param_path<<std::endl;

	boost::filesystem::path p(param_path);
	Eigen::Matrix3f rotations_matrix; 
	Eigen::Vector3f translation_vector;
	float radial_dist1; 
	float radial_dist2;
	unsigned int img_width;
	unsigned int img_height;

	fi::CamPOSEParser tmp_pose;
	tmp_pose.getCamRotationMatrix(p, rotations_matrix);
	tmp_pose.getCamTranslationVector(p, translation_vector);
	tmp_pose.getRadialDistortionParams(p, radial_dist1, radial_dist2);
	img_height = tmp_pose.getImageHeight(p);
	img_width = tmp_pose.getImageWidth(p);

	Eigen::Vector3f tmp_point;
	tmp_point(0) = reconstructed_3D_point.x;
	tmp_point(1) = reconstructed_3D_point.y;
	tmp_point(2) = reconstructed_3D_point.z;

	//place point to camera axis
	Eigen::Vector3f point_cam = rotations_matrix*tmp_point + translation_vector;

	//project on x-y plane
	Eigen::Vector3f point_xy1;
	point_xy1(0) = point_cam(0)/point_cam(2);
	point_xy1(1) = point_cam(1)/point_cam(2);
	point_xy1(2) = 1.0f;

	Eigen::Vector3f p_h = cam_calib*point_xy1;

	//undistortion
	float r = sqrtf(p_h(0)*p_h(0) + p_h(1)*p_h(1)); 

	float xa = p_h(0)*(1 + radial_dist1*(r*r) + radial_dist2*(r*r*r*r));
	float ya = p_h(1)*(1 + radial_dist1*(r*r) + radial_dist2*(r*r*r*r));

	//rescale
	float img_scale = (img_width + img_height)/2;
	float step_x = img_width/2;
	float step_y = img_height/2;

	float pixel_x = xa * img_scale + step_x;
	float pixel_y = -ya * img_scale + step_y;

	image_point(0) = pixel_x;
	image_point(1) = pixel_y;

	std::cout<<"translation vector"<<translation_vector<<std::endl;
	std::cout<<"rotation Matrix"<<rotations_matrix<<std::endl;
	std::cout<<"radial distortion1"<<radial_dist1<<"radial2"<<radial_dist2<<std::endl;
	std::cout<<"cam calib matrix: "<<cam_calib<<std::endl;
	std::cout<<"image height: "<<img_height<<std::endl;
	std::cout<<"image width: "<<img_width<<std::endl;
	//boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));
}




void fi::HouseModelBuilder::addLineToRenderer(const pcl::PointXYZ &fPA, const pcl::PointXYZ &fPB, vtkSmartPointer<vtkActor> &fLineActor)
{
	double fLineResolution = 0.01;

	// Create two points, P0 and P1
	double p0[3] = {fPA.x, fPA.y, fPA.z}; 
	double p1[3] = {fPB.x, fPB.y, fPB.z};


	vtkSmartPointer<vtkLineSource> lineSource = 
		vtkSmartPointer<vtkLineSource>::New();
	lineSource->SetPoint1(p0);
	lineSource->SetPoint2(p1);
	lineSource->Update();

	// Visualize
	vtkSmartPointer<vtkPolyDataMapper> mapper = 
		vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(lineSource->GetOutputPort());
	vtkSmartPointer<vtkActor> actor = 
		vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetLineWidth(4);
	fLineActor = actor;

	//vtkSmartPointer<vtkRenderer> renderer = 
	//	vtkSmartPointer<vtkRenderer>::New();
	//vtkSmartPointer<vtkRenderWindow> renderWindow = 
	//	vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->AddRenderer(renderer);
	//vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
	//	vtkSmartPointer<vtkRenderWindowInteractor>::New();
	//renderWindowInteractor->SetRenderWindow(renderWindow);

	//renderer->AddActor(actor);

	//renderWindow->Render();
	//renderWindowInteractor->Start();

	//return EXIT_SUCCESS;

}



void fi::HouseModelBuilder::addCloudActorToRenderer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, vtkSmartPointer<vtkActor> &fCloudActor)
{
	unsigned int nPointsPresent = point_cloud->points.size();
	vtkSmartPointer<vtkPoints> sPoints = vtkSmartPointer<vtkPoints>::New();

	for (int i=0; i < nPointsPresent; i++)
	{
		const float  tmpPoint[3] = {point_cloud->points[i].x, point_cloud->points[i].y, point_cloud->points[i].z}; 
		sPoints->InsertNextPoint(tmpPoint);
	}

	vtkSmartPointer<vtkPolyData> sPointsPolyData = vtkSmartPointer<vtkPolyData>::New();
	sPointsPolyData->SetPoints(sPoints);

	vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter =	vtkSmartPointer<vtkVertexGlyphFilter>::New();
	vertexFilter->SetInputConnection(sPointsPolyData->GetProducerPort());
	vertexFilter->Update();

	vtkSmartPointer<vtkPolyData> polydata =	vtkSmartPointer<vtkPolyData>::New();
	polydata->ShallowCopy(vertexFilter->GetOutput());

	//Visualize
	vtkSmartPointer<vtkPolyDataMapper> mapper = 
		vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInput(polydata);
	vtkSmartPointer<vtkActor> actor = 
		vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(0.0,0.0,0.0);
	fCloudActor = actor;
}



void fi::HouseModelBuilder::GetEdges(const unsigned int fTopPercent, const Eigen::Vector3f &fVP, std::vector<int> &fEdges)
{
	unsigned int nPointsCandidates = m_cloud->points.size();

	//Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (m_cloud);
	n.setInputCloud (m_cloud);
	n.setSearchMethod (tree);
	n.setKSearch (30);
	n.compute (*normals);

	std::vector<float> fPrincipalCurvatures;

	float fMaxCurvature = 0.0f;

	for (unsigned int i= 0; i < nPointsCandidates; i++)
	{
		fPrincipalCurvatures.push_back(normals->points[i].curvature);

		if(fMaxCurvature < normals->points[i].curvature)
			fMaxCurvature = normals->points[i].curvature;
	}
	//for_each(fPrincipalCurvatures.begin(), fPrincipalCurvatures.end(),[&](float a){ fPrincipalCurvatures});
	//normalise curvature value between (1-0)
	for (unsigned int i= 0; i < nPointsCandidates; i++)
	{
		fPrincipalCurvatures[i] = fPrincipalCurvatures[i]/fMaxCurvature;
	}

	//take the top 10%
	const float fpercentageLimit = 1.0f - float(fTopPercent)/100;

	unsigned int fCounts = 0;

	for (unsigned int i = 0; i < nPointsCandidates; i++)
	{
		if (fPrincipalCurvatures[i] > float(fTopPercent)/100)
		{
			fCounts++;
			fEdges.push_back(i);
		}
	}

	std::cout <<"number of edge points: "<<fCounts<<std::endl;
	std::cout <<"original cloud size: "<<m_cloud->points.size()<<std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_color(new pcl::PointCloud<pcl::PointXYZ>);

	cloud_to_color->height = 1;
	cloud_to_color->width = fEdges.size();
	cloud_to_color->resize(cloud_to_color->width * cloud_to_color->height);

	for(unsigned int i = 0; i < fEdges.size(); i++)
	{
		cloud_to_color->points[i] = m_cloud->points[fEdges.at(i)];
	}

	//remove outliers
	//PtCloudDataPtr mfToColorFiltered(new PtCloudData);
	//RadiusOutlierFiltering(mfToColor, mfToColorFiltered, 20, 0.3);


	pcl::visualization::PCLVisualizer viewer ("3D Viewer");

	viewer.setBackgroundColor (0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle (m_cloud, 255, 0, 255);
	viewer.addPointCloud (m_cloud, fOriginalCloudHandle, "Original Cloud");

	viewer.setBackgroundColor (0.3, 0.3, 0.3);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler (cloud_to_color, 0, 255, 0);
	viewer.addPointCloud (cloud_to_color, EdgesInliersHandler, "edges cloud");

	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "edges cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Original Cloud");
	viewer.addCoordinateSystem (1.0f);

	while (!viewer.wasStopped () /*&& range_image_widget.isShown ()*/)
	{
		//	range_image_widget.spinOnce ();
		viewer.spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}


	//Create the segmentation object for the linear model model and set all the parameters

	//// initialize PointClouds
	//pcl::PointCloud < pcl::PointXYZ >::Ptr final (new pcl::PointCloud <	pcl::PointXYZ >);

	//std::vector < int >inliers1;

	//// created RandomSampleConsensus object and compute the appropriated model
	//pcl::SampleConsensusModelPerpendicularPlane <pcl::PointXYZ >::Ptr model_p (new pcl::SampleConsensusModelParallelLine <pcl::PointXYZ > (mfToColorFiltered));
	//

	//model_p->setAxis (Eigen::Vector3f (fVP.x, fVP.y, fVP.z));
	//model_p->setEpsAngle (pcl::deg2rad (10.0));
	//pcl::RandomSampleConsensus < pcl::PointXYZ > ransac (model_p);
	//ransac.setDistanceThreshold (0.05);
	//ransac.computeModel(2);                  //Segment Fault when run this line
	//ransac.getInliers (inliers1);
	//pcl::copyPointCloud < pcl::PointXYZ > (*mfToColorFiltered, inliers1, *final); 









	//oriented lines segmentation
	///////*PtCloudDataPtr fInliers(new PtCloudData);*/
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr fInliers (new pcl::PointCloud<pcl::PointXYZ> );
	//////PtCloudDataPtr fRemainders(new PtCloudData);

	//////int fNIterations = 2000;
	//////const float fTolerance = 0.2;

	//////
	////////use ransac to detect lines

	//////std::cout<<"running OrientedLineRansac..."<<std::endl;
	//////
	//////const Vec3d mLineOrientation = {fVP.x, fVP.y, fVP.z};
	//////OrientedLinesRANSAC(mfToColorFiltered, fInliers, fRemainders, mLineOrientation, fNIterations,fTolerance);
	//////
	//////std::cout<<"finnished OrientedLineRansac!"<<std::endl;

	//////
	//////pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");

	//////viewer0.setBackgroundColor (0, 0, 0);
	//////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (m_wPclDataPtr, 255, 0, 255);
	////////viewer.addPointCloud (m_wPclDataPtr, fOriginalCloudHandle, "Original Cloud");


	//////
	//////viewer0.setBackgroundColor (0.3, 0.3, 0.3);
	//////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (mfToColorFiltered, 0, 255, 0);
	//////viewer0.addPointCloud (mfToColorFiltered, EdgesInliersHandler0, "edges cloud");

	//////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "edges cloud");
	//////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Original Cloud");
	//////viewer0.addCoordinateSystem (1.0f);
	//////float fLineScale = 3.0f;

	//////pcl::PointXYZ fLineStart;
	//////fLineStart.x=fInliers->points[0].x - fLineScale*mLineOrientation.x;
	//////fLineStart.y=fInliers->points[0].y - fLineScale*mLineOrientation.y;
	//////fLineStart.z=fInliers->points[0].z - fLineScale*mLineOrientation.z;
	//////pcl::PointXYZ fLineEndPoint;

	//////fLineEndPoint.x = fInliers->points[0].x + fLineScale*mLineOrientation.x;
	//////fLineEndPoint.y = fInliers->points[0].y + fLineScale*mLineOrientation.y;
	//////fLineEndPoint.z = fInliers->points[0].z + fLineScale*mLineOrientation.z;

	//////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, "rhandel bow", 0); 

	//////Eigen::VectorXf fLineoptimized_coefficients;
	//////LSLineFitting(fInliers, fLineoptimized_coefficients);
	//////pcl::ModelCoefficients fLineCoefs;//(new pcl::ModelCoefficients());

	//////pcl::PointXYZ fPointCentroid_linestart;
	//////fPointCentroid_linestart.x = fLineoptimized_coefficients[0] - 1.5*fLineScale*fLineoptimized_coefficients[3];
	//////fPointCentroid_linestart.y = fLineoptimized_coefficients[1] - 1.5*fLineScale*fLineoptimized_coefficients[4];
	//////fPointCentroid_linestart.z = fLineoptimized_coefficients[2] - 1.5*fLineScale*fLineoptimized_coefficients[5];
	//////pcl::PointXYZ fPointCentroid_lineEnd;
	//////fPointCentroid_lineEnd.x = fLineoptimized_coefficients[0] + 1.5*fLineScale*fLineoptimized_coefficients[3];
	//////fPointCentroid_lineEnd.y = fLineoptimized_coefficients[1] + 1.5*fLineScale*fLineoptimized_coefficients[4];
	//////fPointCentroid_lineEnd.z = fLineoptimized_coefficients[2] + 1.5*fLineScale*fLineoptimized_coefficients[5];
	//////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fPointCentroid_linestart, fPointCentroid_lineEnd, 0.0, 2.0, 255.0, "New Better Line",0);
	//////
	//////viewer0.setBackgroundColor (0, 0, 0);
	//////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fLineCloudHandle0 (fInliers, 255, 0, 0);
	//////std::string fCloudId = "Line Cloud" + boost::lexical_cast<std::string>( 0);;
	//////viewer0.addPointCloud (fInliers, fLineCloudHandle0, "Line Cloud" );
	//////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
	//////viewer0.addCoordinateSystem (1.0f);

	//////while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
	//////{
	//////	//	range_image_widget.spinOnce ();
	//////	viewer0.spinOnce (100);
	//////	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//////}


	//////std::cout<<"Plotted results of OrientedLineRansac"<<std::endl;
	//////	
	//////// Create the segmentation object for the planar model and set all the parameters
	//////boost::shared_ptr<pcl::SACSegmentation<pcl::PointXYZ>> seg(new pcl::SACSegmentation<pcl::PointXYZ>);
	//////pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	//////pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ),
	//////	cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
	//////	
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>(*mfToColorFiltered));

	//////seg->setOptimizeCoefficients (true);
	//////seg->setModelType (pcl::SACMODEL_LINE);
	////////seg->setModelType (pcl::SACMODEL_PLANE);
	//////seg->setMethodType (pcl::SAC_RANSAC);
	//////seg->setMaxIterations (5000);
	//////seg->setAxis (Eigen::Vector3f (fVP.x, fVP.y, fVP.z));
	////////seg.setEpsAngle (pcl::deg2rad (19.0));
	//////seg->setDistanceThreshold (0.1);


	//////std::vector<PtCloudDataPtr> fLinesInliers;


	//////int i=0, nr_points = (int) cloud_filtered ->points.size ();
	//////int kk=0;
	//////while (cloud_filtered->points.size () > 0.3 * nr_points)
	//////{
	//////	// Segment the largest planar component from the remaining cloud
	//////	seg->setInputCloud(cloud_filtered );
	//////	seg->segment (*inliers, *coefficients); //*
	//////	if (inliers->indices.size () == 0)
	//////	{
	//////		std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
	//////		break;
	//////	}

	//////	// Extract the planar inliers from the input cloud
	//////	pcl::ExtractIndices<pcl::PointXYZ> extract ;
	//////	extract.setInputCloud (cloud_filtered );
	//////	extract.setIndices (inliers);
	//////	extract.setNegative (false);

	//////	// Write the planar inliers to disk
	//////	extract.filter (*cloud_plane); //*
	//////	std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

	//////	fLinesInliers.push_back(cloud_plane);


	//////	//pcl::ProjectInliers<pcl::PointXYZ> proj;
	//////	//proj.setModelType (pcl::SACMODEL_LINE);
	//////	//proj.setInputCloud (cloud_filtered);
	//////	//proj.setModelCoefficients (coefficients);
	//////	//proj.filter(*cloud_projected);

	//////	pcl::visualization::PCLVisualizer viewer2 ("3D Viewer");

	//////	viewer2.setBackgroundColor (0, 0, 0);
	//////	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle (m_wPclDataPtr, 255, 0, 255);
	//////	//viewer.addPointCloud (m_wPclDataPtr, fOriginalCloudHandle, "Original Cloud");


	//////	viewer2.setBackgroundColor (0.3, 0.3, 0.3);
	//////	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler (mfToColorFiltered, 0, 255, 0);
	//////	viewer2.addPointCloud (mfToColorFiltered, EdgesInliersHandler, "edges cloud");

	//////	viewer2.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "edges cloud");
	//////	viewer2.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Original Cloud");
	//////	viewer2.addCoordinateSystem (1.0f);





	//////	
	//////	viewer2.setBackgroundColor (0, 0, 0);
	//////	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fLineCloudHandle (cloud_plane, 255, 0, 0);
	//////	std::string fCloudId = "Line Cloud" + boost::lexical_cast<std::string>( kk);;
	//////	viewer2.addPointCloud (cloud_plane, fLineCloudHandle,fCloudId );
	//////	viewer2.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
	//////	viewer2.addCoordinateSystem (1.0f);

	//////	while (!viewer2.wasStopped () /*&& range_image_widget.isShown ()*/)
	//////	{
	//////	//	range_image_widget.spinOnce ();
	//////		viewer2.spinOnce (100);
	//////		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//////	}

	//////	
	//////	// Remove the planar inliers, extract the rest
	//////	PtCloudDataPtr fRemainderCloud(new PtCloudData);
	//////	extract.setNegative (true);
	//////	extract.filter (*fRemainderCloud ); //*
	//////	*cloud_filtered = *fRemainderCloud;

	////////	unsigned int nnewclouds = cloud_filtered->points.size() - inliers->indices.size();

	////////	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	////////	new_cloud_filtered->width = nnewclouds;
	////////	new_cloud_filtered->height = 1;
	////////	new_cloud_filtered->resize(new_cloud_filtered->width * new_cloud_filtered->height);

	////////	//std::vector <int>::iterator result;

	////////	for (int i = 0; i < cloud_filtered->points.size(); i++)
	////////	{

	////////		auto result = std::find_if(inliers->indices.begin(), inliers->indices.end(),[=](int j){return i==j;});
	////////		void* pt = static_cast<void*>(&result[0]);

	////////		
	////////		
	////////		std::cout<<&result<<std::endl;
	////////		std::cout<<(int)pt<<std::endl;


	////////		/*if( *result!=i)*/
	////////		new_cloud_filtered->points[i] = cloud_filtered->points[inliers->indices.at(i)];
	////////		
	////////	}

	////////cloud_filtered = new_cloud_filtered;
	//////	
	//////kk++;
	//////}










	////////pcl::SACSegmentation<pcl::PointXYZ> seg;
	////////pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	////////pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	////////pcl::PointCloud<pcl::PointXYZ>::Ptr fcloudlines (new pcl::PointCloud<pcl::PointXYZ> ());
	////////
	////////seg.setOptimizeCoefficients (true);
	////////seg.setModelType (pcl::SACMODEL_PARALLEL_LINE);
	////////
	////////seg.setAxis (Eigen::Vector3f (fVP.x, fVP.y, fVP.z));
	////////seg.setEpsAngle (pcl::deg2rad (10.0));
	////////seg.setMethodType (pcl::SAC_RANSAC);
	////////seg.setMaxIterations (500);
	////////seg.setDistanceThreshold (0.2);
	////////
	////////int i=0, nr_points = (int) mfToColorFiltered->points.size ();
	//////// while (mfToColorFiltered->points.size () > 0.3 * nr_points)
	//////// {
	////////   // Segment the largest planar component from the remaining cloud
	////////   seg.setInputCloud(mfToColorFiltered);
	////////   seg.segment (*inliers, *coefficients); //*
	////////   if (inliers->indices.size () == 0)
	////////   {
	////////     std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
	////////     break;
	////////   }

	////////   // Extract the linear inliers from the input cloud
	////////   pcl::ExtractIndices<pcl::PointXYZ> extract;
	////////   extract.setInputCloud (mfToColorFiltered);
	////////   extract.setIndices (inliers);
	////////   extract.setNegative (false);

	////////   // Write the planar inliers to disk
	////////   extract.filter (*mfToColorFiltered); //*
	////////   std::cout << "PointCloud representing the planar component: " << mfToColorFiltered->points.size () << " data points." << std::endl;

	////////   // Remove the linear inliers, extract the rest
	////////   extract.setNegative (true);
	////////   extract.filter (*mfToColorFiltered); //*











	////////boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerM (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	////////	
	//////////viewerM->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 3, "sample cloud");
	/////////*pcl::visualization::CloudViewer fviewer("Simple Inliers Test!");*/

	////////// Open a 3D viewer
	////////pcl::visualization::PCLVisualizer viewer ("3D Viewer");
	////////// Set the background of viewer
	//////////pcl_visualization::PCLVisualizer::setBackgroundColor (const double&r, const double &g, const double &b, int viewport);
	////////viewer.setBackgroundColor (0.0, 0.0, 0.0);
	////////// Add system coordiante to viewer
	////////viewer.addCoordinateSystem (2.0f);
	////////// Add the point cloud data
	//////////viewer.addPointCloud (m_wPclDataPtr, "cloud1");
	////////// And wait until key is pressed
	/////////*viewer.spin (); */
	////////viewer.addPointCloud(mfToColorFiltered, "cloud2");
	////////viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud2");

	////////viewer.addCoordinateSystem (1.0f);
	////////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_cloud_color_handler(mfToColor, 150,150,150);
	////////viewer.addPointCloud (mfToColor, point_cloud_color_handler, "original point cloud");

	////////while(!viewer.wasStopped())
	////////{
	////////viewer.spinOnce (100);
	////////boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	////////}
	////////
	///////*

	//////fviewer.showCloud(mfToColor);
	//////while (!fviewer.wasStopped())
	//////{
	//////}
	//////*/







	////////normals should not contain the point normals + surface curvatures

	//////////Concatenate the XYZ and normal fields*
	////////pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	////////pcl::concatenateFields (*m_wPclDataPtr, *normals, *cloud_with_normals);
	////////cloud_with_normals = m_wPclDataPtr + normals;

	////////const PtCloudDataPtr &fIndata = m_wPclDataPtr;
	////////// Create the normal estimation class, and pass the input dataset to it
	////////pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	////////ne.setInputCloud (m_wPclDataPtr);

	////////// Create an empty kdtree representation, and pass it to the normal estimation object.
	////////// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	////////pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	////////ne.setSearchMethod (tree);

	////////// Output datasets
	////////pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

	////////// Use all neighbors in a sphere of radius 3cm
	//////////ne.setRadiusSearch (0.08);
	////////ne.setKSearch(20);

	////////// Compute the features
	////////ne.compute (*cloud_normals);

	////////PtCloudDataPtr outPutf(new PtCloudData);
	////////
	////////a(*m_PCLInputData, *cloud_normals, outPutf);
	//////////
	//////////*pcl::io::savePCDFile ("bun0-mls.pcd", *outPutf);*/
	////////pcl::PCDWriter writer;
	////////writer.write ("haus51k20.pcd", *cloud_with_normals, false);

}


void fi::HouseModelBuilder::getCloudHull(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, pcl::PointCloud<pcl::PointXYZ>::Ptr &ordered_hull_points)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ConvexHull<pcl::PointXYZ> chull;
	chull.setInputCloud (quad_points);
	chull.setComputeAreaVolume(true);
	//chull.setKeepInformation(true);
	/*chull.setAlpha (0.1);*/
	chull.reconstruct (*cloud_hull);
	//	auto rnt = chull.getDim();

	double fArea = chull.getTotalArea();
	double fVolume = chull.getTotalVolume();
	std::cout<<"hull computation succeed "<<std::endl;
	*ordered_hull_points = *cloud_hull ;
	std::cout<<"hull copying succeeded "<<std::endl;


	//		pcl::visualization::PCLVisualizer viewer ("3D Viewer");
	//	
	//		viewer.setBackgroundColor (0, 0, 0);
	//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle (cloud_hull, 255, 0, 255);
	//		viewer.addPointCloud (cloud_hull, fOriginalCloudHandle, "Original Cloud");
	//	
	//	unsigned int l_id = 0;
	//		for (unsigned int i = 0; i < cloud_hull->points.size()-1; i++)
	//		{
	//			pcl::PointXYZ &l_a = cloud_hull->points[i];
	//			pcl::PointXYZ &l_b = cloud_hull->points[i+1];
	//			std::string fName = " line1" + boost::lexical_cast<std::string, unsigned int>(l_id++); 
	//			viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(l_a, l_b, 0.0, 255.0, 0.0, fName, 0);
	//			viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, fName);
	//	
	//		}
	//	
	//		//add the last part of the line
	//		pcl::PointXYZ &l_a = cloud_hull->points[3];
	//		pcl::PointXYZ &l_b = cloud_hull->points[0];
	//		std::string fName = " line1" + boost::lexical_cast<std::string, unsigned int>(l_id++); 
	//		viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(l_a, l_b, 0.0, 255.0, 0.0, fName, 0);
	//		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, fName);
	//
	//
	//	
	//
	//		///*viewer0.setBackgroundColor (0, 0, 0);
	//		//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fLineCloudHandle0 (fInliers, 255, 0, 0);
	//		//std::string fCloudId = "Line Cloud" + boost::lexical_cast<std::string>( 0);;
	//		//viewer0.addPointCloud (fInliers, fLineCloudHandle0, "Line Cloud" );*/
	//		//viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Original Cloud");
	//		//viewer.addCoordinateSystem (1.0f);*/
	//	
	//		while (!viewer.wasStopped () /*&& range_image_widget.isShown ()*/)
	//		{
	//			//	range_image_widget.spinOnce ();
	//			viewer.spinOnce (100);
	//			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//		}
	//boost::this_thread::sleep (boost::posix_time::microseconds (1000000000));
	//std::cout<<"Ploting outline"<<std::endl;
}


void fi::HouseModelBuilder::addQuadActorToRenderer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &fQuadEdges, vtkSmartPointer<vtkActor> &fQuadActor)
{	
	//// Create four points (must be in counter clockwise order)
	/*double p0[3] = {0.0, 0.0, 0.0};
	double p1[3] = {1.0, 0.0, 0.0};
	double p2[3] = {1.0, 1.0, 0.0};
	double p3[3] = {0.0, 1.0, 0.0};*/


	double p0[3] = {fQuadEdges->points[0].x, fQuadEdges->points[0].y, fQuadEdges->points[0].z};
	double p1[3] = {fQuadEdges->points[1].x, fQuadEdges->points[1].y, fQuadEdges->points[1].z};
	double p2[3] = {fQuadEdges->points[2].x, fQuadEdges->points[2].y, fQuadEdges->points[2].z};
	double p3[3] = {fQuadEdges->points[3].x, fQuadEdges->points[3].y, fQuadEdges->points[3].z};

	// Add the points to a vtkPoints object
	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	points->InsertNextPoint(p0);
	points->InsertNextPoint(p1);
	points->InsertNextPoint(p2);
	points->InsertNextPoint(p3);

	// Create a quad on the four points
	vtkSmartPointer<vtkQuad> quad =
		vtkSmartPointer<vtkQuad>::New();
	quad->GetPointIds()->SetId(0,0);
	quad->GetPointIds()->SetId(1,1);
	quad->GetPointIds()->SetId(2,2);
	quad->GetPointIds()->SetId(3,3);

	// Create a cell array to store the quad in
	vtkSmartPointer<vtkCellArray> quads =
		vtkSmartPointer<vtkCellArray>::New();
	quads->InsertNextCell(quad);

	// Create a polydata to store everything in
	vtkSmartPointer<vtkPolyData> polydata =
		vtkSmartPointer<vtkPolyData>::New();

	// Add the points and quads to the dataset
	polydata->SetPoints(points);
	polydata->SetPolys(quads);

	// Setup actor and mapper
	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInput(polydata);

	vtkSmartPointer<vtkActor> actor =
		vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	fQuadActor = actor;
}



void fi::HouseModelBuilder::addTexturedQuadActorToRenderer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, const std::string &image_name_jpeg, vtkSmartPointer<vtkActor> &textured_quad_actor)
{	
	// Read the image which will be the texture
	vtkSmartPointer<vtkJPEGReader> jPEGReader =
		vtkSmartPointer<vtkJPEGReader>::New();
	jPEGReader->SetFileName ( image_name_jpeg.c_str() );

	// Create a plane
	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	points->InsertNextPoint(quad_points->points[3].x, quad_points->points[3].y, quad_points->points[3].z);
	points->InsertNextPoint(quad_points->points[2].x, quad_points->points[2].y, quad_points->points[2].z);
	points->InsertNextPoint(quad_points->points[1].x, quad_points->points[1].y, quad_points->points[1].z);
	points->InsertNextPoint(quad_points->points[0].x, quad_points->points[0].y, quad_points->points[0].z);




	vtkSmartPointer<vtkCellArray> polygons =
		vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPolygon> polygon =
		vtkSmartPointer<vtkPolygon>::New();
	polygon->GetPointIds()->SetNumberOfIds(4); //make a quad
	polygon->GetPointIds()->SetId(0, 0);
	polygon->GetPointIds()->SetId(1, 1);
	polygon->GetPointIds()->SetId(2, 2);
	polygon->GetPointIds()->SetId(3, 3);

	polygons->InsertNextCell(polygon);

	vtkSmartPointer<vtkPolyData> quad =
		vtkSmartPointer<vtkPolyData>::New();
	quad->SetPoints(points);
	quad->SetPolys(polygons);

	vtkSmartPointer<vtkFloatArray> textureCoordinates =
		vtkSmartPointer<vtkFloatArray>::New();
	textureCoordinates->SetNumberOfComponents(3);
	textureCoordinates->SetName("TextureCoordinates");

	float tuple[3] = {0.0, 0.0, 0.0};
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 1.0; tuple[1] = 0.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 1.0; tuple[1] = 1.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 0.0; tuple[1] = 1.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);

	quad->GetPointData()->SetTCoords(textureCoordinates);

	// Apply the texture
	vtkSmartPointer<vtkTexture> texture =
		vtkSmartPointer<vtkTexture>::New();
	texture->SetInputConnection(jPEGReader->GetOutputPort());

	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
	mapper->SetInput(quad);
#else
	mapper->SetInputData(quad);
#endif

	vtkSmartPointer<vtkActor> texturedQuad =
		vtkSmartPointer<vtkActor>::New();
	texturedQuad->SetMapper(mapper);
	texturedQuad->SetTexture(texture);
	textured_quad_actor = texturedQuad;

	//// Visualize the textured plane
	//vtkSmartPointer<vtkRenderer> renderer =
	//	vtkSmartPointer<vtkRenderer>::New();
	//renderer->AddActor(texturedQuad);
	//renderer->SetBackground(1,1,1); // Background color white
	//renderer->ResetCamera();

	//vtkSmartPointer<vtkRenderWindow> renderWindow =
	//	vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->AddRenderer(renderer);

	//vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
	//	vtkSmartPointer<vtkRenderWindowInteractor>::New();
	//renderWindowInteractor->SetRenderWindow(renderWindow);

	//renderWindow->Render();

	//renderWindowInteractor->Start();

	//return EXIT_SUCCESS;
}


void fi::HouseModelBuilder::addTexturedQuadActorToRendererR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, const std::string &image_name_jpeg, vtkSmartPointer<vtkActor> &textured_quad_actor)
{	
	// Read the image which will be the texture
	vtkSmartPointer<vtkJPEGReader> jPEGReader =
		vtkSmartPointer<vtkJPEGReader>::New();
	jPEGReader->SetFileName ( image_name_jpeg.c_str() );

	// Create a plane
	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	points->InsertNextPoint(quad_points->points[3].x, quad_points->points[3].y, quad_points->points[3].z);
	points->InsertNextPoint(quad_points->points[0].x, quad_points->points[0].y, quad_points->points[0].z);
	points->InsertNextPoint(quad_points->points[1].x, quad_points->points[1].y, quad_points->points[1].z);
	points->InsertNextPoint(quad_points->points[2].x, quad_points->points[2].y, quad_points->points[2].z);
	

	vtkSmartPointer<vtkCellArray> polygons =
		vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPolygon> polygon =
		vtkSmartPointer<vtkPolygon>::New();
	polygon->GetPointIds()->SetNumberOfIds(4); //make a quad
	polygon->GetPointIds()->SetId(0, 0);
	polygon->GetPointIds()->SetId(1, 1);
	polygon->GetPointIds()->SetId(2, 2);
	polygon->GetPointIds()->SetId(3, 3);

	polygons->InsertNextCell(polygon);

	vtkSmartPointer<vtkPolyData> quad =
		vtkSmartPointer<vtkPolyData>::New();
	quad->SetPoints(points);
	quad->SetPolys(polygons);

	vtkSmartPointer<vtkFloatArray> textureCoordinates =
		vtkSmartPointer<vtkFloatArray>::New();
	textureCoordinates->SetNumberOfComponents(3);
	textureCoordinates->SetName("TextureCoordinates");

	float tuple[3] = {0.0, 0.0, 0.0};
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 1.0; tuple[1] = 0.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 1.0; tuple[1] = 1.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 0.0; tuple[1] = 1.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);

	quad->GetPointData()->SetTCoords(textureCoordinates);

	// Apply the texture
	vtkSmartPointer<vtkTexture> texture =
		vtkSmartPointer<vtkTexture>::New();
	texture->SetInputConnection(jPEGReader->GetOutputPort());

	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
	mapper->SetInput(quad);
#else
	mapper->SetInputData(quad);
#endif

	vtkSmartPointer<vtkActor> texturedQuad =
		vtkSmartPointer<vtkActor>::New();
	texturedQuad->SetMapper(mapper);
	texturedQuad->SetTexture(texture);
	textured_quad_actor = texturedQuad;

	//// Visualize the textured plane
	//vtkSmartPointer<vtkRenderer> renderer =
	//	vtkSmartPointer<vtkRenderer>::New();
	//renderer->AddActor(texturedQuad);
	//renderer->SetBackground(1,1,1); // Background color white
	//renderer->ResetCamera();

	//vtkSmartPointer<vtkRenderWindow> renderWindow =
	//	vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->AddRenderer(renderer);

	//vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
	//	vtkSmartPointer<vtkRenderWindowInteractor>::New();
	//renderWindowInteractor->SetRenderWindow(renderWindow);

	//renderWindow->Render();

	//renderWindowInteractor->Start();

	//return EXIT_SUCCESS;
}



void fi::HouseModelBuilder::addTexturedQuadActorToRendererRR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, const std::string &image_name_jpeg, vtkSmartPointer<vtkActor> &textured_quad_actor)
{	
	// Read the image which will be the texture
	vtkSmartPointer<vtkJPEGReader> jPEGReader =
		vtkSmartPointer<vtkJPEGReader>::New();
	jPEGReader->SetFileName ( image_name_jpeg.c_str() );

	// Create a plane
	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
		
	
	points->InsertNextPoint(quad_points->points[3].x, quad_points->points[3].y, quad_points->points[3].z);
		points->InsertNextPoint(quad_points->points[2].x, quad_points->points[2].y, quad_points->points[2].z);
		points->InsertNextPoint(quad_points->points[1].x, quad_points->points[1].y, quad_points->points[1].z);
		  points->InsertNextPoint(quad_points->points[0].x, quad_points->points[0].y, quad_points->points[0].z);
	  
		
		
	

	vtkSmartPointer<vtkCellArray> polygons =
		vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPolygon> polygon =
		vtkSmartPointer<vtkPolygon>::New();
	polygon->GetPointIds()->SetNumberOfIds(4); //make a quad
	polygon->GetPointIds()->SetId(0, 0);
	polygon->GetPointIds()->SetId(1, 1);
	polygon->GetPointIds()->SetId(2, 2);
	polygon->GetPointIds()->SetId(3, 3);

	polygons->InsertNextCell(polygon);

	vtkSmartPointer<vtkPolyData> quad =
		vtkSmartPointer<vtkPolyData>::New();
	quad->SetPoints(points);
	quad->SetPolys(polygons);

	vtkSmartPointer<vtkFloatArray> textureCoordinates =
		vtkSmartPointer<vtkFloatArray>::New();
	textureCoordinates->SetNumberOfComponents(3);
	textureCoordinates->SetName("TextureCoordinates");

	float tuple[3] = {0.0, 0.0, 0.0};
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 1.0; tuple[1] = 0.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 1.0; tuple[1] = 1.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);
	tuple[0] = 0.0; tuple[1] = 1.0; tuple[2] = 0.0;
	textureCoordinates->InsertNextTuple(tuple);

	quad->GetPointData()->SetTCoords(textureCoordinates);

	// Apply the texture
	vtkSmartPointer<vtkTexture> texture =
		vtkSmartPointer<vtkTexture>::New();
	texture->SetInputConnection(jPEGReader->GetOutputPort());

	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
	mapper->SetInput(quad);
#else
	mapper->SetInputData(quad);
#endif

	vtkSmartPointer<vtkActor> texturedQuad =
		vtkSmartPointer<vtkActor>::New();
	texturedQuad->SetMapper(mapper);
	texturedQuad->SetTexture(texture);
	textured_quad_actor = texturedQuad;

	//// Visualize the textured plane
	//vtkSmartPointer<vtkRenderer> renderer =
	//	vtkSmartPointer<vtkRenderer>::New();
	//renderer->AddActor(texturedQuad);
	//renderer->SetBackground(1,1,1); // Background color white
	//renderer->ResetCamera();

	//vtkSmartPointer<vtkRenderWindow> renderWindow =
	//	vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->AddRenderer(renderer);

	//vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
	//	vtkSmartPointer<vtkRenderWindowInteractor>::New();
	//renderWindowInteractor->SetRenderWindow(renderWindow);

	//renderWindow->Render();

	//renderWindowInteractor->Start();

	//return EXIT_SUCCESS;
}



//#include <ModelSegmentation/HouseModelBuilder.h>
//#include <ModelSegmentation/PlaneModel.h>
//#include <ModelSegmentation/ModelSegmentation.h>
//#include <ModelSegmentation/ModelContext.h>
//#include "VanishingPointDetection/VPDetection.h"
//#include "VanishingPointDetection/VPDetectionCloud.h"
//#include "VanishingPointDetection/VPDetectionContext.h"
//#include "VFRMath/VFRMath.h"
//
//#include <pcl/features/normal_3d.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include "pcl/ModelCoefficients.h"
//#include "pcl/sample_consensus/method_types.h"
//#include "pcl/sample_consensus/model_types.h"
//#include "pcl/segmentation/sac_segmentation.h"
//#include "pcl/filters/extract_indices.h"
//#include <pcl/common/angles.h>
//#include <pcl/common/common.h>
//#include "pcl/common/centroid.h"
//#include "pcl/common/eigen.h"
//#include <pcl/ModelCoefficients.h>
//#include <pcl/filters/project_inliers.h>
//#include <boost/thread/thread.hpp> 
//
//
//#include <pcl/surface/concave_hull.h>
//#include <pcl/surface/convex_hull.h>
//#include <iostream>
//#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
//#include <pcl/filters/voxel_grid.h>
//
//
////Just for test purpose
//#include "pcl/visualization/pcl_visualizer.h"
//#include <pcl/visualization/cloud_viewer.h>
//#include <boost/thread.hpp> 
//#include <boost/timer.hpp>
//#include <boost/progress.hpp>
//
//
//
//
//
////test plotting
//#include <vtkSmartPointer.h>
//
//#include <vtkChartXY.h>
//#include <vtkContextScene.h>
//#include <vtkContextView.h>
//#include <vtkFloatArray.h>
//#include <vtkPlotPoints.h>
//#include <vtkRenderWindow.h>
//#include <vtkRenderWindowInteractor.h>
//#include <vtkRenderer.h>
//#include <vtkTable.h>
//#include <vtkVRMLExporter.h>
//
//#include "Unit1.h"
//
////
////template<class RandIt, class Compare>
////bool next_k_permutation(RandIt first, RandIt mid, RandIt last, Compare comp)
////{
////	std::sort(mid, last, std::tr1::bind(comp, std::tr1::placeholders::_2
////		, std::tr1::placeholders::_1));
////	return std::next_permutation(first, last, comp);
////}
////
////
////template<class RandIt, class Compare>
////bool next_combination(RandIt first, RandIt mid, RandIt last)
////{
////	typedef typename std::iterator_traits< RandIt >::value_type value_type;
////	std::sort(mid, last, std::greater< value_type >() );
////	while(std::next_permutation(first, last)){
////		if(std::adjacent_find(first, mid, std::greater< value_type >() ) == mid){
////			return true;
////		}
////		std::sort(mid, last, std::greater< value_type >() );
////		return false;
////	}
////}
////
////template<class BiDiIt, class Compare>
////bool next_combination(BiDiIt first, BiDiIt mid, BiDiIt last, Compare comp)
////{
////	bool result;
////	do
////	{
////		result = next_k_permutation(first, mid, last, comp);
////	} while (std::adjacent_find( first, mid,
////		std::tr1::bind(comp, std::tr1::placeholders::_2
////		, std::tr1::placeholders::_1) )
////		!= mid );
////	return result;
////}
////
////template <typename Iterator>
////inline bool next_combination(const Iterator first, Iterator k, const Iterator last)
////{
////	/* Credits: Thomas Draper */
////	if ((first == last) || (first == k) || (last == k))
////		return false;
////	Iterator itr1 = first;
////	Iterator itr2 = last;
////	++itr1;
////	if (last == itr1)
////		return false;
////	itr1 = last;
////	--itr1;
////	itr1 = k;
////	--itr2;
////	while (first != itr1)
////	{
////		if (*--itr1 < *itr2)
////		{
////			Iterator j = k;
////			while (!(*itr1 < *j)) ++j;
////			std::iter_swap(itr1,j);
////			++itr1;
////			++j;
////			itr2 = k;
////			std::rotate(itr1,j,last);
////			while (last != j)
////			{
////				++j;
////				++itr2;
////			}
////			std::rotate(k,itr2,last);
////			return true;
////		}
////	}
////	std::rotate(first,k,last);
////	return false;
////} 
//
//
//
//HouseModelBuilder::~HouseModelBuilder(){}
//
//void HouseModelBuilder::setInputCloud(const PtCloudDataPtr &fInputcloud)
//{
//	m_InputCloud = fInputcloud;
//}
//
//void HouseModelBuilder::setModelParameter(const PolygonModelParamsPtr &fModelParams)
//{
//	m_ModelParams = fModelParams;
//}
//
//void HouseModelBuilder::getHouseWallSegemenst(std::vector<SegmentationResultPtr> &_HouseWallSegements)
//{
//	_HouseWallSegements = _m_outPlaneSegments;
//}
//
//void HouseModelBuilder::getVP(Eigen::VectorXf &_estimatedVP)
//{
//	_estimatedVP = _m_computedVP;
//}
//
//void HouseModelBuilder::buildModel(std::vector<VerticalEdgeLinePtr> &fVPlines, 	PtCloudDataPtr &fCloudTopPoints, PtCloudDataPtr &fCloudBottomPoints, std::vector<fWallLinkagePtr> &m_LinkageMatrix, std::vector<fWallLinkagePtr> &ValidatedLinkageMatrix)
//
//{
//
//	////test addquad!
//	//PtCloudDataPtr quad_cloud_a(new PtCloudData);
//	//quad_cloud_a->height = 1;
//	//quad_cloud_a->width = 4;
//	//quad_cloud_a->resize(quad_cloud_a->height * quad_cloud_a->width);
//
//	//quad_cloud_a->points[0].x =0.0; 
//	//quad_cloud_a->points[0].y =0.0;
//	//quad_cloud_a->points[0].z =0.0;
//
//	//quad_cloud_a->points[1].x =1.0; 
//	//quad_cloud_a->points[1].y =0.0;
//	//quad_cloud_a->points[1].z =0.0;
//
//	//quad_cloud_a->points[2].x =1.0; 
//	//quad_cloud_a->points[2].y =1.0;
//	//quad_cloud_a->points[2].z =0.0;
//
//	//quad_cloud_a->points[3].x =0.0; 
//	//quad_cloud_a->points[3].y =1.0;
//	//quad_cloud_a->points[3].z =0.0;
//
//	//PtCloudDataPtr tmpcloud(new PtCloudData);
//
//	//_FCloudHull(quad_cloud_a, tmpcloud);
//
//	//vtkSmartPointer<vtkActor> actor =
//	//	vtkSmartPointer<vtkActor>::New();
//	//AddQuadActorToRenderer(tmpcloud, actor);
//
//	//// Visualize
//	//vtkSmartPointer<vtkRenderer> renderer =
//	//	vtkSmartPointer<vtkRenderer>::New();
//	//vtkSmartPointer<vtkRenderWindow> renderWindow =
//	//	vtkSmartPointer<vtkRenderWindow>::New();
//	//renderWindow->AddRenderer(renderer);
//	//vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
//	//	vtkSmartPointer<vtkRenderWindowInteractor>::New();
//	//renderWindowInteractor->SetRenderWindow(renderWindow);
//
//	//renderer->AddActor(actor);
//	//renderer->SetBackground(.5,.3,.31); // Background color salmon
//
//	//renderWindow->Render();
//	//renderWindowInteractor->Start();
//
//	//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//	//Do the actual work here!!
//	//Get VP!
//	PtCloudDataPtr voxelgridFilteredCloud(new PtCloudData);
//	_RobustlyEstimateVP(voxelgridFilteredCloud);
//
//	std::cout<<"Estimated VP: "<<_m_computedVP[3]<<","<<_m_computedVP[4]<<","<<_m_computedVP[5]<<std::endl;
//
//	//estimate a point on the plane assuming a known height
//	float fheight = 4.0f;
//	Vec3d haus51 = {0.000, 0.000, 1.00};
//	Eigen::Vector4f fPlaneCentroid;
//	pcl::compute3DCentroid(*m_InputCloud, fPlaneCentroid);
//	float fPlaneD = -(haus51.x * (fPlaneCentroid[0] + fheight)  + haus51.y * (fPlaneCentroid[1] + fheight) + haus51.z * (fPlaneCentroid[2] + fheight));
//	Eigen::Vector4f fVPPlaneCoefs;
//	fVPPlaneCoefs.resize(4);
//	fVPPlaneCoefs[0] = haus51.x;
//	fVPPlaneCoefs[1] = haus51.y;
//	fVPPlaneCoefs[2] = haus51.z;
//	fVPPlaneCoefs[3] = fPlaneD;
//	
//
//	//save all the intersection points and render on the cloud
//	std::vector<pcl::PointXYZ> fTopIntersectionPoints;
//
//	//Get planes
//	_RobustlySegmentaPlanes();
//
//
//	//123Catch only!
//
//	testCongress();
//
//	//_end
//
//
//	//save all intersecting pairs
//	std::vector<fWallLinkagePtr> fLinkageMatrix;
//
//	//Get all planes intersecting the selected plane
//	for (unsigned int i = 0; i < _m_outPlaneSegments.size(); i++)
//	{
//		SegmentationResultPtr aPlane = _m_outPlaneSegments.at(i);
//	
//		fIntersectorsPtr fIntersectingPlanes(new fIntersectors());
//
//		for (unsigned int j = i + 1; j < _m_outPlaneSegments.size(); j ++)
//		{
//			SegmentationResultPtr bPlane = _m_outPlaneSegments.at(j);
//			Eigen::VectorXf fIntersectionLine;
//			bool ifIntersecting = _PlanePlaneIntersection(aPlane->get<0>(), bPlane->get<0>(), fIntersectionLine);
//	
//			if (ifIntersecting)
//			{
//				pcl::PointXYZ fPntOfIntersection;
//				_PlaneLineIntersection(fVPPlaneCoefs, fIntersectionLine, fPntOfIntersection);
//				fIntersectingPlanes->push_back(boost::make_tuple(boost::ref(fIntersectionLine), boost::ref(j), boost::ref(fPntOfIntersection)));
//				fTopIntersectionPoints.push_back(fPntOfIntersection);
//			}
//			
//			//std::cout<<"Plane A!!"<<std::endl;
//			//std::cout <<*(aPlane->get<0>())<<std::endl;
//
//			//std::cout<<"Plane B!!"<<std::endl;
//			//std::cout <<*(bPlane->get<0>())<<std::endl;
//
//			if (ifIntersecting)
//			{
//			//	std::cout<<"Intersecting!!"<<std::endl;
//
//
//				////_Startplot cloud 
//				//pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//
//				//viewer0.setBackgroundColor (0, 0, 0);
//				//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (aPlane->get<0>(), 0, 255, 0);
//				//viewer0.addPointCloud (aPlane->get<0>(), fOriginalCloudHandle0, "Original Cloud");
//				//viewer0.setBackgroundColor (0.0, 0.0, 0.0);
//				//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Original Cloud");
//				//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (bPlane->get<0>(),0, 0, 255);
//				//unsigned int fRen = 1;
//				///*std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);*/
//				//viewer0.addPointCloud (bPlane->get<0>(), EdgesInliersHandler0, "Edges Cloud");
//
//				//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Edges Cloud");
//				//
//				//float fLineScale = 5.0f;
//				//pcl::PointXYZ fLineStart_b;
//				//fLineStart_b.x = fIntersectionLine[0] - fLineScale* fIntersectionLine[3];
//				//fLineStart_b.y = fIntersectionLine[1] - fLineScale* fIntersectionLine[4];
//				//fLineStart_b.z = fIntersectionLine[2] - fLineScale* fIntersectionLine[5];
//
//				//pcl::PointXYZ fLineEndPoint_b;
//				//fLineEndPoint_b.x = fIntersectionLine[0] + fLineScale* fIntersectionLine[3];
//				//fLineEndPoint_b.y = fIntersectionLine[1] + fLineScale* fIntersectionLine[4];
//				//fLineEndPoint_b.z = fIntersectionLine[2] + fLineScale* fIntersectionLine[5];
//
//
//				//std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);
//				//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId2, 0); 
//				//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, fCloudId2);
//				////std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//				////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//				////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//				//viewer0.addCoordinateSystem (1.0f);
//
//				//while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//				//{
//				//	//////	range_image_widget.spinOnce ();
//				//	viewer0.spin();
//				//	//viewer0.spinOnce(1000);
//				//	boost::this_thread::sleep(boost::posix_time::seconds(0));
//				//	//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//
//				//}
//				////_Endplot
//
//			}
//			else{
//				//std::cout<<"Non-Intersecting!!"<<std::endl;
//
//				////_Startplot cloud with edges
//				//pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//
//				//viewer0.setBackgroundColor (0, 0, 0);
//				//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (aPlane->get<0>(), 0, 255, 0);
//				//viewer0.addPointCloud (aPlane->get<0>(), fOriginalCloudHandle0, "Original Cloud");
//				//viewer0.setBackgroundColor (0.0, 0.0, 0.0);
//				//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Original Cloud");
//				//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (bPlane->get<0>(),0, 0, 255);
//				//unsigned int fRen = 1;
//				///*std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);*/
//				//viewer0.addPointCloud (bPlane->get<0>(), EdgesInliersHandler0, "Edges Cloud");
//
//				//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Edges Cloud");
//
//				////pcl::PointXYZ fLineStart_b;
//				////fLineStart_b.x = fBottomLineParams[0] - fLineScale* fBottomLineParams[3];
//				////fLineStart_b.y = fBottomLineParams[1] - fLineScale* fBottomLineParams[4];
//				////fLineStart_b.z = fBottomLineParams[2] - fLineScale* fBottomLineParams[5];
//
//				////pcl::PointXYZ fLineEndPoint_b;
//				////fLineEndPoint_b.x = fBottomLineParams[0] + fLineScale* fBottomLineParams[3];
//				////fLineEndPoint_b.y = fBottomLineParams[1] + fLineScale* fBottomLineParams[4];
//				////fLineEndPoint_b.z = fBottomLineParams[2] + fLineScale* fBottomLineParams[5];
//
//
//				///*std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);*/
//				////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fCloudId2, 0); 
//				////std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//				////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//				//////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//				//viewer0.addCoordinateSystem (1.0f);
//
//				//while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//				//{
//				//	//////	range_image_widget.spinOnce ();
//				//	viewer0.spin();
//				//	//viewer0.spinOnce(1000);
//				//	boost::this_thread::sleep(boost::posix_time::seconds(0));
//				//	//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//
//				//}
//				////_Endplot
//
//			}
//		}
//		fWallLinkagePtr tmpLink(new fWallLinkage);
//		tmpLink->first = i;
//		tmpLink->second = fIntersectingPlanes;
//		fLinkageMatrix.push_back(tmpLink);
//	}
//	m_LinkageMatrix = fLinkageMatrix;
//	
//	unsigned int nIntersectionPnts = fTopIntersectionPoints.size();
//
//	std::cout<<"Number of intersections: "<< nIntersectionPnts<<std::endl;
//
//	std::vector<fWallLinkagePtr> fPreprocessedLinkageMatrix;
//	float fToleranceRadius = 0.3;//haus51 = 0.08;
//	PtCloudDataPtr ppinters(new PtCloudData);
//	unsigned int fCounterMin = 1000;
//	_RoughlyValidateIntersection(fLinkageMatrix, fPreprocessedLinkageMatrix, fToleranceRadius, ppinters, fCounterMin);
//	
//	_Print2DProjections(fPreprocessedLinkageMatrix);	
//	std::cout<<"Number of intersections after preprocessing is: "<< ppinters->points.size()<<std::endl;
//	
//	//generate all connecting edges
//	std::vector<EdgeConnectorPtr> fConnectingEdges;
//
//	boost::timer t; // start timing
//	//_FilterConnectingSegments(fPreprocessedLinkageMatrix, fConnectingEdges);
//	_FilterConnectingSegments(fLinkageMatrix, fConnectingEdges);
//	double elapsedTime = t.elapsed();
//	std::cout<< "Time used: "<<elapsedTime<<std::endl;
//	std::cout<< "Size of connecting edges: "<<fConnectingEdges.size()<<std::endl;
//
//	
//	//validate house model using BIC/GRIC/AIC/GIC
//	std::vector<EdgeConnectorPtr> fValidWalls;
//	bool isModelValid = _ValidateSegments(fConnectingEdges, fValidWalls);
//
//
//	//for congress only!
//	if (!isModelValid)
//	{
//		return;
//	}
//
//
//	PtCloudDataPtr fCloudFromIntersectionPnts(new PtCloudData);
//	fCloudFromIntersectionPnts->width = nIntersectionPnts;	
//	fCloudFromIntersectionPnts->height = 1;
//	fCloudFromIntersectionPnts->resize(fCloudFromIntersectionPnts->width * fCloudFromIntersectionPnts->height);
//
//	//fill in the data to the created cloud
//	for (unsigned int i = 0 ; i < nIntersectionPnts; i++)
//	{
//		fCloudFromIntersectionPnts->points[i] = fTopIntersectionPoints.at(i);
//	}
//
//
//	/*fCloudFromIntersectionPnts->is_dense = false;*/
//
//	//_Startplot cloud with edges
//	pcl::visualization::PCLVisualizer viewer1 ("3D Viewer");
//
//
//	viewer1.setBackgroundColor (0, 0, 0);
//	//Intersections
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fintersectionHandle0 (fCloudFromIntersectionPnts, 0, 0, 255);
//	viewer1.addPointCloud (fCloudFromIntersectionPnts, fintersectionHandle0, "intersectionCloud");
//	/*viewer1.setBackgroundColor (0.0, 0.0, 0.0);*/
//	viewer1.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "intersectionCloud");
//	
//	//preprocessed
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fppHandle (ppinters, 255, 0, 0);
//	viewer1.addPointCloud (ppinters, fppHandle, "ppIntersectionCloud");
//	/*viewer1.setBackgroundColor (0.0, 0.0, 0.0);*/
//	viewer1.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "ppIntersectionCloud");
//
//	//original cloud
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginals (m_InputCloud, 0, 255, 0);
//	unsigned int fRen = 1;
//    //std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);*/
//	viewer1.addPointCloud (m_InputCloud, fOriginals, "Original");
//	viewer1.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Original");
//
//	//pcl::PointXYZ fLineStart_b;
//	//fLineStart_b.x = fBottomLineParams[0] - fLineScale* fBottomLineParams[3];
//	//fLineStart_b.y = fBottomLineParams[1] - fLineScale* fBottomLineParams[4];
//	//fLineStart_b.z = fBottomLineParams[2] - fLineScale* fBottomLineParams[5];
//
//	//pcl::PointXYZ fLineEndPoint_b;
//	//fLineEndPoint_b.x = fBottomLineParams[0] + fLineScale* fBottomLineParams[3];
//	//fLineEndPoint_b.y = fBottomLineParams[1] + fLineScale* fBottomLineParams[4];
//	//fLineEndPoint_b.z = fBottomLineParams[2] + fLineScale* fBottomLineParams[5];
//
//
//	//std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);*/
//	//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fCloudId2, 0); 
//	//std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//	//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//	////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//	viewer1.addCoordinateSystem (1.0f);
//
//	while (!viewer1.wasStopped () /*&& range_image_widget.isShown ()*/)
//	{
//		//////	range_image_widget.spinOnce ();
//		viewer1.spin();
//		//viewer0.spinOnce(1000);
//		boost::this_thread::sleep(boost::posix_time::seconds(0));
//		//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//
//	}
//	//_Endplot
//
//
//
//
//
//
//
//
//
//	
//	//collect curvature lines
//	std::vector<VerticalEdgeLinePtr> fAllLineOptimized;
//	int fNumIterations = m_ModelParams->fVPModelParams->fNumOfIterations;
//	int fMinSupport = 10;
//_GetAllLines(voxelgridFilteredCloud, fAllLineOptimized, fNumIterations, fMinSupport);
//
//	if (fAllLineOptimized.size() == 0)
//	{
//	std::cout<<" GetALlLines() Failed, Probably SACLINE has a too small ransac threshold!"<<std::endl;
//	return;
//	}
//	
//	
//	std::vector<VerticalEdgeLinePtr> fAllLinesParallelToVector;
//	pcl::PointXYZ fDirectionVector;
//	fDirectionVector.x = _m_computedVP[3];
//	fDirectionVector.y = _m_computedVP[4];
//	fDirectionVector.z = _m_computedVP[5];
//	float fAngularTolerance = 0.1;
//	_GetLinesParallelToVector(fAllLineOptimized, fAllLinesParallelToVector, fDirectionVector, fAngularTolerance);
//	
//	if (fAllLinesParallelToVector.size() == 0)
//	{
//		std::cout<<" fAllLinesParallelToVector() Failed, Probably No Line parallel to VP found, try adjust the angular tolerance or SAC Line threshold ?!"<<std::endl;
//		return;
//	}
//
//
//	//choose only the longer lines
//	std::vector<VerticalEdgeLinePtr> fVPBestNLines;
//	//unsigned int fTopNPercent = 10; //Haus23_5=10
//	//unsigned int fTopNPercent = 95; //haus51
//	unsigned int fTopNPercent = 30; //fachwerkshaus
//	_GetNBestVerticalLines(fAllLinesParallelToVector, fVPBestNLines, fTopNPercent);
//
//	if (fVPBestNLines.size() == 0)
//	{
//		std::cout<<" Not enough long VP lines, _GetNBestVPLines() to high a percentatge limit, Probably No Line parallel to VP found, try adjust the angular tolerance or SAC Line threshold ?!"<<std::endl;
//		return;
//	}
//
//	//validate the linkage matrix with this lines of strong curvature
//	std::vector<fWallLinkagePtr> outLinkageMatrix;
//	double fConnectorTolerance = 1.0;
//	_ValidateLines(fVPBestNLines, fLinkageMatrix, outLinkageMatrix, fConnectorTolerance);
//
//	PtCloudDataPtr fHullPnts(new PtCloudData);
//	_InterpreteHull(fLinkageMatrix, fHullPnts);
//
//	ValidatedLinkageMatrix = outLinkageMatrix;
//
//	//////////Validated the line using the point distribution on the lines
//	////////Not Properly working!!!!
//	////////std::vector<VerticalEdgeLinePtr> fLinesValidates;
//	////////_ValidateLines(fAllLinesParallelToVector, fLinesValidates, 1.0);
//
//	//////Vec3dPtr fVPOptimized(new Vec3d);
//	//////fVPOptimized->x = _m_computedVP[3];  
//	//////fVPOptimized->y = _m_computedVP[4];
//	//////fVPOptimized->z = _m_computedVP[5];
//
//	//////_DetermineQuadPoints( fAllLinesParallelToVector, fVPOptimized, fCloudTopPoints, fCloudBottomPoints);
//	//////
//	//////fVPlines = fAllLinesParallelToVector;
//
//	////////Validate the lines using the number of points and or the length of the lines and consider only the most dense lines
//
//
//	////////SegmentPlanes
//	////////Collect intersections
//	////////compare distances
//}
//
//void HouseModelBuilder::ExtractModel(std::vector<Eigen::Vector4f> &fHousModelWalls)
//{
//	//write results to output
//	fHousModelWalls = m_HousModelWalls;
//}
//
//////ToDo: do things better using Eigen
////void HouseModelBuilder::_PlaneLineIntersection(const Eigen::VectorXf &fOptimPlaneCoefs, const Eigen::VectorXf &fOptimLinesCoefs, pcl::PointXYZ &fPntOfIntersection)
////{
////	//ToDo: Check if the point exist i.e the line does not lie  on the plane!
////
////	const float fNumerator = fOptimLinesCoefs[0]*fOptimPlaneCoefs[0] + fOptimLinesCoefs[1]*fOptimPlaneCoefs[1] + fOptimLinesCoefs[2]*fOptimPlaneCoefs[2] + fOptimPlaneCoefs[3];
////	const float fDenum = fOptimLinesCoefs[3]*fOptimPlaneCoefs[0] + fOptimLinesCoefs[4]*fOptimPlaneCoefs[1] + fOptimLinesCoefs[5]*fOptimPlaneCoefs[2];
////	const float ftParam = - fNumerator/fDenum;
////
////	fPntOfIntersection.x = fOptimLinesCoefs[0] + ftParam*fOptimLinesCoefs[3];
////	fPntOfIntersection.y = fOptimLinesCoefs[1] + ftParam*fOptimLinesCoefs[4];
////	fPntOfIntersection.z = fOptimLinesCoefs[2] + ftParam*fOptimLinesCoefs[5];
////}
//
//
//
//
//void HouseModelBuilder::_RobustlyEstimateVP(PtCloudDataPtr &voxelgridFilteredCloud)
//{
//	unsigned int nIterations = 3000;
//	unsigned int nCrossProds = nIterations/2;
//	unsigned int nValidationRounds = 1;
//	unsigned int nPercentageNormals = 30;
//	double fEPS = 3;
//    
//	VPModelParamsPtr tmpModelParams (new VPModelParams());
//	tmpModelParams->fNumOfIterations = nIterations;
//	tmpModelParams->fNumberOfValidationRounds = nValidationRounds;
//	tmpModelParams->fEPSInDegrees = fEPS;
//	tmpModelParams->fNumOfCrossProducts = nCrossProds;
//	tmpModelParams->fPercentageOfNormals = nPercentageNormals;
//	tmpModelParams->fVoxGridSize = m_ModelParams->fVPModelParams->fVoxGridSize;
//	tmpModelParams->m_k = m_ModelParams->fVPModelParams->m_k;
//	tmpModelParams->fRadiusSearched = m_ModelParams->fVPModelParams->fRadiusSearched;
//	m_ModelParams->fVPModelParams = tmpModelParams;
//
//	//Do the actual work here
//	VPDetectionCloud *vp = new VPDetectionCloud;
//	VPDetectionContext vpContext(vp);
//	vpContext.setInputCloud(m_InputCloud);
//	vpContext.setModelParams(tmpModelParams);
//	vpContext.validateResults(true);
//	vpContext.ComputeVPDetection();
//	vpContext.getvoxelgridedFilteredCloud(voxelgridFilteredCloud);
//	vpContext.extractVPs(_m_computedVP);
//	Eigen::VectorXf congressVP;
//		
//	//congressVP.resize(6);
//	//congressVP[0] = 0.0012131;
//	//congressVP[1] = -0.966683;
//	//congressVP[2] = 0.255974;
//	//congressVP[3] = 0.0012131;
//	//congressVP[4] = -0.966683;
//	//congressVP[5] = 0.255974;
//
//	//congressVP.resize(6);
//	//congressVP[0] = 0.0030848;
//	//congressVP[1] = -0.999579;
//	//congressVP[2] = -0.0288597;
//	//congressVP[3] = 0.0030848;
//	//congressVP[4] = -0.999579;
//	//congressVP[5] = -0.0288597;
//
//	
//
//	//congressVP.resize(6);
//	//congressVP[0] = 0.0;
//	//congressVP[1] = 0.0;
//	//congressVP[2] = 1.0;
//	//congressVP[3] = 0.0;
//	//congressVP[4] = 0.0;
//	//congressVP[5] = 1.0;
//
//
//	//_m_computedVP = congressVP;//[0] = 0.0;
//	delete vp;
//}
//
//void HouseModelBuilder::_RobustlySegmentaPlanes()
//{
//	unsigned int nIterations = 2000;
//	//Extract Model Parameters from GUI Elements
//	SegModelParamsPtr tmpModelParams(new SegModelParams);
//	tmpModelParams->fNumOfIterations = nIterations;
//	tmpModelParams->fMinNumInliers = 100;
//	tmpModelParams->fMaxNumModels = 4;//8;
//
//	tmpModelParams->fRansacThresh = m_ModelParams->fSegModelParams->fRansacThresh;
//	tmpModelParams->fVoxGridSize = m_ModelParams->fSegModelParams->fVoxGridSize;
//	
//	//collect ppnormal params
//	tmpModelParams->fPPNormal(0) = _m_computedVP[3];
//	tmpModelParams->fPPNormal(1) = _m_computedVP[4];
//	tmpModelParams->fPPNormal(2) = _m_computedVP[5];
//
//	//Check if voxel grid is needed otherwise do not down sample things!
//	PlaneModel *p = new PlaneModel;
//	ModelContext fContext(p);
//	fContext.setInputCloud(m_InputCloud);
//	fContext.setModelParams(tmpModelParams);
//	fContext.ExecuteSegmentation();
//	fContext.extractSegments(_m_outPlaneSegments);
//	delete p;
//}
//
//
//
//
////ToDo: use indices instead of cloud type: ptr
//void HouseModelBuilder::_GetCloudEdges(const PtCloudDataPtr &fInCloud, const unsigned int fTopnPercent, const int kNeigbouhood, PtCloudDataPtr &fEdgesCloud)
//{
//	////int fTopnPercent = 70;
//	//int fTopnPercent = 15;//haus23_5=15//unichurch=15; //haus51=15
//	//int fKNeighbourhood = 13; //haus23_5=7//unichurch=20; //haus51=6 
//	//PtCloudDataPtr fCloudWithEdges(new PtCloudData);
//	//bool fFilterin = false;
//	//GetCloudEdges(fTopnPercent, fKNeighbourhood, fCloudWithEdges, false);
//
//
//	unsigned int nPointsCandidates = fInCloud->points.size();
//	//Normal estimation*
//	/*pcl::NormalEstimationOMP fd;*/ // 6-8 times faster ?
//	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
//	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
//	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//	tree->setInputCloud (fInCloud);
//	n.setInputCloud (fInCloud);
//	n.setSearchMethod (tree);
//	n.setKSearch (m_ModelParams->fVPModelParams->m_k);
//	n.compute (*normals);
//
//	std::vector<float> fPrincipalCurvatures;
//	std::vector<float> fPrincipalCurvaturesTmp; //for computing a good value for the max curvature value
//
//	float fMaxCurvature = 0.0f;
//
//	for (unsigned int i= 0; i < nPointsCandidates; i++)
//	{
//		fPrincipalCurvatures.push_back(normals->points[i].curvature);
//		fPrincipalCurvaturesTmp.push_back(normals->points[i].curvature);
//		//if(fMaxCurvature < normals->points[i].curvature)
//		//	fMaxCurvature = normals->points[i].curvature;
//	}
//
//	//Get a good estimate of the highest curvature by smoothing over a range
//	std::sort(fPrincipalCurvaturesTmp.begin(),fPrincipalCurvaturesTmp.end(),[](float a, float b){return a > b;});
//	int fSmoother = 0;
//	for (unsigned int i= 10; i < 50; i++)
//	{
//		//fPrincipalCurvatures.push_back(normals->points[i].curvature);
//		//fPrincipalCurvatures.at(i) = fPrincipalCurvatures.at(i)/fMaxCurvature;
//		fMaxCurvature+=fPrincipalCurvaturesTmp.at(i);
//		/*if(fMaxCurvature > normals->points[i].curvature)
//		fMaxCurvature = normals->points[i].curvature;*/
//		fSmoother++;
//	}
//
//	fMaxCurvature = fMaxCurvature/fSmoother;
//
//
//	//for_each(fPrincipalCurvatures.begin(), fPrincipalCurvatures.end(),[&](float a){ fPrincipalCurvatures});
//	//normalise curvature value between (1-0)
//	//for (unsigned int i= 0; i < nPointsCandidates; i++)
//	//{
//	//	//fPrincipalCurvatures.push_back(normals->points[i].curvature);
//	//	fPrincipalCurvatures.at(i) = fPrincipalCurvatures.at(i)/fMaxCurvature;
//	//	/*if(fMaxCurvature > normals->points[i].curvature)
//	//	fMaxCurvature = normals->points[i].curvature;*/
//	//}
//
//	////score with the topn%
//	//const float fpercentageLimit = 1.0f - float(fTopnPercent)/100;
//	//const float fpercentageLimit = float(fTopnPercent)/100;
//	unsigned int fCounts = 0;
//
//	std::vector<unsigned int> fEdges;
//	for (unsigned int i = 0; i < nPointsCandidates; i++)
//	{
//		if (fPrincipalCurvatures.at(i) > fMaxCurvature*float(fTopnPercent)/100)
//		{
//			fCounts++;
//			fEdges.push_back(i);
//		}
//	}
//
//	PtCloudDataPtr mfToColor(new PtCloudData);
//
//	mfToColor->height = 1;
//	mfToColor->width = fEdges.size();
//	mfToColor->resize(mfToColor->width * mfToColor->height);
//
//	for(unsigned int i = 0; i < fEdges.size(); i++)
//	{
//		mfToColor->points[i] = fInCloud->points[fEdges.at(i)];
//	}
//
//	////remove outliers
//	//PtCloudDataPtr mfToColorFiltered(new PtCloudData);
//	//if (fDoRadiusFiltering)
//	//{
//	//	RadiusOutlierFiltering(mfToColor, mfToColorFiltered, fRadiusKNeighbourhood, fMinFilteringRadius);
//	//	fEdgesCloud = mfToColorFiltered;
//	//}else 
//	//	fEdgesCloud = mfToColor;
//
//
//	//unsigned int nPointsCandidates = m_wPclDataPtr->points.size();
//	////Normal estimation*
//	//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
//	//pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
//	//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//	//tree->setInputCloud (m_wPclDataPtr);
//	//n.setInputCloud (m_wPclDataPtr);
//	//n.setSearchMethod (tree);
//	//n.setKSearch (kNeigbouhood);
//	//n.compute (*normals);
//
//	//std::vector<float> fPrincipalCurvatures;
//	//std::vector<float> fPrincipalCurvaturesTmp; //for computing a good value for the max curvature value
//
//	//float fMaxCurvature = 0.0f;
//
//	//for (unsigned int i= 0; i < nPointsCandidates; i++)
//	//{
//	//	fPrincipalCurvatures.push_back(normals->points[i].curvature);
//	//	fPrincipalCurvaturesTmp.push_back(normals->points[i].curvature);
//	//	//if(fMaxCurvature < normals->points[i].curvature)
//	//	//	fMaxCurvature = normals->points[i].curvature;
//	////}
//
//	//	if(fMaxCurvature < normals->points[i].curvature)
//	//		fMaxCurvature = normals->points[i].curvature;
//	//}
//	////for_each(fPrincipalCurvatures.begin(), fPrincipalCurvatures.end(),[&](float a){ fPrincipalCurvatures});
//	////normalise curvature value between (1-0)
//	//for (unsigned int i= 0; i < nPointsCandidates; i++)
//	////Get a good estimate of the highest curvature by smoothing over a range
//	//std::sort(fPrincipalCurvaturesTmp.begin(),fPrincipalCurvaturesTmp.end(),[](float a, float b){return a > b;});
//	//int fSmoother = 0;
//	//for (unsigned int i= 10; i < 50; i++)
//	//{
//	//	//fPrincipalCurvatures.push_back(normals->points[i].curvature);
//	//	fPrincipalCurvatures.at(i) = fPrincipalCurvatures.at(i)/fMaxCurvature;
//	//	//fPrincipalCurvatures.at(i) = fPrincipalCurvatures.at(i)/fMaxCurvature;
//	//	fMaxCurvature+=fPrincipalCurvaturesTmp.at(i);
//	//	/*if(fMaxCurvature > normals->points[i].curvature)
//	//	fMaxCurvature = normals->points[i].curvature;*/
//	//	fSmoother++;
//	//}
//
//	////score with the topn%
//	//const float fpercentageLimit = 1.0f - float(fTopnPercent)/100;
//	//fMaxCurvature = fMaxCurvature/fSmoother;
//
//
//	////for_each(fPrincipalCurvatures.begin(), fPrincipalCurvatures.end(),[&](float a){ fPrincipalCurvatures});
//	////normalise curvature value between (1-0)
//	////for (unsigned int i= 0; i < nPointsCandidates; i++)
//	////{
//	////	//fPrincipalCurvatures.push_back(normals->points[i].curvature);
//	////	fPrincipalCurvatures.at(i) = fPrincipalCurvatures.at(i)/fMaxCurvature;
//	////	/*if(fMaxCurvature > normals->points[i].curvature)
//	////	fMaxCurvature = normals->points[i].curvature;*/
//	////}
//
//	//////score with the topn%
//	////const float fpercentageLimit = 1.0f - float(fTopnPercent)/100;
//	////const float fpercentageLimit = float(fTopnPercent)/100;
//	//unsigned int fCounts = 0;
//
//	//std::vector<unsigned int> fEdges;
//	//for (unsigned int i = 0; i < nPointsCandidates; i++)
//	//{
//	//	if (fPrincipalCurvatures.at(i) > fpercentageLimit)
//	//	if (fPrincipalCurvatures.at(i) > fMaxCurvature*float(fTopnPercent)/100)
//	//	{
//	//		fCounts++;
//	//		fEdges.push_back(i);
//	//	}
//	//}
//
//	//PtCloudDataPtr mfToColor(new PtCloudData);
//
//	//mfToColor->height = 1;
//	//mfToColor->width = fEdges.size();
//	//mfToColor->resize(mfToColor->width * mfToColor->height);
//
//	//for(unsigned int i = 0; i < fEdges.size(); i++)
//	//{
//	//	mfToColor->points[i] = m_wPclDataPtr->points[fEdges.at(i)];
//	//}
//
//	////remove outliers
//	//PtCloudDataPtr mfToColorFiltered(new PtCloudData);
//	//if (fDoRadiusFiltering)
//	//{
//	//	RadiusOutlierFiltering(mfToColor, mfToColorFiltered, fRadiusKNeighbourhood, fMinFilteringRadius);
//	//	fEdgesCloud = mfToColorFiltered;
//	//}else 
//		fEdgesCloud = mfToColor;
//}
//
//
//
////ToDO: Change verticalEdgeLine to CornerEdgeLine for the general case
//void HouseModelBuilder::_GetAllLines(const PtCloudDataPtr &fInCloud, std::vector<VerticalEdgeLinePtr> &fAllLineOptimized, const int fNumIterations, const int fMinSupport)
//{
//
//	//do smoothing using moving least squares smoother
//
//
//	//int fTopnPercent = 70;
//	int fTopnPercent = 15;//haus23_5=15//unichurch=15; //haus51=15
//	int fKNeighbourhood = m_ModelParams->fVPModelParams->m_k; //haus23_5=7//unichurch=20; //haus51=6 
//	PtCloudDataPtr fCloudWithEdges(new PtCloudData);
//	_GetCloudEdges(fInCloud, fTopnPercent, fKNeighbourhood, fCloudWithEdges);
//
//		//_Startplot cloud with edges
//		pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//	
//	
//		viewer0.setBackgroundColor (0, 0, 0);
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (fInCloud, 0, 255, 0);
//		viewer0.addPointCloud (fInCloud, fOriginalCloudHandle0, "Original Cloud");
//		viewer0.setBackgroundColor (0.0, 0.0, 0.0);
//		viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Original Cloud");
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (fCloudWithEdges,0, 0, 255);
//		unsigned int fRen = 1;
//		/*std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);*/
//		viewer0.addPointCloud (fCloudWithEdges, EdgesInliersHandler0, "Edges Cloud");
//	
//		viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Edges Cloud");
//	
//		//pcl::PointXYZ fLineStart_b;
//		//fLineStart_b.x = fBottomLineParams[0] - fLineScale* fBottomLineParams[3];
//		//fLineStart_b.y = fBottomLineParams[1] - fLineScale* fBottomLineParams[4];
//		//fLineStart_b.z = fBottomLineParams[2] - fLineScale* fBottomLineParams[5];
//	
//		//pcl::PointXYZ fLineEndPoint_b;
//		//fLineEndPoint_b.x = fBottomLineParams[0] + fLineScale* fBottomLineParams[3];
//		//fLineEndPoint_b.y = fBottomLineParams[1] + fLineScale* fBottomLineParams[4];
//		//fLineEndPoint_b.z = fBottomLineParams[2] + fLineScale* fBottomLineParams[5];
//	
//	
//		/*std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);*/
//		//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fCloudId2, 0); 
//		//std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//		//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//		////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//		viewer0.addCoordinateSystem (1.0f);
//	
//		while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//		{
//			//////	range_image_widget.spinOnce ();
//			viewer0.spin();
//			//viewer0.spinOnce(1000);
//			boost::this_thread::sleep(boost::posix_time::seconds(0));
//			//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//	
//		}
//	//_Endplot
//
//
//	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
//	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
//	// Create the segmentation object
//	pcl::SACSegmentation<pcl::PointXYZ> seg;
//	// Optional
//	seg.setOptimizeCoefficients (true);
//	// Mandatory
//	seg.setModelType (pcl::SACMODEL_LINE);
//	seg.setMethodType (pcl::SAC_RANSAC);
//	seg.setMaxIterations (m_ModelParams->fSegModelParams->fNumOfIterations);
//	float fLineThreshold = m_ModelParams->fSegModelParams->fRansacThresh;
//	seg.setDistanceThreshold (fLineThreshold);//0.015=fachwerkshaus //haus23=0.09//unibwchurch=0.005 //haus51=0.10 - 0.18
//
//	// Create the filtering object
//	pcl::ExtractIndices<pcl::PointXYZ> extract;
//
//	int nr_points = (int) fCloudWithEdges->points.size ();
//
//	// While 30% of the original cloud is still there
//	while (fCloudWithEdges->points.size() > 0.001 * nr_points)
//	{
//		// Segment the largest planar component from the remaining cloud
//		seg.setInputCloud (fCloudWithEdges);
//		seg.segment (*inliers, *coefficients);
//		if (inliers->indices.size () < fMinSupport)
//		{
//			std::cerr << "Could not estimate a linear model for the given dataset." << std::endl;
//			break;
//		}
//
//		// Extract the inliers
//		PtCloudDataPtr cloud_p (new pcl::PointCloud<pcl::PointXYZ>);
//		PtCloudDataPtr fProjectedLineInliers (new pcl::PointCloud<pcl::PointXYZ>);
//		extract.setInputCloud (fCloudWithEdges);
//		extract.setIndices (inliers);
//		extract.setNegative (false);
//		extract.filter (*cloud_p);
//		std::cerr << "PointCloud representing the linear component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;
//
//		//fit line using ls approach
//		Eigen::VectorXf fALinesOptimisedCoefs;
//		_LSLineFitting(cloud_p, fALinesOptimisedCoefs);
//
//
//		//Check the direction w.r.t VP
//		//get the angle between both lines and vote
//		Eigen::Vector4f fChoosenLine(fALinesOptimisedCoefs[3], fALinesOptimisedCoefs[4], fALinesOptimisedCoefs[5], 0);
//		Eigen::Vector4f fRefDir(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//
//		double fAngleInRad = pcl::getAngle3D(fChoosenLine, fRefDir);
//		double fAngleVal = pcl::rad2deg(fAngleInRad) ;
//		std::cout<<"Angle in Degrees between line and Ref direction: "<<fAngleVal<<std::endl;
//
//		Vec3dPtr DirectionA(new Vec3d);
//		DirectionA->x = fALinesOptimisedCoefs[3];
//		DirectionA->y = fALinesOptimisedCoefs[4];
//		DirectionA->z = fALinesOptimisedCoefs[5];
//
//		Vec3dPtr DirectionB(new Vec3d);
//		DirectionB->x = _m_computedVP[3];
//		DirectionB->y = _m_computedVP[4];
//		DirectionB->z = _m_computedVP[5];
//
//		double fAnleTol = pcl::deg2rad(0.5);
//		std::cout<<" Angle in Radiens between both: "<<fAngleInRad <<std::endl;
//
//		////boost::this_thread::sleep(boost::posix_time::seconds(1));
//
//		//if(VFRMath::IsParallel(DirectionA, DirectionB, 0.1))
//		//{
//
//			//do a soft radius filtering on the line might help!
//
//			std::pair<pcl::PointXYZ, pcl::PointXYZ> fLinePnts;
//			float fLineLenght = _MaxWideOfCloud(cloud_p, fLinePnts, fProjectedLineInliers);
//			fLineLenght = _MaxWideOfCloud(cloud_p); //A do around patche, To Clean!
//
//			VerticalEdgeLine fTEdgeLine;
//			fTEdgeLine.fOptimLinesCoefs = fALinesOptimisedCoefs;
//			fTEdgeLine.fLinePoints = fLinePnts;
//			fTEdgeLine.fLineLength = fLineLenght;
//			fTEdgeLine.fLineInliers = cloud_p;
//			fTEdgeLine.fProjectedToModel = fProjectedLineInliers;
//			VerticalEdgeLinePtr fTmpEdgeLinePtr(new VerticalEdgeLine(fTEdgeLine));
//
//			//save the fitted line;
//			fAllLineOptimized.push_back(fTmpEdgeLinePtr);
//
//			//create a filtering object and extract the reset
//			PtCloudDataPtr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
//			extract.setNegative (true);
//			extract.filter (*cloud_filtered);
//			*fCloudWithEdges = *cloud_filtered;
//
//		//	if (fAllLineOptimized.size() > 10)
//		//	{
//		//		break;
//		//	}
//		//}//_isparallel
//	}
//
//	std::cout<<"Finished Extracting all lines"<<std::endl;
//}
//
//
////ToDo:put this inside vfrmath
//void HouseModelBuilder::_LSLineFitting(const PtCloudDataPtr &fInClouds, Eigen::VectorXf &optimized_coefficients)
//{
//	// Needs a valid set of model coefficients
//	//if (!isModelValid (model_coefficients))
//	//{
//	//	optimized_coefficients = model_coefficients;
//	//	return;
//	//}
//
//	// Need at least 2 points to estimate a line
//	if (fInClouds->points.size () <= 2)
//	{
//		std::cout<<"Needs at least 3 points to compute the covariance matrix!..."<<std::endl;
//		return;
//	}
//
//
//	optimized_coefficients.resize (6);
//
//	// Compute the 3x3 covariance matrix
//	Eigen::Vector4f centroid;
//	pcl::compute3DCentroid (*fInClouds, centroid);
//
//
//	Eigen::Matrix3f covariance_matrix;
//	//pcl::computeCovarianceMatrix(*input_, inliers, centroid, covariance_matrix);
//	pcl::computeCovarianceMatrix(*fInClouds, centroid, covariance_matrix);
//	optimized_coefficients[0] = centroid[0];
//	optimized_coefficients[1] = centroid[1];
//	optimized_coefficients[2] = centroid[2];
//
//	// Extract the eigenvalues and eigenvectors
//	EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
//	EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
//	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);
//
//	optimized_coefficients.template tail<3> () = eigen_vectors.col (2).normalized ();
//}
//
//
//
////ToDO: Put into vfrcommon or so 
//float HouseModelBuilder::_MaxWideOfCloud(const PtCloudDataPtr &fLineCloud, std::pair<pcl::PointXYZ, pcl::PointXYZ> &fLineEndingPnts, PtCloudDataPtr &fCloudProjectedOnLine)
//{
//	unsigned int nPointsPresent = fLineCloud->points.size();
//
//	if (nPointsPresent <2)
//	{
//		std::cout<<" Input points not enough!"<<std::endl;
//		return 0;
//		return 0.0f;
//	}
//
//	Eigen::VectorXf fOptimLinesCoefs;
//	_LSLineFitting(fLineCloud, fOptimLinesCoefs);
//
//	PtCloudDataPtr fProjectedLineCloud(new PtCloudData);
//	fProjectedLineCloud->width = nPointsPresent;
//	fProjectedLineCloud->height = 1;
//	fProjectedLineCloud->resize(fProjectedLineCloud->width * fProjectedLineCloud->height);
//
//	for(unsigned int i = 0; i < nPointsPresent; i++)
//	{
//		_ProjectPointOnLine(fLineCloud->points[i], fOptimLinesCoefs,fProjectedLineCloud->points[i]);
//	}
//
//	fCloudProjectedOnLine = fProjectedLineCloud;
//
//	float fLengthMax = 0;
//	for (unsigned int i = 0 ; i < nPointsPresent; i++)
//	{
//		for (unsigned int j=i+1; j <nPointsPresent; j++)
//		{
//			const float fTmpDist = pcl::euclideanDistance(fProjectedLineCloud->points[i], fProjectedLineCloud->points[j]);
//			//const float fTmpDistA = pcl::euclideanDistance(fLineCloud->points[i], fLineCloud->points[j]);
//			//std::cout<<"ProjectedCloud:    "<<fTmpDist<<std::endl;
//			//std::cout<<"NonProjectedCloud: "<<fTmpDistA<<std::endl;
//			//boost::this_thread::sleep(boost::posix_time::seconds(2));
//			if (fTmpDist > fLengthMax)
//			{
//				fLengthMax = fTmpDist;
//				fLineEndingPnts = std::make_pair<pcl::PointXYZ, pcl::PointXYZ>(fProjectedLineCloud->points[i], fProjectedLineCloud->points[j]);
//			}
//		}
//	}
//
//
//	//pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//	//viewer0.setBackgroundColor (0, 0, 0);
//	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (fLineCloud, 255, 0, 255);
//	//viewer0.addPointCloud (fLineCloud, fOriginalCloudHandle0, "Original Cloud");
//	//viewer0.setBackgroundColor (0.3, 0.3, 0.3);
//	//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Original Cloud");
//
//	////_test_N
//
//	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (fProjectedLineCloud, 0, 255, 0);
//	//unsigned int fRen = 1;
//	//std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);
//	//viewer0.addPointCloud (fProjectedLineCloud, EdgesInliersHandler0, fwall_id);
//
//	//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fwall_id);
//
//	//float fLineScale = 3.0f;
//
//	//pcl::PointXYZ fLineStart;
//	//fLineStart.x = fOptimLinesCoefs[0] - fLineScale* fOptimLinesCoefs[3];
//	//fLineStart.y = fOptimLinesCoefs[1] - fLineScale* fOptimLinesCoefs[4];
//	//fLineStart.z = fOptimLinesCoefs[2] - fLineScale* fOptimLinesCoefs[5];
//
//
//
//	//pcl::PointXYZ fLineEndPoint;
//	//fLineEndPoint.x = fOptimLinesCoefs[0] + fLineScale* fOptimLinesCoefs[3];
//	//fLineEndPoint.y = fOptimLinesCoefs[1] + fLineScale* fOptimLinesCoefs[4];
//	//fLineEndPoint.z = fOptimLinesCoefs[2] + fLineScale* fOptimLinesCoefs[5];
//
//	////pcl::PointXYZ fLineStart_b;
//	////fLineStart_b.x = fBottomLineParams[0] - fLineScale* fBottomLineParams[3];
//	////fLineStart_b.y = fBottomLineParams[1] - fLineScale* fBottomLineParams[4];
//	////fLineStart_b.z = fBottomLineParams[2] - fLineScale* fBottomLineParams[5];
//
//	////pcl::PointXYZ fLineEndPoint_b;
//	////fLineEndPoint_b.x = fBottomLineParams[0] + fLineScale* fBottomLineParams[3];
//	////fLineEndPoint_b.y = fBottomLineParams[1] + fLineScale* fBottomLineParams[4];
//	////fLineEndPoint_b.z = fBottomLineParams[2] + fLineScale* fBottomLineParams[5];
//
//
//	//std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);
//	////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fCloudId2, 0); 
//	////std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//	////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//	//////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//	//viewer0.addCoordinateSystem (1.0f);
//
//	//while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//	//{
//	//	//////	range_image_widget.spinOnce ();
//	//	viewer0.spin();
//	//	//viewer0.spinOnce(1000);
//	//	boost::this_thread::sleep(boost::posix_time::seconds(100));
//	//	//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//
//	//}
//	return fLengthMax;
//}
//
//
//
//float HouseModelBuilder::_MaxWideOfCloud(const PtCloudDataPtr &fLineCloud)
//{
//	unsigned int nPointsPresent = fLineCloud->points.size();
//
//	if (nPointsPresent <2)
//	{
//		std::cout<<" Input points not enough!"<<std::endl;
//		return 0;
//		return 0.0f;
//	}
//
//	Eigen::VectorXf fOptimLinesCoefs;
//	_LSLineFitting(fLineCloud, fOptimLinesCoefs);
//
//	PtCloudDataPtr fProjectedLineCloud(new PtCloudData);
//	fProjectedLineCloud->width = nPointsPresent;
//	fProjectedLineCloud->height = 1;
//	fProjectedLineCloud->resize(fProjectedLineCloud->width * fProjectedLineCloud->height);
//
//	for(unsigned int i = 0; i < nPointsPresent; i++)
//	{
//		_ProjectPointOnLine(fLineCloud->points[i], fOptimLinesCoefs,fProjectedLineCloud->points[i]);
//	}
//
//	float fLengthMax = 0;
//	for (unsigned int i = 0 ; i < nPointsPresent; i++)
//	{
//		for (unsigned int j=i+1; j <nPointsPresent; j++)
//		{
//			const float fTmpDist = pcl::euclideanDistance(fProjectedLineCloud->points[i], fProjectedLineCloud->points[j]);
//			const float fTmpDistA = pcl::euclideanDistance(fLineCloud->points[i], fLineCloud->points[j]);
//			//std::cout<<"ProjectedCloud:    "<<fTmpDist<<std::endl;
//			//std::cout<<"NonProjectedCloud: "<<fTmpDistA<<std::endl;
//			//boost::this_thread::sleep(boost::posix_time::seconds(2));
//			if (fTmpDist > fLengthMax)
//				fLengthMax = fTmpDist;
//		}
//	}
//
//
//	//pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//	//viewer0.setBackgroundColor (0, 0, 0);
//	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (fLineCloud, 255, 0, 255);
//	//viewer0.addPointCloud (fLineCloud, fOriginalCloudHandle0, "Original Cloud");
//	//viewer0.setBackgroundColor (0.3, 0.3, 0.3);
//	//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Original Cloud");
//
//	////_test_N
//
//	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (fProjectedLineCloud, 0, 255, 0);
//	//unsigned int fRen = 1;
//	//std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);
//	//viewer0.addPointCloud (fProjectedLineCloud, EdgesInliersHandler0, fwall_id);
//
//	//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fwall_id);
//
//	//float fLineScale = 3.0f;
//
//	//pcl::PointXYZ fLineStart;
//	//fLineStart.x = fOptimLinesCoefs[0] - fLineScale* fOptimLinesCoefs[3];
//	//fLineStart.y = fOptimLinesCoefs[1] - fLineScale* fOptimLinesCoefs[4];
//	//fLineStart.z = fOptimLinesCoefs[2] - fLineScale* fOptimLinesCoefs[5];
//
//
//
//	//pcl::PointXYZ fLineEndPoint;
//	//fLineEndPoint.x = fOptimLinesCoefs[0] + fLineScale* fOptimLinesCoefs[3];
//	//fLineEndPoint.y = fOptimLinesCoefs[1] + fLineScale* fOptimLinesCoefs[4];
//	//fLineEndPoint.z = fOptimLinesCoefs[2] + fLineScale* fOptimLinesCoefs[5];
//
//	////pcl::PointXYZ fLineStart_b;
//	////fLineStart_b.x = fBottomLineParams[0] - fLineScale* fBottomLineParams[3];
//	////fLineStart_b.y = fBottomLineParams[1] - fLineScale* fBottomLineParams[4];
//	////fLineStart_b.z = fBottomLineParams[2] - fLineScale* fBottomLineParams[5];
//
//	////pcl::PointXYZ fLineEndPoint_b;
//	////fLineEndPoint_b.x = fBottomLineParams[0] + fLineScale* fBottomLineParams[3];
//	////fLineEndPoint_b.y = fBottomLineParams[1] + fLineScale* fBottomLineParams[4];
//	////fLineEndPoint_b.z = fBottomLineParams[2] + fLineScale* fBottomLineParams[5];
//
//
//	//std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);
//	////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fCloudId2, 0); 
//	////std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//	////viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//	//////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//	//viewer0.addCoordinateSystem (1.0f);
//
//	//while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//	//{
//	//	//////	range_image_widget.spinOnce ();
//	//	viewer0.spin();
//	//	//viewer0.spinOnce(1000);
//	//	boost::this_thread::sleep(boost::posix_time::seconds(100));
//	//	//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//
//	//}
//	return fLengthMax;
//}
//
//
//
//
//float HouseModelBuilder::_meanInterPntDistance(const PtCloudDataPtr &fInCloud, unsigned int fNumPntsConsidered)
//{
//	unsigned int fSmoother = 200;
//	float fSumDist = 0.0f;
//	for (unsigned int i = 0; i < fSmoother; i++)
//	{
//		unsigned int fIndx1 = rand() % fNumPntsConsidered;
//		unsigned int fIndx2;
//		do { fIndx2 = rand() % fNumPntsConsidered; } while (fIndx2 == fIndx1);
//		const float fDist = pcl::euclideanDistance(fInCloud->points[fIndx1], fInCloud->points[fIndx2]);
//		fSumDist += fDist;
//	}
//	return fSumDist/fSmoother;
//}
//
//
////toDo: clean and put in pcl
//void HouseModelBuilder::_ProjectPointOnLine(const pcl::PointXYZ &pPoint, const Eigen::VectorXf fOptimLinesCoefs, pcl::PointXYZ &qPoint)
//{
//
//	// Iterate through the 3d points and calculate the distances from them to the line
//	//for (size_t i = 0; i < inliers.size (); ++i)
//	//{
//	// Obtain the line point and direction
//	Eigen::Vector4f line_pt  (fOptimLinesCoefs[0], fOptimLinesCoefs[1], fOptimLinesCoefs[2], 0);
//	Eigen::Vector4f line_dir (fOptimLinesCoefs[3], fOptimLinesCoefs[4], fOptimLinesCoefs[5], 0);
//		
//	Eigen::Vector4f pt (pPoint.x, pPoint.y, pPoint.z, 0);
//		// double k = (DOT_PROD_3D (points[i], p21) - dotA_B) / dotB_B;
//		float k = (pt.dot (line_dir) - line_pt.dot (line_dir)) / line_dir.dot (line_dir);
//
//		Eigen::Vector4f pp = line_pt + k * line_dir;
//		// Calculate the projection of the point on the line (pointProj = A + k * B)
//		qPoint.x = pp[0];
//		qPoint.y = pp[1];
//		qPoint.z = pp[2];
//
//	//
//	////ray = p2 - p1; // find direction from p1 to p2
//	////rel = p - p1; // find position relative to p1
//	////n = normalize(ray); // create ray normal
//	////l = dot(n, rel); // calculate dot
//	////result = p1 + n * l; // convert back into world space
//
//	//Vec3d fPnt = {pPoint.x, pPoint.y, pPoint.z};
//	//Vec3d fLineCentroid = {fOptimLinesCoefs[0], fOptimLinesCoefs[1], fOptimLinesCoefs[2]};
//	//Vec3d fLineDir = {fOptimLinesCoefs[3], fOptimLinesCoefs[4], fOptimLinesCoefs[5]};
//	//Math3d::NormalizeVec(fLineDir);//create ray normal
//	//Vec3d fRes1;
//	//Math3d::SubtractVecVec(fLineCentroid, fPnt, fRes1);
//
//	//float fDistance = Math3d::ScalarProduct(fRes1, fLineDir);// calculate dot
//	//Vec3d fMult;
//	//Math3d::MulVecScalar(fLineDir, fDistance, fMult);
//	//Vec3d fRes2;
//	//Math3d::AddVecVec(fLineCentroid, fMult, fRes2);
//	//
//	//qPoint.x = fRes2.x;
//	//qPoint.y = fRes2.y;
//	//qPoint.z = fRes2.z;
//
//	/*projectPoint (const Point &p, const Eigen::Vector4f &model_coefficients, Point &q)
//	//{*/
//	//	// Calculate the distance from the point to the plane
//	//	Eigen::Vector4f pp (pPoint.x, pPoint.y, pPoint.z, 1);
//	//	Eigen::Vector4f pp_centroid (fOptimLinesCoefs[3], fOptimLinesCoefs[4], fOptimLinesCoefs[5], 1);
//
//	//	// use normalized coefficients to calculate the scalar projection 
//	//	float distance_to_plane = pp.dot(model_coefficients);
//
//	//	//TODO: Why doesn't getVector4Map work here?
//	//	//Eigen::Vector4f q_e = q.getVector4fMap ();
//	//	//q_e = pp - model_coefficients * distance_to_plane;
//
//	//	Eigen::Vector4f q_e = pp - distance_to_plane * model_coefficients;
//	//	q.x = q_e[0];
//	//	q.y = q_e[1];
//	//	q.z = q_e[2];
//	
//}
//
//
//void HouseModelBuilder::_GetLinesParallelToVector(const std::vector<VerticalEdgeLinePtr> &fAllLinesOptimized, std::vector<VerticalEdgeLinePtr> &fAllLinesParallelToVector, pcl::PointXYZ &fDirectionVector, float fAngularTolerance)
//{
//	unsigned int nLineCandidates = fAllLinesOptimized.size();
//	Eigen::Vector4f fRefDirection (fDirectionVector.x, fDirectionVector.y, fDirectionVector.z, 0);
//
//
//	for (unsigned int i = 0; i < nLineCandidates; i++)
//	{
//
//		const VerticalEdgeLine &aChosenLine = *(fAllLinesOptimized.at(i));
//		const Eigen::VectorXf &fLine1 = aChosenLine.fOptimLinesCoefs;
//		Eigen::Vector4f fChosenLine (fLine1[3], fLine1[4], fLine1[5], 0);
//
//		//get the angle between both lines and vote
//		double fAngleInRad = pcl::getAngle3D(fChosenLine, fRefDirection);
//		double fAngleVal = pcl::rad2deg(fAngleInRad) ;
//		std::cout<<"Angle in Degrees between line and Ref direction: "<<fAngleVal<<std::endl;
//
//		Vec3dPtr DirectionA(new Vec3d);
//		DirectionA->x = fLine1[3];
//		DirectionA->y = fLine1[4];
//		DirectionA->z = fLine1[5];
//
//		Vec3dPtr DirectionB(new Vec3d);
//		DirectionB->x = fDirectionVector.x;
//		DirectionB->y = fDirectionVector.y;
//		DirectionB->z = fDirectionVector.z;
//
//		double fAnleTol = pcl::deg2rad(fAngularTolerance);
//		std::cout<<" Angle in Radiens between both: "<<fAngleInRad <<std::endl;
//
//		//boost::this_thread::sleep(boost::posix_time::seconds(1));
//
//		if(VFRMath::IsParallel(DirectionA, DirectionB, fAnleTol))
//			fAllLinesParallelToVector.push_back(fAllLinesOptimized.at(i));
//	}
//}
//
//
//void HouseModelBuilder::_RoughlyValidateIntersection(const std::vector<fWallLinkagePtr> &fInLinkageMatrix, std::vector<fWallLinkagePtr> &fPreprocessedLinkageMatrix, const float &fToleranceRadius, PtCloudDataPtr &fOutinterPnts, unsigned int fCounterMin)
//{
//
//	std::cout <<"the number of planes before processing: "<<fInLinkageMatrix.size()<<std::endl;
//
//	unsigned int n3DPoints = m_InputCloud->points.size();
//	unsigned int nPlanes = fInLinkageMatrix.size();
//	std::vector<pcl::PointXYZ> fPntsOfintersections;
//
//	for (unsigned int i = 0; i < nPlanes; i++)
//	{
//		//create a new intersection component
//		fIntersectorsPtr NewIntersectionEntries(new fIntersectors());
//
//		//get the first plane and validate all of its intersection points
//		fIntersectorsPtr tmpIntersect(new fIntersectors());
//		tmpIntersect = fInLinkageMatrix.at(i)->second;
//
//		unsigned int nIntersectionPoints = tmpIntersect->size();
//		for (unsigned int j = 0; j < nIntersectionPoints; j++)
//		{
//			//get the intersecting line 
//			Eigen::VectorXf &tmpintersectingline = tmpIntersect->at(j).get<0>();
//			unsigned int fIndx = tmpIntersect->at(j).get<1>();
//			pcl::PointXYZ &tmpPntinter2Vp = tmpIntersect->at(j).get<2>();
//			//count the number of supporting points within a defined radius
//
//			Eigen::Vector4f line_pt  (tmpintersectingline[0], tmpintersectingline[1], tmpintersectingline[2], 0); 
//			Eigen::Vector4f line_dir (tmpintersectingline[3], tmpintersectingline[4], tmpintersectingline[5], 0); //actually the directions are just the VP
//			line_dir.normalize ();
//
//			unsigned int fCountermax = 0;
//			//Iterate through the 3d points and calculate the distances from them to the line
//			for (size_t ii = 0; ii < n3DPoints; ++ii)
//			{
//				// Calculate the distance from the point to the line
//				// D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) / norm(p2-p1)
//				// Need to estimate sqrt here to keep MSAC and friends general
//				float fDistance = sqrt ((line_pt - m_InputCloud->points[ii].getVector4fMap ()).cross3 (line_dir).squaredNorm ());
//				if (fDistance < fToleranceRadius)
//				{
//					fCountermax++;
//				}
//			}
//
//			if (fCountermax > fCounterMin) //if the max counter is less than the minimum allowed, remove the entry from linkage matrix
//			{
//				NewIntersectionEntries->push_back(boost::make_tuple(boost::ref(tmpintersectingline), boost::ref(fIndx), boost::ref(tmpPntinter2Vp)));
//				fPntsOfintersections.push_back(tmpPntinter2Vp);
//			}
//		}
//
//		//Do we have any valid entry even ?
//		if (NewIntersectionEntries->size() == 0)
//		{
//			continue;
//		}
//
//		fWallLinkagePtr fnewLinkages(new fWallLinkage());
//		fnewLinkages->first = i;
//		fnewLinkages->second = NewIntersectionEntries;
//		fPreprocessedLinkageMatrix.push_back(fnewLinkages);
//	}
//
//	std::cout <<"the number of planes after processing: "<<fPreprocessedLinkageMatrix.size()<<std::endl;
//
//	PtCloudDataPtr fppoutintersections(new PtCloudData);
//	fppoutintersections->width = fPntsOfintersections.size();
//	fppoutintersections->height = 1;
//	fppoutintersections->resize(fppoutintersections->width * fppoutintersections->height);
//	
//	for (unsigned int i = 0; i < fPntsOfintersections.size(); i++)
//	{
//fppoutintersections->points[i] = fPntsOfintersections.at(i);
//	}
//
//	fOutinterPnts = fppoutintersections;
//
//}
//
//
//
//void HouseModelBuilder::convertIntersectionPointsToGraphEdges(const std::vector<fWallLinkagePtr> &fInLinkageMatrix)
//{
//	//first select all the intersections on the same edge line
//	unsigned  int nPlanes = fInLinkageMatrix.size();
//
//	for (unsigned int i = 0; i < nPlanes; i++)
//	{
//		//create a new intersection component
//		fIntersectorsPtr NewIntersectionEntries(new fIntersectors());
//
//		//get the first plane and validate all of its intersection points
//		fIntersectorsPtr tmpIntersect(new fIntersectors());
//		tmpIntersect = fInLinkageMatrix.at(i)->second;
//
//		unsigned int index_of_plane_a = fInLinkageMatrix.at(i)->first;
//
//		unsigned int nIntersectionPoints = tmpIntersect->size();
//	
//		if (nIntersectionPoints > 1)
//		{
//			PtCloudDataPtr on_line_intersection_point(new PtCloudData);
//			on_line_intersection_point->height = 1;
//			on_line_intersection_point->width = nIntersectionPoints;
//			on_line_intersection_point->resize(on_line_intersection_point->height * on_line_intersection_point->width);
//			
//			//porpulate the data points
//			for(unsigned int j = 0; j < nIntersectionPoints; j++)
//			{
//				//get the intersecting line 
//				Eigen::VectorXf &tmpintersectingline = tmpIntersect->at(j).get<0>();
//				unsigned int fIndx = tmpIntersect->at(j).get<1>();
//				pcl::PointXYZ &tmpPntinter2Vp = tmpIntersect->at(j).get<2>();
//				//count the number of supporting points within a defined radius
//				on_line_intersection_point->points[j] = tmpPntinter2Vp;
//			}
//
//			//build combinations of two and compute the distances between and score these with the number of inliers between any two selected points
//
//
//
//
//			//
//			//unsigned int fCountermax = 0;
//			//	//Iterate through the 3d points and calculate the distances from them to the line
//			//	for (size_t ii = j+1; ii < nIntersectionPoints; ii++)
//			//	{
//			//		// Calculate the distance from the point to the line
//			//		// D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) / norm(p2-p1)
//			//		// Need to estimate sqrt here to keep MSAC and friends general
//			//		float fDistance = sqrt ((line_pt - m_InputCloud->points[ii].getVector4fMap ()).cross3 (line_dir).squaredNorm ());
//			//		if (fDistance < fToleranceRadius)
//			//		{
//			//			fCountermax++;
//			//		}
//			//	}
//			}
//		}
//
//
//
//
//
//	//		if (fCountermax > fCounterMin) //if the max counter is less than the minimum allowed, remove the entry from linkage matrix
//	//		{
//	//			NewIntersectionEntries->push_back(boost::make_tuple(boost::ref(tmpintersectingline), boost::ref(fIndx), boost::ref(tmpPntinter2Vp)));
//	//			fPntsOfintersections.push_back(tmpPntinter2Vp);
//	//		}
//	//	}
//
//	//	//Do we have any valid entry even ?
//	//	if (NewIntersectionEntries->size() == 0)
//	//	{
//	//		continue;
//	//	}
//
//	//	fWallLinkagePtr fnewLinkages(new fWallLinkage());
//	//	fnewLinkages->first = i;
//	//	fnewLinkages->second = NewIntersectionEntries;
//	//	fPreprocessedLinkageMatrix.push_back(fnewLinkages);
//	//}
//}
//
//
//
//
//void HouseModelBuilder::_Print2DProjections(const std::vector<fWallLinkagePtr> &fInLinkageMatrix)
//{
//
//	// Compute the 3x3 covariance matrix
//	Eigen::Vector4f centroid;
//	pcl::compute3DCentroid (*m_InputCloud, centroid);
//
//	//Define the 2D x- and y-axis
//	fIntersectorsPtr firstIntersectionPnt(new fIntersectors());
//	firstIntersectionPnt = fInLinkageMatrix.at(2)->second;
//	//get the intersecting line 
//	//Eigen::VectorXf &firstIntersectingline = firstIntersectionPnt->at(0).get<0>();
//	//unsigned int fIndx = firstIntersectionPnt->at(0).get<1>(); //this index should be zero
//	pcl::PointXYZ &FirstPntinter2Vp = firstIntersectionPnt->at(0).get<2>();
//
//	//compute the direction vector from centroid to intersection point
//	Eigen::Vector4f Xaxis2D(FirstPntinter2Vp.x - centroid[0], FirstPntinter2Vp.y - centroid[1], FirstPntinter2Vp.z - centroid[2], 0);
//	Xaxis2D.normalize();
//	
//	Eigen::Vector4f Zaxis2D(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//	Eigen::Vector4f Yaxis2D = Zaxis2D.cross3(Xaxis2D);
//	Yaxis2D.normalize();
//	
//	Eigen::Vector4f line_pt_xaxis  (centroid[0], centroid[1], centroid[2], 0); 
//	Eigen::Vector4f line_dir_xaxis (Xaxis2D[0], Xaxis2D[1], Xaxis2D[2], 0); 
//
//	Eigen::Vector4f line_pt_yaxis  (centroid[0], centroid[1], centroid[2], 0); 
//	Eigen::Vector4f line_dir_yaxis (Yaxis2D[0], Yaxis2D[1], Yaxis2D[2], 0); 
//
//	unsigned int nIntersections = fInLinkageMatrix.size();
//	std::vector<Eigen::Vector2f> fCoordPoints;
//
//	for (unsigned int i = 0; i < nIntersections; i++)
//	{
//		//create a new intersection component
//		fIntersectorsPtr NewIntersectionEntries(new fIntersectors());
//
//		//get the first plane and validate all of its intersection points
//		fIntersectorsPtr tmpIntersect(new fIntersectors());
//		tmpIntersect = fInLinkageMatrix.at(i)->second;
//
//		unsigned int nIntersectionPoints = tmpIntersect->size();
//		for (unsigned int j = 0; j < nIntersectionPoints; j++)
//		{
//			//get the intersecting line
//			Eigen::VectorXf &tmpintersectingline = tmpIntersect->at(j).get<0>();
//			unsigned int fIndx = tmpIntersect->at(j).get<1>();
//			pcl::PointXYZ &tmpPntinter2Vp = tmpIntersect->at(j).get<2>();
//
//			//compute the direction vector from centroid to intersection point
//			Eigen::Vector4f tmpPnt2D(tmpPntinter2Vp.x - centroid[0], tmpPntinter2Vp.y - centroid[1], tmpPntinter2Vp.z - centroid[2], 0);
//			tmpPnt2D.normalize();
//
//			Vec3d xa = {line_dir_xaxis[0], line_dir_xaxis[1], line_dir_xaxis[2]};
//			Vec3d ya = {line_dir_yaxis[0], line_dir_yaxis[1], line_dir_yaxis[2]};
//			Vec3d cb = {tmpPnt2D[0], tmpPnt2D[1], tmpPnt2D[2]};
//
//			const float sAnglesInRadianstmp_xa = Math3d::Angle(xa, cb);
//			const float sAnglesInRadianstmp_xb = Math3d::Angle(ya, cb);
//			const float sAnglesInDegreestmp_xa = sAnglesInRadianstmp_xa * 180.0f / CV_PI;
//			const float sAnglesInDegreestmp_xb = sAnglesInRadianstmp_xb * 180.0f / CV_PI;
//
//
//			//Determine the quadrant of the vector
//			double fAngleToXaxisRadiens = pcl::getAngle3D(line_dir_xaxis, tmpPnt2D);
//			double fAngleToXaxisDegrees = pcl::rad2deg(fAngleToXaxisRadiens);
//			double fAngleToYaxisRadiens = pcl::getAngle3D(line_dir_yaxis, tmpPnt2D);
//			double fAngleToYaxisDegrees = pcl::rad2deg(fAngleToYaxisRadiens);
//
//			//count the number of supporting points within a defined radius
//
//			// Calculate the distance from the point to the line
//			// D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p2-p0)) / norm(p2-p1)
//			// Need to estimate sqrt here to keep MSAC and friends general
//			float fDistanceX = sqrt ((line_pt_xaxis - tmpPntinter2Vp.getVector4fMap ()).cross3(line_dir_xaxis).squaredNorm ());
//			float fDistanceY = sqrt ((line_pt_yaxis - tmpPntinter2Vp.getVector4fMap ()).cross3(line_dir_yaxis).squaredNorm ());
//
//			//ToDo: 
//			//- check the sign using angles according to the quadrant where the point resides
//			//- count the number of inliers
//			//- compute the spreading factor
//			//...
//			double eps_angles = 1.0;
//
//			//first quadrant
//			/*if ((abs(fAngleToXaxisDegrees - 90 ) < eps_angles) && (abs(fAngleToYaxisDegrees - 90 ) < eps_angles))*/
//			if ((fAngleToXaxisDegrees < 90 ) && (fAngleToYaxisDegrees < 90 ))
//			{
//				fDistanceX = fDistanceX;
//				fDistanceY = fDistanceY;
//			}
//
//			//second quadrant
//			/*if((abs(fAngleToXaxisDegrees - 90 ) < eps_angles) && (abs(fAngleToYaxisDegrees - 90 ) > eps_angles))*/
//			if((fAngleToXaxisDegrees > 90 ) && (fAngleToYaxisDegrees < 90))
//			{
//				fDistanceX = -fDistanceX;
//				fDistanceY = fDistanceY;
//			}
//
//			//third quadrant
//			if((fAngleToXaxisDegrees > 90 ) && (fAngleToYaxisDegrees > 90 ))
//			{
//				fDistanceX = -fDistanceX;
//				fDistanceY = -fDistanceY;
//			}
//			
//			//fourth quadrant
//			if((fAngleToXaxisDegrees < 90 ) && (fAngleToYaxisDegrees > 90 ))
//			{
//				fDistanceX = fDistanceX;
//				fDistanceY = -fDistanceY;
//			}
//
//			Eigen::Vector2f tmpCoordinates(fDistanceX, fDistanceY);
//			fCoordPoints.push_back(tmpCoordinates);
//
//			//if (fCountermax > fCounterMin) //if the max counter is less than the minimum allowed, remove the entry from linkage matrix
//			//{
//			//	NewIntersectionEntries->push_back(boost::make_tuple(boost::ref(tmpintersectingline), boost::ref(fIndx), boost::ref(tmpPntinter2Vp)));
//			//	fPntsOfintersections.push_back(tmpPntinter2Vp);
//			//}
//		}
//
//		////Do we have any valid entry even ?
//		//if (NewIntersectionEntries->size() == 0)
//		//{
//		//	continue;
//		//}
//
//		//fWallLinkagePtr fnewLinkages(new fWallLinkage());
//		//fnewLinkages->first = i;
//		//fnewLinkages->second = NewIntersectionEntries;
//		//fPreprocessedLinkageMatrix.push_back(fnewLinkages);
//	}
//
//	//PtCloudDataPtr fppoutintersections(new PtCloudData);
//	//fppoutintersections->width = fPntsOfintersections.size();
//	//fppoutintersections->height = 1;
//	//fppoutintersections->resize(fppoutintersections->width * fppoutintersections->height);
//
//	//for (unsigned int i = 0; i < fPntsOfintersections.size(); i++)
//	//{
//	//	fppoutintersections->points[i] = fPntsOfintersections.at(i);
//	//}
//
//	//fOutinterPnts = fppoutintersections;
//
//	//export to matlab
//	int numPoints = fCoordPoints.size();
//	ofstream myfile;
//	myfile.open ("example_intersections.txt");
//	
//	for ( unsigned int i = 0; i < numPoints; i++)
//	{
//		myfile << fCoordPoints.at(i)[0];
//		myfile << " ";
//		myfile << fCoordPoints.at(i)[1]<<"\n";
//
//	}
//	myfile.close();
//
//
//
//	// Set up a 2D scene, add an XY chart to it
//	vtkSmartPointer<vtkContextView> view =
//		vtkSmartPointer<vtkContextView>::New();
//	view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
//	view->GetRenderWindow()->SetSize(400, 300);
//
//	vtkSmartPointer<vtkChartXY> chart =
//		vtkSmartPointer<vtkChartXY>::New();
//	view->GetScene()->AddItem(chart);
//	chart->SetShowLegend(true);
//
//	// Create a table with some points in it...
//	vtkSmartPointer<vtkTable> table =
//		vtkSmartPointer<vtkTable>::New();
//
//	vtkSmartPointer<vtkFloatArray> arrX =
//		vtkSmartPointer<vtkFloatArray>::New();
//	arrX->SetName("X Axis");
//	table->AddColumn(arrX);
//
//	vtkSmartPointer<vtkFloatArray> arrC =
//		vtkSmartPointer<vtkFloatArray>::New();
//	arrC->SetName("Intersection Point");
//	table->AddColumn(arrC);
//
//	//vtkSmartPointer<vtkFloatArray> arrS =
//	//	vtkSmartPointer<vtkFloatArray>::New();
//	//arrS->SetName("Sine");
//	//table->AddColumn(arrS);
//
//	//vtkSmartPointer<vtkFloatArray> arrT =
//	//	vtkSmartPointer<vtkFloatArray>::New();
//	//arrT->SetName("Sine - Cosine");
//	//table->AddColumn(arrT);
//
//	// Test charting with a few more points...
//
//	//float inc = 7.5 / (numPoints-1);
//	table->SetNumberOfRows(numPoints);
//	for (int i = 0; i < numPoints; i++)
//	{
//		table->SetValue(i, 1, fCoordPoints.at(i)[0]);
//		table->SetValue(i, 0, fCoordPoints.at(i)[1]);
//		//table->SetValue(i, 0, i * inc);
//		//table->SetValue(i, 1, cos(i * inc) + 0.0);
//		//table->SetValue(i, 2, sin(i * inc) + 0.0);
//		//table->SetValue(i, 3, sin(i * inc) - cos(i * inc));
//
//	}
//
//	// Add multiple scatter plots, setting the colors etc
//	vtkPlot *points = chart->AddPlot(vtkChart::POINTS);
//#if VTK_MAJOR_VERSION <= 5
//	points->SetInput(table, 0, 1);
//#else
//	points->SetInputData(table, 0, 1);
//#endif
//	points->SetColor(0, 0, 0, 255);
//	points->SetWidth(2.0);
//	vtkPlotPoints::SafeDownCast(points)->SetMarkerStyle(vtkPlotPoints::CROSS);
////
////	points = chart->AddPlot(vtkChart::POINTS);
////#if VTK_MAJOR_VERSION <= 5
////	points->SetInput(table, 0, 2);
////#else
////	points->SetInputData(table, 0, 2);
////#endif
////	points->SetColor(0, 0, 0, 255);
////	points->SetWidth(1.0);
////	vtkPlotPoints::SafeDownCast(points)->SetMarkerStyle(vtkPlotPoints::PLUS);
////
////	points = chart->AddPlot(vtkChart::POINTS);
////#if VTK_MAJOR_VERSION <= 5
////	points->SetInput(table, 0, 3);
////#else
////	points->SetInputData(table, 0, 3);
////#endif
////	points->SetColor(0, 0, 255, 255);
////	points->SetWidth(1.0);
////	vtkPlotPoints::SafeDownCast(points)->SetMarkerStyle(vtkPlotPoints::CIRCLE);
//
//	//Finally render the scene
//	view->GetRenderWindow()->SetMultiSamples(0);
//	view->GetInteractor()->Initialize();
//	view->GetInteractor()->Start();
//
//	
//}
//
////generate all connecting edges and their edge likelyhoods
//void HouseModelBuilder::_FilterConnectingSegments(const std::vector<fWallLinkagePtr> &fInLinkageMatrix, std::vector<EdgeConnectorPtr> &fConnectingEdges)
//{
//	boost::timer fStartTime; // the destructor will display the time when fStartTime is out of scope anyway!
//	//progress_timer t;  // start timing
//				unsigned int nErrors = 0;
//
//	unsigned int nIntersections = fInLinkageMatrix.size();
//	double fThresh = m_ModelParams->fSegModelParams->fRansacThresh;
//
//	std::cout<<"RANSAC THRESHOLD: "<< fThresh<<std::endl;
//
//	std::vector<Eigen::VectorXf> fIntersectionLines;
//	unsigned int fMinNumPntsOnSegment = 980;
//	//Get all intersection lines. ToDo. these code is just a copy and paste and is already being computed above and can be made used of
//	for (unsigned int i = 0; i < nIntersections; i++)
//	{
//		//create a new intersection component
//		fIntersectorsPtr NewIntersectionEntries(new fIntersectors());
//
//		//get the first plane and validate all of its intersection points
//		fIntersectorsPtr tmpIntersect(new fIntersectors());
//		tmpIntersect = fInLinkageMatrix.at(i)->second;
//
//		unsigned int nIntersectionPoints = tmpIntersect->size();
//		
//		for (unsigned int j = 0; j < nIntersectionPoints; j++)
//		{
//			//get the intersecting line
//			const Eigen::VectorXf &tmpintersectingline = tmpIntersect->at(j).get<0>();
//			//unsigned int fIndx = tmpIntersect->at(j).get<1>();
//			//pcl::PointXYZ &tmpPntinter2Vp = tmpIntersect->at(j).get<2>();
//			fIntersectionLines.push_back(tmpintersectingline);
//		}//_i
//	}//_j
//
//	unsigned int nInterstingLines = fIntersectionLines.size();
//	std::vector<PtCloudDataPtr> fSegmentA, fSegmentsFiltered;
//	Eigen::VectorXf fLineA, fLineB;
//	unsigned int fcounters = 0;
//
//	for (unsigned int i = 0; i < nInterstingLines; i++)
//	{
//		fLineA = fIntersectionLines[i];
//		Eigen::Vector4f fLineAmodel(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 1);
//		pcl::PointXYZ tmpPA;
//		tmpPA.x = fLineA[0];
//		tmpPA.y = fLineA[1];
//		tmpPA.z = fLineA[2];
//
//		for (unsigned int j = i + 1; j < nInterstingLines; j++)
//		{
//			fLineB = fIntersectionLines[j];
//			pcl::PointXYZ tmpPB;
//			tmpPB.x = fLineB[0];
//			tmpPB.y = fLineB[1];
//			tmpPB.z = fLineB[2];
//			Eigen::Vector4f fLineBmodel(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 1);
//			//fLineBmodel[3] = -1 * fLineBmodel.dot (tmpPB.getVector4fMap());
//			//fLineAmodel[3] = -1 * fLineAmodel.dot (tmpPA.getVector4fMap());
//			fLineBmodel[3] = -1 * ((tmpPB.x * fLineBmodel[0]) + (tmpPB.y * fLineBmodel[1]) + (tmpPB.z * fLineBmodel[2]));
//			fLineAmodel[3] = -1 * ((tmpPA.x * fLineAmodel[0]) + (tmpPA.y * fLineAmodel[1]) + (tmpPA.z * fLineAmodel[2]));
//
//			//build the plane spanned by fLineA and fLineB
//			pcl::PointXYZ fLineAPoint, fLineBPoint, fLineAPointProjected, fLineAPointProjectedpcl, fLineAPointProjectedpcl2;
//			fLineAPoint.x = fLineA[0];
//			fLineAPoint.y = fLineA[1];
//			fLineAPoint.z = fLineA[2];
//			fLineBPoint.x = fLineB[0];
//			fLineBPoint.y = fLineB[1];
//			fLineBPoint.z = fLineB[2];
//			
//			_ProjectPointOnLine(tmpPB, fLineA, fLineAPointProjected);
//			//pcl::projectPoint(tmpPB, fLineBmodel, fLineAPointProjectedpcl);
//
//			//std::cout<<fLineAPointProjectedpcl.x<<","<<fLineAPointProjectedpcl.y<<","<<fLineAPointProjectedpcl.z<<std::endl;
//			//std::cout<<fLineAPointProjected.x<<","<<fLineAPointProjected.y<<","<<fLineAPointProjected.z<<std::endl;
//			//boost::this_thread::sleep (boost::posix_time::seconds(100));
//
//			//if the lines are just too close together do not consider
//			if (pcl::euclideanDistance(fLineAPointProjected, tmpPB) < 10*fThresh )
//			{
//				continue;
//			}
//
//			Eigen::Vector4f line_ptB(fLineB[0], fLineB[1], fLineB[2], 0);
//			Eigen::Vector4f line_ptA(fLineA[0], fLineA[1], fLineA[2], 0);
//			Eigen::Vector4f line_dir(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//
//			float fDistA2B = sqrt ((line_ptA -tmpPB.getVector4fMap ()).cross3 (line_dir).squaredNorm ());
//
//			if (fDistA2B < 3)
//			{
//				continue;
//			}
//
//
//			Eigen::Vector4f fVPDir(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//			//Eigen::Vector4f fDirB(tmpPB.x - fLineAPointProjected.x, tmpPB.y - fLineAPointProjected.y, tmpPB.z - fLineAPointProjected.z, 0);
//			Eigen::Vector4f fDirB(tmpPB.getVector4fMap() - fLineAPointProjected.getVector4fMap());
//			Eigen::Vector4f fPlaneCoefficients = fVPDir.cross3(fDirB);
//			fPlaneCoefficients.normalize();
//			Eigen::Vector4f fPlaneCoefficientsFinal(fPlaneCoefficients[0], fPlaneCoefficients[1], fPlaneCoefficients[2], 1);
//
//			// Hessian form (D = nc . p_plane (centroid here) + p)
//			fPlaneCoefficientsFinal[3] = -1 * ((fLineAPoint.x * fPlaneCoefficientsFinal[0]) + (fLineAPoint.y * fPlaneCoefficientsFinal[1]) + (fLineAPoint.z * fPlaneCoefficientsFinal[2]));//.dot (fLineAPoint.getVector4fMap());
//
//			//temporal storage of the inliers
//			CVec3dArrayPtr tmpSegmentPoints(new CVec3dArray());
//			CVec3dArrayPtr tmpSegmentPointsProjected(new CVec3dArray());
//			CVec3dArrayPtr tmpSegmentPointsFiltered(new CVec3dArray());
//			CVec3dArrayPtr tmpSegmentPointsFilteredProjected(new CVec3dArray());
//
//			unsigned int tmpcounter = 0;
//
//			//get all inliers:: this step is also irrelevant recomputation could be done better
//			// Iterate through the 3d points and calculate the distances from them to the plane
//
//			for (size_t h = 0; h < m_InputCloud->points.size (); ++h)
//			{
//				// Calculate the distance from the point to the plane normal as the dot product
//				// D = (P-A).N/|N|
//				/*distances[i] = fabs (model_coefficients[0] * input_->points[(*indices_)[i]].x +
//				model_coefficients[1] * input_->points[(*indices_)[i]].y +
//				model_coefficients[2] * input_->points[(*indices_)[i]].z +
//				model_coefficients[3]);*/
//				//Eigen::Vector4f pt (m_InputCloud->points[h].x,
//				//	m_InputCloud->points[h].y,
//				//	m_InputCloud->points[h].z,
//				//	1);
//				pcl::PointXYZ pt = m_InputCloud->points[h];
//				double fDistance = fabs ( (fPlaneCoefficientsFinal[0] * pt.x) + (fPlaneCoefficientsFinal[1] * pt.y) + (fPlaneCoefficientsFinal[2] * pt.z) + fPlaneCoefficientsFinal[3]);//fPlaneCoefficientsFinal.dot (pt));
//				double fmDist =  pcl::pointToPlaneDistance(pt, fPlaneCoefficientsFinal);
//
//				//std::cout<<"DistancesA: "<< fDistance<<std::endl;
//				//std::cout<<"DistancesB: "<< fmDist<<std::endl;
//
//				//if (fmDist > 70)
//				//{ 
//				//	nErrors++;
//				//	//std::cout<<"Errors!!"<<std::endl;
//				//	
//				//}
//
//
//
//				if (fmDist <= fThresh)
//				{
//					//tmpcounter++;
//					Vec3d tmpInlier, tmpInlierProjected ;
//					tmpInlier.x = m_InputCloud->points[h].x;
//					tmpInlier.y = m_InputCloud->points[h].y;
//					tmpInlier.z = m_InputCloud->points[h].z;
//					tmpSegmentPoints->AddElement(tmpInlier);
//
//					//project point on plane
//					pcl::PointXYZ pOrigPnt, pPnt;
//					pOrigPnt.x = tmpInlier.x;
//					pOrigPnt.y = tmpInlier.y;
//					pOrigPnt.z = tmpInlier.z;
//					pcl::projectPoint(pOrigPnt, fPlaneCoefficientsFinal, pPnt);
//					tmpInlierProjected.x = pPnt.x;
//					tmpInlierProjected.y = pPnt.y;
//					tmpInlierProjected.z = pPnt.z;
//					tmpSegmentPointsProjected->AddElement(tmpInlierProjected);
//
//				}
//
//			}//_h
//
//			//if (tmpcounter > fcounters)
//			//	fcounters=tmpcounter;
//			
//		/*	
//
//			Vec3d PlaneNormal, p1;
//			PlaneNormal.x = fPlaneCoefficientsFinal[0];
//			PlaneNormal.y = fPlaneCoefficientsFinal[1];
//			PlaneNormal.z = fPlaneCoefficientsFinal[2];
//			p1.x = fLineBPoint.x;
//			p1.y = fLineBPoint.y;
//			p1.z = fLineBPoint.z;
//
//			const float c = Math3d::ScalarProduct(PlaneNormal, p1);
//			for (size_t h = 0; h < m_InputCloud->points.size (); ++h)
//			{
//				Vec3d tmpInlier;
//				tmpInlier.x = m_InputCloud->points[h].x;
//				tmpInlier.y = m_InputCloud->points[h].y;
//				tmpInlier.z = m_InputCloud->points[h].z;
//
//				if (fabsf(Math3d::ScalarProduct(PlaneNormal, tmpInlier) - c) <= fThresh)
//				{
//					tmpSegmentPoints->AddElement(tmpInlier);
//				}
//
//			}*/
//			//validate the model to have a minimum number of points
//			if (tmpSegmentPoints->GetSize() < fMinNumPntsOnSegment)
//			{
//				continue;
//			}
//	//	std::cout<<" Segment Size: "<<tmpSegmentPoints->GetSize()<<std::endl;
//		//pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//		//viewer0.setBackgroundColor (0.3, 0.3, 0.3);
//		//float fLineScale = 5.0f;
//		//Eigen::VectorXf tmpLineParams = fLineA;
//		//pcl::PointXYZ fLineStart;
//		//fLineStart.x = tmpLineParams[0] - fLineScale* tmpLineParams[3];
//		//fLineStart.y = tmpLineParams[1] - fLineScale* tmpLineParams[4];
//		//fLineStart.z = tmpLineParams[2] - fLineScale* tmpLineParams[5];
//		//pcl::PointXYZ fLineEndPoint;
//
//		//fLineEndPoint.x = tmpLineParams[0] + fLineScale* tmpLineParams[3];
//		//fLineEndPoint.y = tmpLineParams[1] + fLineScale* tmpLineParams[4];
//		//fLineEndPoint.z = tmpLineParams[2] + fLineScale* tmpLineParams[5];
//
//		//Eigen::VectorXf tmpLineParamsB = fLineB;
//		//pcl::PointXYZ fLineStartB;
//		//fLineStartB.x = tmpLineParamsB[0] - fLineScale* tmpLineParamsB[3];
//		//fLineStartB.y = tmpLineParamsB[1] - fLineScale* tmpLineParamsB[4];
//		//fLineStartB.z = tmpLineParamsB[2] - fLineScale* tmpLineParamsB[5];
//		//pcl::PointXYZ fLineEndPointB;
//
//		//fLineEndPointB.x = tmpLineParamsB[0] + fLineScale* tmpLineParamsB[3];
//		//fLineEndPointB.y = tmpLineParamsB[1] + fLineScale* tmpLineParamsB[4];
//		//fLineEndPointB.z = tmpLineParamsB[2] + fLineScale* tmpLineParamsB[5];
//
//		//std::string fCloudIdA = "Line CloudA";// + boost::lexical_cast<std::string>(ii);
//		//std::string fCloudIdB = "Line CloudB";// + boost::lexical_cast<std::string>(ii);
//		//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fCloudIdA, 0); 
//		//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStartB, fLineEndPointB, 255.0, 255.0, 0.0, fCloudIdB,0);
//
//		//PtCloudDataPtr tmpsegmentcloudPr0(new PtCloudData());
//		//VFRMath::CVecArrayToPCLDataPtr(tmpSegmentPointsProjected, tmpsegmentcloudPr0);
//
//
//		//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aHandler (tmpsegmentcloudPr0, 0.0, 255.0, 0);
//		//	viewer0.addPointCloud (tmpsegmentcloudPr0, aHandler, "clouda");
//
//
//
//			//while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//			//{
//			//	//	range_image_widget.spinOnce ();
//			//	viewer0.spinOnce (10);
//			//	boost::this_thread::sleep (boost::posix_time::microseconds(100));
//			//}
//			////_End_Display
//
//
//		//	std::cout<<"Segment Size (Unfiltered): "<<tmpSegmentPoints->GetSize()<<std::endl; 
//		//	std::cout<<"Projected Segment Size (Unfiltered): "<<tmpSegmentPointsProjected->GetSize()<<std::endl; 
//
//			Eigen::Vector4f fLineACoefs(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 1);
//			Eigen::Vector4f fLineBCoefs(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 1);
//
//			// Hessian form (D = nc . p_line (centroid here) + p)
//			fLineACoefs[3] = -1 *((_m_computedVP[3] * fLineA[0] ) + (_m_computedVP[4] * fLineA[1]) + (_m_computedVP[5] * fLineA[2]));// fLineACoefs.dot (fLineAPoint.getVector4fMap());
//			fLineBCoefs[3] = -1 *((_m_computedVP[3] * fLineB[0] ) + (_m_computedVP[4] * fLineB[1]) + (_m_computedVP[5] * fLineB[2]));
//
//
//			//get the distance between the two lines
//			//Eigen::Vector4f line_ptB(fLineB[0], fLineB[1], fLineB[2], 0);
//			//Eigen::Vector4f line_ptA(fLineA[0], fLineA[1], fLineA[2], 0);
//			//Eigen::Vector4f line_dir(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//
//			//float fDistA2B = sqrt ((line_ptA -tmpPB.getVector4fMap ()).cross3 (line_dir).squaredNorm ());
//			
//
//			//extract all points between the two lines using change of signs
//			for (unsigned int h2 = 0; h2 < tmpSegmentPointsProjected->GetSize(); h2++)
//			{
//
//
//				Vec3d tmpP = (*tmpSegmentPointsProjected)[h2];
//				pcl::PointXYZ fChoosenPoint2;
//				fChoosenPoint2.x = tmpP.x;
//				fChoosenPoint2.y = tmpP.y;
//				fChoosenPoint2.z = tmpP.z;
//
//				float fDist2A = sqrt ((line_ptA -fChoosenPoint2.getVector4fMap()).cross3 (line_dir).squaredNorm ());
//				float fDist2B = sqrt ((line_ptB -fChoosenPoint2.getVector4fMap()).cross3 (line_dir).squaredNorm ());
//
//				//float fDistanceA = fLineACoefs.dot (fChoosenPoint2);
//				//float fDistanceB = fLineBCoefs.dot (fChoosenPoint2);
//
//				if ((fDist2A <= fDistA2B) && (fDist2B <=fDistA2B)/*fDistanceA*fDistanceB <= 0*/)//equal zero will account for points on the lines
//				{
//					tmpSegmentPointsFilteredProjected->AddElement(tmpP);
//				}
//
//			}//_h2
//
//			////std::cout<<"Projected Segment Size (Filtered): "<<tmpSegmentPointsFilteredProjected->GetSize()<<std::endl; 
//
//			//validate the filtered model to have a minimum number of points
//			if (tmpSegmentPointsFilteredProjected->GetSize() < fMinNumPntsOnSegment)
//			{
//				continue;
//			}
//
//			PtCloudDataPtr tmpsegmentcloud(new PtCloudData());
//			PtCloudDataPtr tmpsegmentcloudProjectedFiltered(new PtCloudData());
//			VFRMath::CVecArrayToPCLDataPtr(tmpSegmentPoints, tmpsegmentcloud);
//			VFRMath::CVecArrayToPCLDataPtr(tmpSegmentPointsFilteredProjected, tmpsegmentcloudProjectedFiltered);
//
//			////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> bHandler (tmpsegmentcloudProjected, 255.0, 0.0, 0.0);
//			////viewer0.addPointCloud (tmpsegmentcloudProjected, bHandler, "cloudb");
//			////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloudb");
//			////viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "clouda");
//			////viewer0.addCoordinateSystem (1.0f);
//
//			////while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//			////{
//			////	//	range_image_widget.spinOnce ();
//			////	viewer0.spinOnce (10);
//			////	boost::this_thread::sleep (boost::posix_time::microseconds(100));
//			////}
//			//////_End_Display
//			
//			Vec3d tPCoes;
//			tPCoes.x = fPlaneCoefficientsFinal[0];
//			tPCoes.y = fPlaneCoefficientsFinal[1];
//			tPCoes.z = fPlaneCoefficientsFinal[2];
//
//			EdgeConnectorPtr tmpEdgeCandidate(new EdgeConnector());
//			tmpEdgeCandidate->fLineA = fLineA;
//			tmpEdgeCandidate->fLineB = fLineB;
//			tmpEdgeCandidate->fPlaneCoeficients = fPlaneCoefficientsFinal;
//			tmpEdgeCandidate->PlaneNormal = tPCoes;
//			tmpEdgeCandidate->fVPLineIndxA = i;
//			tmpEdgeCandidate->fVPLineIndxB = j;
//			tmpEdgeCandidate->EdgeSegmentInliers = tmpsegmentcloud;
//			tmpEdgeCandidate->ProjEdgeSegmentInliers = tmpsegmentcloudProjectedFiltered;
//			tmpEdgeCandidate->fWeight = float(tmpsegmentcloudProjectedFiltered->points.size());
//			tmpEdgeCandidate->SegWidth = fDistA2B;
//			fConnectingEdges.push_back(tmpEdgeCandidate);
//		}//_j
//	}//_i
////std::cout<<"MaxCounts: "<<fcounters<<std::endl;
////std::cout<<"MaxErros: "<<nErrors<<std::endl;
//}
//
//
//
//
//void HouseModelBuilder::_InterpreteHull(const std::vector<fWallLinkagePtr> &fInLinkageMatrix, PtCloudDataPtr &foutHullPnts)
//{
//
//	unsigned int nPlanes = fInLinkageMatrix.size();
//
//	for (unsigned int i = 0; i < nPlanes; i++)
//	{
//		fIntersectorsPtr tmpIntersect(new fIntersectors());
//		tmpIntersect = fInLinkageMatrix.at(i)->second;
//
//		if (tmpIntersect->size() == 0)
//		{
//			continue;
//		}
//	}
//
//
//	//Do search for circles in the graph
//}
//
//
//
//void HouseModelBuilder::_ValidateLines(const std::vector<VerticalEdgeLinePtr> &fLinesOptimized, const std::vector<fWallLinkagePtr> &fInLinkageMatrix, std::vector<fWallLinkagePtr> &fOutLinkageMatrix, double fConnectionTolerrance)
//{
//	unsigned int nPlanes = fInLinkageMatrix.size();
//	unsigned int nCurvatureLines = fLinesOptimized.size();
//
//	//over all curvature lines
//	for (unsigned int i = 0; i < nCurvatureLines; i++)
//	{
//		Eigen::VectorXf &fSelectedCurvatureLine = fLinesOptimized.at(i)->fOptimLinesCoefs;
//
//		//over all intersection lines
//		fIntersectorsPtr tmpIntersectors(new fIntersectors());
//		for (unsigned int j = 0; j < nPlanes; j++)
//		{
//			fIntersectorsPtr outNewIntersectors(new fIntersectors());
//
//			tmpIntersectors = fInLinkageMatrix.at(j)->second; // all the intersection points on plane[j]
//			for (unsigned int ki = 0; ki < tmpIntersectors->size(); ki++)
//			{
//				Eigen::VectorXf &fTmpLine = tmpIntersectors->at(ki).get<0>();
//				unsigned int fIndexOfPlane = tmpIntersectors->at(ki).get<1>();
//				pcl::PointXYZ &fProjPnt = tmpIntersectors->at(ki).get<2>();
//
//				//compute the perpendicular distance between the two lines
//				Eigen::Vector4f line_pt0 (fTmpLine[0], fTmpLine[1], fTmpLine[2], 0);
//				Eigen::Vector4f line_dir0 (fTmpLine[3], fTmpLine[4], fTmpLine[5], 0);
//				Eigen::Vector4f fCurvLinePoint(fSelectedCurvatureLine[0], fSelectedCurvatureLine[1], fSelectedCurvatureLine[2], 0 );
//
//				//ToDo: Check thier directions to be sure that both lines are parallel
//
//				// Calculate the distance from the point to the line
//				double fShortestDistance = sqrt ((line_pt0 - fCurvLinePoint).cross3 (line_dir0).squaredNorm ());
//
//				if ( fShortestDistance < fConnectionTolerrance)
//				{
//					outNewIntersectors->push_back(boost::make_tuple(boost::ref(fTmpLine), boost::ref(fIndexOfPlane), boost::ref(fProjPnt)));
//				}
//			}
//			fWallLinkagePtr fnewLinkages(new fWallLinkage());
//			fnewLinkages->first = j;
//			fnewLinkages->second = outNewIntersectors;
//			fOutLinkageMatrix.push_back(fnewLinkages);
//		}
//
//	}
//
//	
//	//fIntersectorsPtr tmpIntersectors(new fIntersectors());
//	//for(unsigned int i = 0; i < nVertices; i++)
//	//{
//	//	tmpIntersectors = fInLinkageMatrix.at(i)->second;
//	//
//	//	for (unsigned int j = 0; j < nCurvatureLines)
//	//	{
//	//	}
//
//}
//
//
//void HouseModelBuilder::_ValidateLines(const std::vector<VerticalEdgeLinePtr> &fLinesOptimized, std::vector<VerticalEdgeLinePtr> &ffLinesOptimizedValidated, float fStdMult)
//{
//	unsigned int fNumLines = fLinesOptimized.size();
//	unsigned int fIndxSmallest = 0;
//	unsigned int fNumPntsCons = 20;
//	std::vector<VerticalEdgeLinePtr> tmpLines;
//	//select the smallest 
//	for(unsigned int i = 0 ; i < fNumLines; i++)
//	{
//		unsigned int tmpIndex = fLinesOptimized.at(i)->fProjectedToModel->points.size();
//		if (tmpIndex < fNumPntsCons)
//			continue;
//		if (tmpIndex > fIndxSmallest )
//		{
//			fIndxSmallest = fIndxSmallest;
//			tmpLines.push_back(fLinesOptimized.at(i));
//		}
//	}
//
//	//compute the mean
//	std::vector<float> fDistances;
//	float fMeanDistance = 0.0f;
//	for (unsigned int i = 0; i < tmpLines.size(); i ++)
//	{
//		float tmpDist = _meanInterPntDistance(tmpLines.at(i)->fProjectedToModel, fNumPntsCons);
//		fDistances.push_back(tmpDist);
//		fMeanDistance +=tmpDist;
//	}
//
//	fMeanDistance = fMeanDistance/tmpLines.size();
//
//	//compute the std
//	float fSTD = 0.0f;
//	for (unsigned int i = 0; i < tmpLines.size(); i ++)
//	{
//		fSTD += ((fDistances.at(i) - fMeanDistance)*(fDistances.at(i) - fMeanDistance));
//	}
//
//	//get sampled std
//	fSTD = fSTD / (tmpLines.size() - 1);
//	fSTD = sqrtf(fSTD);
//
//	//accept only those which are a number of multiples of std
//	for (unsigned int i = 0; i < tmpLines.size(); i++)
//	{
//
//		float tMDistDiff = abs(fDistances.at(i) - (fStdMult * fMeanDistance));
//
//		if (tMDistDiff < fMeanDistance)
//		{
//			ffLinesOptimizedValidated.push_back(tmpLines.at(i));
//		}
//	}
//}
//
//
//void HouseModelBuilder::_DetermineQuadPoints(const std::vector<VerticalEdgeLinePtr> &fVPBestLines, const Vec3dPtr &fVPOptimized, PtCloudDataPtr &fCloudTopPoints, PtCloudDataPtr &fCloudBottomPoints)
//{
//	std::vector<VerticalEdgeLinePtr> fVPBestNLines;
//	//unsigned int fTopNPercent = 10; //Haus23_5=10
//	//unsigned int fTopNPercent = 95; //haus51
//	unsigned int fTopNPercent = 75; //fachwerkshaus
//	_GetNBestVerticalLines(fVPBestLines, fVPBestNLines, fTopNPercent);
//	const float fLength = 0.5 * fVPBestNLines.at(0)->fLineLength;
//
//	PtCloudDataPtr fIntersectionCloudIn(new PtCloudData);
//	PtCloudDataPtr fIntersectionCloudOut(new PtCloudData);
//	_DeterminCentroidPoints( fVPOptimized, fVPBestNLines, fIntersectionCloudIn);
//
//	_FCloudHull(fIntersectionCloudIn, fIntersectionCloudOut);
//
//	unsigned int nHullPnts = fIntersectionCloudOut->points.size();
//
//	PtCloudDataPtr fIntersectionPntTop(new PtCloudData);
//	fIntersectionPntTop->width = nHullPnts;
//	fIntersectionPntTop->height = 1;
//	fIntersectionPntTop->resize(fIntersectionPntTop->height * fIntersectionPntTop->width);
//
//	PtCloudDataPtr fIntersectionPntBottom(new PtCloudData);
//	fIntersectionPntBottom->width = nHullPnts;
//	fIntersectionPntBottom->height = 1;
//	fIntersectionPntBottom->resize(fIntersectionPntBottom->height * fIntersectionPntBottom->width);
//
//
//	//Top
//	for (unsigned int i = 0; i < nHullPnts; i++)
//	{
//		fIntersectionPntTop->points[i].x = fIntersectionCloudOut->points[i].x + fLength*fVPOptimized->x;
//		fIntersectionPntTop->points[i].y = fIntersectionCloudOut->points[i].y + fLength*fVPOptimized->y;
//		fIntersectionPntTop->points[i].z = fIntersectionCloudOut->points[i].z + fLength*fVPOptimized->z;
//	}
//	fCloudTopPoints = fIntersectionPntTop;
//
//	//Bottom
//	for (unsigned int i = 0; i < nHullPnts; i++)
//	{
//		fIntersectionPntBottom->points[i].x = fIntersectionCloudOut->points[i].x - fLength*fVPOptimized->x;
//		fIntersectionPntBottom->points[i].y = fIntersectionCloudOut->points[i].y - fLength*fVPOptimized->y;
//		fIntersectionPntBottom->points[i].z = fIntersectionCloudOut->points[i].z - fLength*fVPOptimized->z;
//	}
//	fCloudBottomPoints = fIntersectionPntBottom;
//}
//
//
//
//
//void HouseModelBuilder::_GetNBestVerticalLines(const std::vector<VerticalEdgeLinePtr> &fVPLineOptimizedInliers, std::vector<VerticalEdgeLinePtr> &fVPBestNLines, unsigned int fTopNPercent)
//{
//	unsigned int nLineCandidates = fVPLineOptimizedInliers.size();
//	std::vector<float> fLengths(nLineCandidates);
//
//	//Determine the max length
//	float fMaxLength = 0;
//	for (unsigned int i = 0; i < nLineCandidates; i++)
//	{
//		const float fLengthTmp = fVPLineOptimizedInliers.at(i)->fLineLength;
//		fLengths.at(i) = fLengthTmp;
//		if (fLengthTmp > fMaxLength)
//		{
//			fMaxLength = fLengthTmp;
//		}
//	}
//
//	////Normalize with the max length
//	//for (unsigned int i = 0; i < nLineCandidates; i++)
//	//{
//	//	fLengths.at(i) = fLengths.at(i)/fMaxLength;
//	//}
//
//	////score with the top n%
//	//const float fPercentageLimit = 1.0f - float(fTopNPercent)/100;
//	////const float fPercentageLimit = float(fTopNPercent)/100;
//	std::vector<unsigned int> fValidEdgeLine;
//	for (unsigned int i = 0; i < nLineCandidates; i++)
//	{
//		if (fLengths.at(i) > fMaxLength*float(fTopNPercent)/100)
//		{
//			fValidEdgeLine.push_back(i);
//		}
//	}
//
//	//Extract the valid line candidates to output
//	for (unsigned int i = 0; i < fValidEdgeLine.size(); i++)
//	{
//		fVPBestNLines.push_back(fVPLineOptimizedInliers.at(fValidEdgeLine.at(i)));
//	}
//
//
//
//
//
//	//unsigned int nLineCandidates = fVPLineOptimizedInliers.size();
//	//std::vector<float> fLengths(nLineCandidates);
//
//	////Determine the max length
//	//float fMaxLength = 0;
//	//for (unsigned int i = 0; i < nLineCandidates; i++)
//	//{
//	//	const float fLengthTmp = fVPLineOptimizedInliers.at(i)->fLineLength;
//	//	fLengths.at(i) = fLengthTmp;
//	//	if (fLengthTmp > fMaxLength)
//	//	{
//	//		fMaxLength = fLengthTmp;
//	//	}
//	//}
//
//	////Normalize with the max length
//	//for (unsigned int i = 0; i < nLineCandidates; i++)
//	//{
//	//	fLengths.at(i) = fLengths.at(i)/fMaxLength;
//	//}
//	//////Normalize with the max length
//	////for (unsigned int i = 0; i < nLineCandidates; i++)
//	////{
//	////	fLengths.at(i) = fLengths.at(i)/fMaxLength;
//	////}
//
//	////score with the top n%
//	//const float fPercentageLimit = 1.0f - float(fTopNPercent)/100;
//	//////score with the top n%
//	////const float fPercentageLimit = 1.0f - float(fTopNPercent)/100;
//	//////const float fPercentageLimit = float(fTopNPercent)/100;
//	//std::vector<unsigned int> fValidEdgeLine;
//	//for (unsigned int i = 0; i < nLineCandidates; i++)
//	//{
//	//	if (fLengths.at(i) > fPercentageLimit)
//	//	if (fLengths.at(i) > fMaxLength*float(fTopNPercent)/100)
//	//	{
//	//		fValidEdgeLine.push_back(i);
//	//	}
//	//}
//
//	////Extract the valid line candidates to output
//	//for (unsigned int i = 0; i < fValidEdgeLine.size(); i++)
//	//{
//	//	fVPBestNLines.push_back(fVPLineOptimizedInliers.at(fValidEdgeLine.at(i)));
//	//}
//}
//
//
//
//void HouseModelBuilder::_DeterminCentroidPoints(const Vec3dPtr &fVPOptimized, const std::vector<VerticalEdgeLinePtr> &fOptimLinesCoefs, PtCloudDataPtr &fIntersectionCloud)
//{
//
//	Eigen::Vector4f fPlaneCentroid;
//	pcl::compute3DCentroid(*m_InputCloud, fPlaneCentroid);
//
//	float fPlaneD = -(fVPOptimized->x * fPlaneCentroid[0] + fVPOptimized->y * fPlaneCentroid[1] + fVPOptimized->z * fPlaneCentroid[2] );
//	
//	Eigen::Vector4f fOptimPlaneCoefs;
//	fOptimPlaneCoefs.resize(4);
//	fOptimPlaneCoefs[0] = fVPOptimized->x;
//	fOptimPlaneCoefs[1] = fVPOptimized->y;
//	fOptimPlaneCoefs[2] = fVPOptimized->z;
//	fOptimPlaneCoefs[3] = fPlaneD;
//
//	unsigned int fNumLines = fOptimLinesCoefs.size();
//	PtCloudDataPtr tmpHullCloud(new PtCloudData);
//	tmpHullCloud->height = 1;
//	tmpHullCloud->width = fNumLines;
//	tmpHullCloud->resize(tmpHullCloud->width * tmpHullCloud->height);
//
//	for (unsigned int i = 0; i < fNumLines; i++)
//	{
//		pcl::PointXYZ PntOfIntersection;
//		_PlaneLineIntersection(fOptimPlaneCoefs, (*fOptimLinesCoefs.at(i)).fOptimLinesCoefs, PntOfIntersection);
//		tmpHullCloud->points[i] = PntOfIntersection;
//	}
//	fIntersectionCloud = tmpHullCloud;
//}
//
//
////ToDo: do things better using Eigen
//void HouseModelBuilder::_PlaneLineIntersection(const Eigen::Vector4f &fOptimPlaneCoefs, const Eigen::VectorXf &fOptimLinesCoefs, pcl::PointXYZ &fPntOfIntersection)
//{
//	//ToDo: Check if the point exist i.e the line does not lie  on the plane!
//
//	const float fNumerator = fOptimLinesCoefs[0]*fOptimPlaneCoefs[0] + fOptimLinesCoefs[1]*fOptimPlaneCoefs[1] + fOptimLinesCoefs[2]*fOptimPlaneCoefs[2] + fOptimPlaneCoefs[3];
//	const float fDenum = fOptimLinesCoefs[3]*fOptimPlaneCoefs[0] + fOptimLinesCoefs[4]*fOptimPlaneCoefs[1] + fOptimLinesCoefs[5]*fOptimPlaneCoefs[2];
//	const float ftParam = - fNumerator/fDenum;
//
//	fPntOfIntersection.x = fOptimLinesCoefs[0] + ftParam*fOptimLinesCoefs[3];
//	fPntOfIntersection.y = fOptimLinesCoefs[1] + ftParam*fOptimLinesCoefs[4];
//	fPntOfIntersection.z = fOptimLinesCoefs[2] + ftParam*fOptimLinesCoefs[5];
//}
//
//
//
//void HouseModelBuilder::_FCloudHull(const PtCloudDataPtr &fPlaneCloud, PtCloudDataPtr &fHullCloud)
//{
//	PtCloudDataPtr cloud_hull (new PtCloudData);
//	pcl::ConvexHull<pcl::PointXYZ> chull;
//	chull.setInputCloud (fPlaneCloud);
//	chull.setDimension(2);
//	//chull.setComputeAreaVolume(true);
//	//chull.setKeepInformation(true);
//	/*chull.setAlpha (0.1);*/
//	chull.reconstruct (*cloud_hull);
//	//	auto rnt = chull.getDim();
//
//	//double fArea = chull.getTotalArea();
//	//double fVolume = chull.getTotalVolume();
//	std::cout<<"hull computation succeed "<<std::endl;
//	*fHullCloud = *cloud_hull ;
//	std::cout<<"hull copying succeeded "<<std::endl;
//
//
//	//	pcl::visualization::PCLVisualizer viewer ("3D Viewer");
//	//
//	//	viewer.setBackgroundColor (0, 0, 0);
//	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle (cloud_hull, 255, 0, 255);
//	//	viewer.addPointCloud (cloud_hull, fOriginalCloudHandle, "Original Cloud");
//	//
//	//
//	//	//for (unsigned int i = 0; i < cloud_hull->points.size()-1; i++)
//	//	//{
//	//	//	pcl::PointXYZ &fLineStart = cloud_hull->points[i];
//	//	//	pcl::PointXYZ &fLineEndPoint=cloud_hull->points[i+1];
//	//	//	std::string fName = " line1" + boost::lexical_cast<std::string, unsigned int>(i); 
//	//	//	viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart, fLineEndPoint, 0.0, 255.0, 0.0, fName, 0);
//	//	////	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, fName);
//	//
//	//	//}
//	//
//	//
//	///*
//	//	viewer0.setBackgroundColor (0, 0, 0);
//	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fLineCloudHandle0 (fInliers, 255, 0, 0);
//	//	std::string fCloudId = "Line Cloud" + boost::lexical_cast<std::string>( 0);;
//	//	viewer0.addPointCloud (fInliers, fLineCloudHandle0, "Line Cloud" );*/
//	//	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Original Cloud");
//	//	viewer.addCoordinateSystem (1.0f);
//	//
//	//	while (!viewer.wasStopped () /*&& range_image_widget.isShown ()*/)
//	//	{
//	//		//	range_image_widget.spinOnce ();
//	//		viewer.spinOnce (100);
//	//		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//	//	}
//
//}
//
//
//bool HouseModelBuilder::_PlanePlaneIntersection(const PtCloudDataPtr &PlaneA, const PtCloudDataPtr &PlaneB, Eigen::VectorXf &LineParams)
//{
//	Eigen::Vector4f fPlaneACoefs, fPlaneBCoefs;
//	_LSPlaneFitting(PlaneA, fPlaneACoefs);
//	_LSPlaneFitting(PlaneB, fPlaneBCoefs);
//
//	return _PlanePlaneIntersection1(fPlaneACoefs, fPlaneBCoefs, LineParams);
//
//	//Vec3d nA = {fPlaneACoefs[0], fPlaneACoefs[1], fPlaneACoefs[2]};
//	//Vec3d nB = {fPlaneBCoefs[0], fPlaneBCoefs[1], fPlaneBCoefs[2]};
//
//	//Math3d::NormalizeVec(nA);
//	//Math3d::NormalizeVec(nB);
//
//	////planes should'nt be parallel
//	//if(VFRMath::IsParallel(nA, nB, 0.1))
//	//	return false;
//
//	//Vec3d fLineDirection;
//	//Math3d::CrossProduct(nA, nB, fLineDirection);
//	//Math3d::NormalizeVec(fLineDirection);
//
//	//Eigen::Vector4f cA, cB;
//	//pcl::compute3DCentroid(*PlaneA, cA);
//	//pcl::compute3DCentroid(*PlaneB, cB);
//
//	//Vec3d fClosestPoint = {0.5*(cB[0] + cA[0]), 0.5*(cA[1] + cB[1]), 0.5*(cA[2] + cB[2])}; //mid-point of centroids
//
//	//Vec3d pA = {cA[0], cA[1], cA[2]};
//	//Vec3d pB = {cB[0], cB[1], cB[2]};
//
//	//float fValA = Math3d::ScalarProduct(pA,nA);
//	//float fValB = Math3d::ScalarProduct(pB,nB);
//
//	//Eigen::MatrixXf fLangegrangeCoefs(5,5);
//	//fLangegrangeCoefs << 2,0,0,nA.x,nB.x,  0,2,0,nA.y,nB.y,  0,0,2,nA.z,nB.z,  nA.x,nA.y,nA.z,0,0,  nB.x,nB.y,nB.z,0,0;   
//
//	////std::cout << "Here is the matrix of the langrange equations minimized:\n" << fLangegrangeCoefs << std::endl;
//
//	//Eigen::VectorXf b;
//	//b.resize(5);
//	//b<< 2*fClosestPoint.x,2*fClosestPoint.y,2*fClosestPoint.z,fValA,fValB; 
//	////std::cout << "Here is the vector b:\n" << b << std::endl;
//	//
//	//Eigen::VectorXf x, xLineParams;
//	//xLineParams.resize(6);
//	//xLineParams[3] = fLineDirection.x;
//	//xLineParams[4] = fLineDirection.y;
//	//xLineParams[5] = fLineDirection.z;
//
//	//x.resize(5);
//	//x = fLangegrangeCoefs.colPivHouseholderQr().solve(b);
//
//	//LineParams.resize(6);
//	//LineParams[0] = x[0];
//	//LineParams[1] = x[1];
//	//LineParams[2] = x[2];
//	//LineParams[3] = fLineDirection.x;
//	//LineParams[4] = fLineDirection.y;
//	//LineParams[5] = fLineDirection.z;
//
//	//return true; //ToDo: do a better return
//}
//
//
//bool HouseModelBuilder::_PlanePlaneIntersection1(const Eigen::Vector4f &fPlaneA, const Eigen::Vector4f &fPlaneB, Eigen::VectorXf &fLineCoefs)
//{
//	/**\brief Determine the line of intersection of two non-parallel planes using lagrange multipliers
//	* \described in: "Intersection of Two Planes, John Krumm, Microsoft Research, Redmond, WA, USA"
//	* \param[in] coefficients of plane A and Plane B in the form ax + by + cz + d = 0
//	* \param[out] coefficients of line where fLinecoefs.tail<3>() = direction vector and 
//	* fLinecoefs.head<3>() the point on the line clossest to (0, 0, 0)
//	* \return true if succeeded/planes aren't parallel
//	*/
//
//	//planes shouldn't be parallel
//	float fAngularTolerance = 0.1;//1e-5; //depending on how perfect you want things to be 
//	float fTestCosine = fPlaneA.head<3>().dot(fPlaneB.head<3>());
//	float fUpperLimit = 1 + fAngularTolerance;
//	float fLowerLimit = 1 - fAngularTolerance;
//
//	if ((fTestCosine < fUpperLimit) && (fTestCosine > fLowerLimit))
//	{
//		//PCL_ERROR ("Plane A and Plane B are Parallel");
//		return (false);
//	}
//
//	if ((fTestCosine > -fUpperLimit) && (fTestCosine < -fLowerLimit))
//	{
//		//PCL_ERROR ("Plane A and Plane B are Parallel");
//		return (false);
//
//	}
//
//	Eigen::Vector4f &fLineDirection = fPlaneA.cross3(fPlaneB);
//	fLineDirection.normalized();
//	
//	//reference point is (0,0,0)
//	Eigen::Vector4f fRefToClosestPoint(0, 0, 0, 0);   
//
//	//construct system of equations using lagrange multipliers with one objective function and two constraints
//	Eigen::MatrixXf fLangegrangeCoefs(5,5);
//	fLangegrangeCoefs << 2,0,0,fPlaneA[0],fPlaneB[0],  0,2,0,fPlaneA[1], fPlaneB[1],  0,0,2, fPlaneA[2], fPlaneB[2], fPlaneA[0], fPlaneA[1] , fPlaneA[2], 0,0, fPlaneB[0], fPlaneB[1], fPlaneB[2], 0,0;   
//
//	Eigen::VectorXf b;
//	b.resize(5);
//	//b << 2*fRefToClosestPoint[0], 2*fRefToClosestPoint[1], 2*fRefToClosestPoint[2], PlaneA[3], PlaneB[3]; 
//	b << 0, 0, 0, -fPlaneA[3], -fPlaneB[3]; 
//	
//	//solve for the lagrange Multipliers
//	Eigen::VectorXf x;
//	x.resize(5);
//	x = fLangegrangeCoefs.colPivHouseholderQr().solve(b);
//
//	fLineCoefs.resize(6);
//	fLineCoefs.head<3>() = x.head<3>(); // the x[3] and x[4] are the values of the lagrange multipliers and are neglected
//	fLineCoefs[3] = fLineDirection[0];
//	fLineCoefs[4] = fLineDirection[1];
//	fLineCoefs[5] = fLineDirection[2];
//
//	//just for melbourne purpose, overite the lines to have directions VP
//	fLineCoefs[3] =_m_computedVP[3];
//	fLineCoefs[4] = _m_computedVP[4];
//	fLineCoefs[5] =_m_computedVP[5];
//
//	return true; 
//}
//
//bool HouseModelBuilder::_PlanePlaneIntersection2(const Eigen::Vector4f &fPlaneA, const Eigen::Vector4f &fPlaneB, Eigen::VectorXf &fLineCoefs)
//{
//	/**\brief Determine the line of intersection of two non-parallel planes using lagrange multipliers
//	* \described in: "Intersection of Two Planes, John Krumm, Microsoft Research, Redmond, WA, USA"
//	* \param[in] coefficients of plane A and Plane B in the form ax + by + cz + d = 0
//	* \param[out] coefficients of line where fLinecoefs.tail<3>() = direction vector and 
//	* fLinecoefs.head<3>() the point on the line clossest to (0, 0, 0)
//	* \return true if succeeded/planes aren't parallel
//	*/
//
//	//planes shouldn't be parallel
//	float fAngularTolerance = 0.1;//1e-5; //depending on how perfect you want things to be 
//	float fTestCosine = fPlaneA.head<3>().dot(fPlaneB.head<3>());
//	float fUpperLimit = 1 + fAngularTolerance;
//	float fLowerLimit = 1 - fAngularTolerance;
//
//	if ((fTestCosine < fUpperLimit) && (fTestCosine > fLowerLimit))
//	{
//		//PCL_ERROR ("Plane A and Plane B are Parallel");
//		return (false);
//	}
//
//	if ((fTestCosine > -fUpperLimit) && (fTestCosine < -fLowerLimit))
//	{
//		//PCL_ERROR ("Plane A and Plane B are Parallel");
//		return (false);
//
//	}
//
//	Eigen::Vector4f &fLineDirection = fPlaneA.cross3(fPlaneB);
//	fLineDirection.normalized();
//	
//	//reference point is (0,0,0)
//	Eigen::Vector4f fRefToClosestPoint(0, 0, 0, 0);   
//
//	//construct system of equations using lagrange multipliers with one objective function and two constraints
//	Eigen::MatrixXf fLangegrangeCoefs(5,5);
//	fLangegrangeCoefs << 2,0,0,fPlaneA[0],fPlaneB[0],  0,2,0,fPlaneA[1], fPlaneB[1],  0,0,2, fPlaneA[2], fPlaneB[2], fPlaneA[0], fPlaneA[1] , fPlaneA[2], 0,0, fPlaneB[0], fPlaneB[1], fPlaneB[2], 0,0;   
//
//	Eigen::VectorXf b;
//	b.resize(5);
//	//b << 2*fRefToClosestPoint[0], 2*fRefToClosestPoint[1], 2*fRefToClosestPoint[2], PlaneA[3], PlaneB[3]; 
//	b << 0, 0, 0, -fPlaneA[3], -fPlaneB[3]; 
//	
//	//solve for the lagrange Multipliers
//	Eigen::VectorXf x;
//	x.resize(5);
//	x = fLangegrangeCoefs.colPivHouseholderQr().solve(b);
//
//	fLineCoefs.resize(6);
//	fLineCoefs.head<3>() = x.head<3>(); // the x[3] and x[4] are the values of the lagrange multipliers and are neglected
//	fLineCoefs[3] = fLineDirection[0];
//	fLineCoefs[4] = fLineDirection[1];
//	fLineCoefs[5] = fLineDirection[2];
//
//	return true; 
//}
//
//
////ToDo: use weighted least squares by weighting with the covariance matrix
//void HouseModelBuilder::_LSPlaneFitting(const PtCloudDataPtr &fInClouds, Eigen::Vector4f &optimized_coefficients)
//{
//
//	//// Needs a valid set of model coefficients
//	//if (model_coefficients.size () != 4)
//	//{
//	//	PCL_ERROR ("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] Invalid number of model coefficients given (%lu)!\n", (unsigned long)model_coefficients.size ());
//	//	optimized_coefficients = model_coefficients;
//	//	return;
//	//}
//
//	//// Need at least 3 points to estimate a plane
//	//if (inliers.size () < 4)
//	//{
//	//	PCL_ERROR ("[pcl::SampleConsensusModelPlane::optimizeModelCoefficients] Not enough inliers found to support a model (%lu)! Returning the same coefficients.\n", (unsigned long)inliers.size ());
//	//	optimized_coefficients = model_coefficients;
//	//	return;
//	//}
//
//	if(fInClouds->points.size() < 4)
//	{
//		std::cout<<"Need at least 3 points to estimate a plane"<<std::endl;
//		return;
//	}
//
//	Eigen::Vector4f plane_parameters;
//
//	// Use ordinary Least-Squares to fit the plane through all the given sample points and find out its coefficients
//	EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
//	Eigen::Vector4f xyz_centroid;
//
//	// Estimate the XYZ centroid
//	pcl::compute3DCentroid (*fInClouds, xyz_centroid);
//	xyz_centroid[3] = 0;
//
//	// Compute the 3x3 covariance matrix
//	pcl::computeCovarianceMatrix (*fInClouds, xyz_centroid, covariance_matrix);
//
//	// Compute the model coefficients
//	EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
//	EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
//	pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);
//
//	// Hessian form (D = nc . p_plane (centroid here) + p)
//	optimized_coefficients.resize (4);
//	optimized_coefficients[0] = eigen_vectors (0, 0);
//	optimized_coefficients[1] = eigen_vectors (1, 0);
//	optimized_coefficients[2] = eigen_vectors (2, 0);
//	optimized_coefficients[3] = 0;
//	optimized_coefficients[3] = -1 * optimized_coefficients.dot (xyz_centroid);
//}
//
//
//bool HouseModelBuilder::_ValidateSegments(const std::vector<EdgeConnectorPtr> &fConnectingEdges, std::vector<EdgeConnectorPtr> &fValidWalls)
//{
//	//int nSegments = fConnectingEdges.size();
//	//std::vector<int> fIndxs(nSegments);
//
//	////generate all indices
//	//for (int i = 0; i < nSegments; i++)
//	//{
//	//	fIndxs[i] = i;
//	//}
//
//	//// collect unique solutions
//	//struct cost
//	//{
//	//	//std::map<std::vector<int>, int> set_;
//
//	//	//bool
//	//	//	operator()(int* first, int* last)
//	//	//{
//	//	//	if (first != last)
//	//	//	{
//	//	//		int len = std::accumulate(first, last, 0);
//	//	//		if (len <= 5600)  // do not collect non-solutions
//	//	//		{
//	//	//			std::vector<int> s(first, last);
//	//	//			std::sort(s.begin(), s.end());
//	//	//			// reject duplicate solutions
//	//	//			set_.insert(std::make_pair(s, 5600-len));
//	//	//		}
//	//	//	}
//	//	//	return false;
//	//	//}
//	//	void myfunction (int i) {
//	//		cout << " " << i;
//	//	}
//	//};
//
//	//struct myclass {
//	//	void operator() (int i) {cout << " " << i;}
//	//} myobject;
//
//
//	//std::vector<int>::iterator r = fIndxs.begin() + 4;
//	//for_each_combination( fIndxs.begin(), r, fIndxs.end(), myobject);
//	//struct F
//	//{
//	//	unsigned long long count_;
//
//	//	F() : count_(0) {}
//
//	//	bool operator()(std::vector<int>::iterator, std::vector<int>::iterator)
//	//	{++count_; return false;}
//	//};
//
//	//int n = 100;
//	//std::vector<int> v(n);
//	//std::iota(v.begin(), v.end(), 0);
//	//std::vector<int>::iterator r = v.begin() + 5;
//	//F f;
//	//do
//	//{
//	//	f(v.begin(), r);
//	//} while (next_combination(v.begin(), r, v.end()));
//
//	////std::vector<EdgeConnectorPtr> fConnectingEdgescopy = fConnectingEdges;
//	//std::vector<EdgeConnectorPtr> fConnectingEdgescopyProcess;
//
//	////std::sort(fConnectingEdgescopy.begin(), fConnectingEdgescopy.end(),[&](EdgeConnectorPtr a, EdgeConnectorPtr b){ return a->fWeight > b->fWeight;});
//	//unsigned int nWalls = fConnectingEdges.size();
//	//	//Check if more than two parallel planes present and quit
//	//for (unsigned int i = 0; i < nWalls; i++)
//	//{
//	//	EdgeConnectorPtr tmpEdge1(new EdgeConnector(*(fConnectingEdges->at(i))));
//	//	for (unsigned int j = i + 1; j < nWalls; j++)
//	//	{
//	//		EdgeConnectorPtr tmpEdge2(new EdgeConnector(*(fConnectingEdges->at(j))));
//	//	
//	//		if (VFRMath::IsParallel(tmpEdge1->PlaneNormal, tmpEdge2->PlaneNormal))
//	//		{
//	//			pcl::PointXYZ fPntOnP1 = tmpEdge1->ProjEdgeSegmentInliers->points[0];
//	//			//Distance between parallel planes should be big!
//	//			double tDisA = pcl::pointToPlaneDistance(fPntOnP1, tmpEdge2->fPlaneCoeficients);
//	//			if (tDisA < 3.0)
//	//			{
//	//				return 0.0;
//	//			}
//
//	///*			for (unsigned int g = j + 1; g < nWalls; g++)
//	//			{
//	//				EdgeConnectorPtr tmpEdge3(new EdgeConnector(*(fWall->at(g))));
//	//				if (VFRMath::IsParallel(tmpEdge1->PlaneNormal, tmpEdge3->PlaneNormal))
//	//				{
//	//					return 0.0;
//	//				}
//	//			}*/
//	//		}
//	//	}
//	//}
//
//
//
//
//	//for ( unsigned int i = 0; i < 120; i++)
//	//{
//	//	fConnectingEdgescopyProcess[i] = fConnectingEdgescopy[i];
//	//}
//
//	//std::cout<<"Smallest segment size: "<<fConnectingEdgescopyProcess[119]->fWeight<<std::endl;
//	//std::cout<<"Biggest segment size: "<<fConnectingEdgescopyProcess[0]->fWeight<<std::endl;*/
//
//
//	const int r = 8;
//	//const int n = 160;//fConnectingEdges.size(); haus51
//	const int n = fConnectingEdges.size(); // haus29
//	std::vector<int> v(n);
//	typedef std::vector<unsigned int> res;
//	  std::vector <res> mres, mres2;
//	std::iota(v.begin(), v.end(), 0);
//	//std::uint64_t count = for_each_combination(v.begin(),
//	//	v.begin() + r,
//	//	v.end(),
//	//	f(v.size(), mres));
//		boost::timer t; // start timing
//
//		WallPtr fConnectingEdgesPtr(new std::vector<EdgeConnectorPtr>(fConnectingEdges));
//		WallPtr fValidWallsPtr(new std::vector<EdgeConnectorPtr>());
//		WallPtr fAllModellsWallsPtr(new std::vector<EdgeConnectorPtr>());
//		std::tuple<WallPtr, std::vector<float>, std::vector<WallPtr> > generatedModel;
//		//generatedModel.first = fValidWallsPtr;
//		//generatedModel.second = fAllModellsWallsPtr;
//
//	generatedModel = for_each_combination(v.begin(),
//		v.begin() + r,
//		v.end(),
//		f(v.size(), fConnectingEdgesPtr));
//	double elapsedTime = t.elapsed();
//	std::cout<< "\n Time used for all combinations is: "<<elapsedTime<<std::endl;
//
//	//Evaluate the best hypothesis!
//	WallPtr fBestHypo(new std::vector<EdgeConnectorPtr>());
//	fBestHypo = std::get<0>(generatedModel);
//	std::cout<<"size of Best hypothesis model: "<<fBestHypo->size()<<std::endl;
//
//	pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//	viewer0.setBackgroundColor (0, 0, 0);
//	/*pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (fPlaneCloud, 255, 0, 255);
//	viewer0.addPointCloud (fPlaneCloud, fOriginalCloudHandle0, "Original Cloud");*/
//
//	viewer0.setBackgroundColor (0.3, 0.3, 0.3);
//
//	
//	
//	for (unsigned int i = 0; i < fBestHypo->size(); i++)
//	{
//
//		std::cout<< "The Best hypothesis has a weights of: "<< fBestHypo->at(i)->fWeight<<std::endl;
//	}
//			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aHandler (fBestHypo->at(0)->ProjEdgeSegmentInliers, 0, 255, 0);
//			viewer0.addPointCloud (fBestHypo->at(0)->ProjEdgeSegmentInliers, aHandler, "clouda");
//				
//			//viewer0.setBackgroundColor (0.3, 0.3, 0.3);
//			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> bHandler (fBestHypo->at(1)->ProjEdgeSegmentInliers, 0, 0, 255);
//			viewer0.addPointCloud (fBestHypo->at(1)->ProjEdgeSegmentInliers, bHandler, "cloudb");
//			//
//			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cHandler (fBestHypo->at(2)->ProjEdgeSegmentInliers, 255,0, 0);
//			viewer0.addPointCloud (fBestHypo->at(2)->ProjEdgeSegmentInliers, cHandler, "cloudc");
//		
//			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> dHandler (fBestHypo->at(3)->ProjEdgeSegmentInliers, 255, 0, 255);
//			viewer0.addPointCloud (fBestHypo->at(3)->ProjEdgeSegmentInliers, dHandler, "cloudd");
//		
//			viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "clouda");
//			viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloudb");
//			viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloudc");
//			viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloudd");
//			viewer0.addCoordinateSystem (1.0f);
//		
//			while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//			{
//				//	range_image_widget.spinOnce ();
//				viewer0.spinOnce (10);
//				boost::this_thread::sleep (boost::posix_time::microseconds(100));
//			}
//
//
//			//Recompute the intersection lines
//			std::vector<Eigen::VectorXf> fIntersLine;
//			for (unsigned int i = 0; i < fBestHypo->size(); i++)
//			{
//				Eigen::Vector4f pA = fBestHypo->at(i)->fPlaneCoeficients;
//				Vec3d nA;
//				nA.x = pA[0];
//				nA.y = pA[1];
//				nA.z = pA[2];
//
//				for (unsigned int j = i+1; j < fBestHypo->size(); j++)
//				{
//					Eigen::Vector4f pB = fBestHypo->at(j)->fPlaneCoeficients;
//					Vec3d nB;
//					nB.x = pB[0];
//					nB.y = pB[1];
//					nB.z = pB[2];
//					if(VFRMath::IsParallel(nA, nB))
//					{
//
//
//						continue;
//					}
//
//					Eigen::VectorXf tmpLine;
//					//recompute the plane intersections
//					bool isnotParallel = _PlanePlaneIntersection2(pA, pB,tmpLine);
//
//					pcl::PointXYZ fPointIntersPnt, fPointIntersPntCentroid, fPointRef1A, fPointRef2A, fPointRef1B, fPointRef2B;
//					fPointIntersPnt.x = tmpLine[0];
//					fPointIntersPnt.y = tmpLine[1];
//					fPointIntersPnt.z = tmpLine[2];
//
//					//project on the centroid plane with normal vp,
//					Eigen::Vector4f fCentroidPlaneCoefficients(fPointIntersPnt.x, fPointIntersPnt.y, fPointIntersPnt.z, 0);// .getVector4fMap();
//					Eigen::Vector4f fVPplaneNormal(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//					fVPplaneNormal.normalize();
//
//					fCentroidPlaneCoefficients[3] = -1*fCentroidPlaneCoefficients.dot(fVPplaneNormal);
//
//					pcl::projectPoint(fPointIntersPnt, fCentroidPlaneCoefficients, fPointIntersPntCentroid);
//
//					fPointRef1A.x = fBestHypo->at(i)->fLineA[0]; 
//					fPointRef1A.y = fBestHypo->at(i)->fLineA[1];
//					fPointRef1A.z = fBestHypo->at(i)->fLineA[2];
//					pcl::PointXYZ pa1,pa2,pb1,pb2;
//					pcl::projectPoint(fPointRef1A, fCentroidPlaneCoefficients, pa1);
//
//					fPointRef2A.x = fBestHypo->at(i)->fLineB[0];
//					fPointRef2A.y = fBestHypo->at(i)->fLineB[1];
//					fPointRef2A.z = fBestHypo->at(i)->fLineB[2];
//					pcl::projectPoint(fPointRef2A, fCentroidPlaneCoefficients, pa2);
//
//					fPointRef1B.x = fBestHypo->at(j)->fLineA[0]; 
//					fPointRef1B.y = fBestHypo->at(j)->fLineA[1];
//					fPointRef1B.z = fBestHypo->at(j)->fLineA[2];
//					pcl::projectPoint(fPointRef1B, fCentroidPlaneCoefficients, pb1);
//
//					fPointRef2B.x = fBestHypo->at(j)->fLineB[0];
//					fPointRef2B.y = fBestHypo->at(j)->fLineB[1];
//					fPointRef2B.z = fBestHypo->at(j)->fLineB[2];
//					pcl::projectPoint(fPointRef2B, fCentroidPlaneCoefficients, pb2);
//					//is it lineA or line B ?
//					
//					float fDisttoA1 = pcl::euclideanDistance(fPointIntersPntCentroid, pa1);
//					float fDisttoA2 = pcl::euclideanDistance(fPointIntersPntCentroid, pa2);
//					float fDisttoB1 = pcl::euclideanDistance(fPointIntersPntCentroid, pb1);
//					float fDisttoB2 = pcl::euclideanDistance(fPointIntersPntCentroid, pb2);
//
//
//
//					//Plot the plane and see!
//
//					float fLineScale = 3.0;
//
//					//test addquad!
//					PtCloudDataPtr quad_cloud_a(new PtCloudData);
//					quad_cloud_a->height = 1;
//					quad_cloud_a->width = 4;
//					quad_cloud_a->resize(quad_cloud_a->height * quad_cloud_a->width);
//
//					quad_cloud_a->points[0].x = pa1.x + fLineScale*_m_computedVP[3]; 
//					quad_cloud_a->points[0].y = pa1.y + fLineScale*_m_computedVP[4]; 
//					quad_cloud_a->points[0].z = pa1.z + fLineScale*_m_computedVP[5]; 
//
//					quad_cloud_a->points[1].x = pa1.x - fLineScale*_m_computedVP[3]; 
//					quad_cloud_a->points[1].y =  pa1.y - fLineScale*_m_computedVP[4]; 
//					quad_cloud_a->points[1].z = pa1.z - fLineScale*_m_computedVP[5]; 
//
//					quad_cloud_a->points[2].x = pa2.x + fLineScale*_m_computedVP[3]; 
//					quad_cloud_a->points[2].y = pa2.y + fLineScale*_m_computedVP[4]; 
//					quad_cloud_a->points[2].z = pa2.z + fLineScale*_m_computedVP[5]; 
//
//
//
//					quad_cloud_a->points[3].x = pa2.x - fLineScale*_m_computedVP[3]; 
//					quad_cloud_a->points[3].y =  pa2.y - fLineScale*_m_computedVP[4]; 
//					quad_cloud_a->points[3].z = pa2.z - fLineScale*_m_computedVP[5]; 
//					PtCloudDataPtr tmpcloud(new PtCloudData);
//
//					_FCloudHull(quad_cloud_a, tmpcloud);
//
//					vtkSmartPointer<vtkActor> actor =
//						vtkSmartPointer<vtkActor>::New();
//					AddQuadActorToRenderer(tmpcloud, actor);
//
//					// Visualize
//					vtkSmartPointer<vtkRenderer> renderer =
//						vtkSmartPointer<vtkRenderer>::New();
//					vtkSmartPointer<vtkRenderWindow> renderWindow =
//						vtkSmartPointer<vtkRenderWindow>::New();
//					renderWindow->AddRenderer(renderer);
//					vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
//						vtkSmartPointer<vtkRenderWindowInteractor>::New();
//					renderWindowInteractor->SetRenderWindow(renderWindow);
//
//					renderer->AddActor(actor);
//					renderer->SetBackground(.5,.3,.31); // Background color salmon
//
//					renderWindow->Render();
//					renderWindowInteractor->Start();
//
//
//					//end plot
//
//
//
//
//
//					//test addquad!
//					PtCloudDataPtr quad_cloud_b(new PtCloudData);
//					quad_cloud_b->height = 1;
//					quad_cloud_b->width = 4;
//					quad_cloud_b->resize(quad_cloud_b->height * quad_cloud_b->width);
//
//					quad_cloud_b->points[0].x = pb1.x + fLineScale*_m_computedVP[3]; 
//					quad_cloud_b->points[0].y = pb1.y + fLineScale*_m_computedVP[4]; 
//					quad_cloud_b->points[0].z = pb1.z + fLineScale*_m_computedVP[5]; ;
//
//					quad_cloud_b->points[1].x = pb2.x + fLineScale*_m_computedVP[3]; 
//					quad_cloud_b->points[1].y = pb2.y + fLineScale*_m_computedVP[4]; 
//					quad_cloud_b->points[1].z = pb2.z + fLineScale*_m_computedVP[5]; 
//
//					quad_cloud_b->points[2].x = pb1.x - fLineScale*_m_computedVP[3]; 
//					quad_cloud_b->points[2].y =  pb1.y - fLineScale*_m_computedVP[4]; 
//					quad_cloud_b->points[2].z = pb1.z - fLineScale*_m_computedVP[5]; 
//
//					quad_cloud_b->points[3].x = pb2.x - fLineScale*_m_computedVP[3]; 
//					quad_cloud_b->points[3].y =  pb2.y - fLineScale*_m_computedVP[4]; 
//					quad_cloud_b->points[3].z = pb2.z - fLineScale*_m_computedVP[5]; 
//					PtCloudDataPtr tmpcloudb(new PtCloudData);
//
//					_FCloudHull(quad_cloud_b, tmpcloudb);
//
//					vtkSmartPointer<vtkActor> actorb =
//						vtkSmartPointer<vtkActor>::New();
//					AddQuadActorToRenderer(tmpcloudb, actorb);
//
//					// Visualize
//					vtkSmartPointer<vtkRenderer> rendererb =
//						vtkSmartPointer<vtkRenderer>::New();
//					vtkSmartPointer<vtkRenderWindow> renderWindowb =
//						vtkSmartPointer<vtkRenderWindow>::New();
//					renderWindowb->AddRenderer(rendererb);
//					vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractorb =
//						vtkSmartPointer<vtkRenderWindowInteractor>::New();
//					renderWindowInteractorb->SetRenderWindow(renderWindow);
//
//					rendererb->AddActor(actorb);
//					rendererb->SetBackground(.5,.3,.31); // Background color salmon
//
//					renderWindowb->Render();
//					renderWindowInteractorb->Start();
//
//
//					//end plot
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//					if (fDisttoA1 < fDisttoA2)
//					{
//						fBestHypo->at(i)->fLineA = tmpLine;
//					}
//					else{
//						fBestHypo->at(i)->fLineB = tmpLine;
//					}
//					if (fDisttoB1 < fDisttoB2)
//					{
//						fBestHypo->at(j)->fLineA = tmpLine;
//
//					}else{
//
//						fBestHypo->at(j)->fLineB = tmpLine;
//					}
//				}
//			}
//
//			std::cout<<"Size of Best Hypothesis: "<<fBestHypo->size()<<std::endl;
//
//			//Determine the centroid of the cloud ? or of the segmentation ?
//			Eigen::Vector4f fCentroid;
//			pcl::compute3DCentroid(*m_InputCloud, fCentroid);
//			pcl::PointXYZ cloudcentroid;
//			cloudcentroid.x = fCentroid[0];
//			cloudcentroid.y = fCentroid[1];
//			cloudcentroid.z = fCentroid[2];
//
//			float fLineScale = 3.30;						
//			std::vector<PtCloudDataPtr> quadPoints;
//			PtCloudDataPtr mRoofQuad(new PtCloudData);
//			mRoofQuad->height = 1;
//			mRoofQuad->width = 4;
//			mRoofQuad->resize(mRoofQuad->width * mRoofQuad->height);
//			//Project the centroid to all the edge lines
//			for (unsigned int i = 0; i < fBestHypo->size(); i++)
//			{
//
//				pcl::PointXYZ pProjA, pProjB;
//				_ProjectPointOnLine(cloudcentroid, fBestHypo->at(i)->fLineA,  pProjA);
//				_ProjectPointOnLine(cloudcentroid, fBestHypo->at(i)->fLineB, pProjB);
//
//				PtCloudDataPtr mQuad(new PtCloudData);
//				mQuad->height = 1;
//				mQuad->width = 4;
//				mQuad->resize(mQuad->width * mQuad->height);
//				
//				//00
//				mQuad->points[0].x = pProjA.x + fLineScale*_m_computedVP[3];
//				mQuad->points[0].y = pProjA.y + fLineScale*_m_computedVP[4];
//				mQuad->points[0].z = pProjA.z + fLineScale*_m_computedVP[5];
//				//01
//				mQuad->points[1].x = pProjA.x - fLineScale*_m_computedVP[3];
//				mQuad->points[1].y = pProjA.y - fLineScale*_m_computedVP[4];
//				mQuad->points[1].z = pProjA.z - fLineScale*_m_computedVP[5];
//
//				//11
//				mQuad->points[3].x = pProjB.x + fLineScale*_m_computedVP[3];
//				mQuad->points[3].y = pProjB.y + fLineScale*_m_computedVP[4];
//				mQuad->points[3].z = pProjB.z + fLineScale*_m_computedVP[5];
//				
//				//10
//				mQuad->points[2].x = pProjB.x - fLineScale*_m_computedVP[3];
//				mQuad->points[2].y = pProjB.y - fLineScale*_m_computedVP[4];
//				mQuad->points[2].z = pProjB.z - fLineScale*_m_computedVP[5];
//
//				PtCloudDataPtr mtmpQuad2(new PtCloudData);
//				_FCloudHull(mQuad, mtmpQuad2);
//
//				quadPoints.push_back(mtmpQuad2);
//				
//				mRoofQuad->points[i] = mQuad->points[0]; //assumming A is UP
//			}
//
//
//
//
//
//			// Setup actor and mapper
//
//			// Setup render window, renderer, and interactor
//			vtkSmartPointer<vtkRenderer> renderer =
//				vtkSmartPointer<vtkRenderer>::New();
//			vtkSmartPointer<vtkRenderWindow> renderWindow =
//				vtkSmartPointer<vtkRenderWindow>::New();
//					renderWindow->AddRenderer(renderer);
//			vtkSmartPointer<vtkActor> actorC =
//				vtkSmartPointer<vtkActor>::New();
//
//			for (unsigned int i = 0; i < fBestHypo->size(); i++)
//			{
//				vtkSmartPointer<vtkActor> actorQ =
//					vtkSmartPointer<vtkActor>::New();
//					AddQuadActorToRenderer(quadPoints.at(i), actorQ);
//
//					double a = rand()/((double) RAND_MAX);
//					double b = rand()/((double) RAND_MAX);
//					double c = rand()/((double) RAND_MAX);
//					actorQ->GetProperty()->SetColor(a,b,c);
//					renderer->AddActor(actorQ);
//			}
//
//			//	sProp->SetOpacity(0.8);
//			//	//sProp->SetOpacity(0.25);
//			//	sProp->SetAmbient(0.5);
//			//	sProp->SetDiffuse(0.6);
//			//	sProp->SetSpecular(1.0);
//			//	//sProp->SetColor(105/255,89/255,205/255);
//			//	sProp->SetSpecularPower(10.0);
//			//	sProp->SetPointSize(2);
//
//			vtkSmartPointer<vtkActor> actorRoof =
//				vtkSmartPointer<vtkActor>::New();
//
//			AddQuadActorToRenderer(mRoofQuad, actorRoof);
//			actorRoof->GetProperty()->SetColor(0.3, 0.5, 0.3);
//
//
//			AddCloudActorToRenderer(m_InputCloud, actorC);
//			actorC->GetProperty()->SetPointSize(3);
//			actorC->GetProperty()->SetColor(.3, .6, .3); // Background color green
//			//actorC->GetProperty()->SetAmbientColor(.1,.1,.1);
//			//actorC->GetProperty()->SetDiffuseColor(.1,.2,.4);
//			//actorC->GetProperty()->SetSpecularColor(1,1,1);
//	
//			vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
//				vtkSmartPointer<vtkRenderWindowInteractor>::New();
//			renderWindowInteractor->SetRenderWindow(renderWindow);
//	
//
//			vtkSmartPointer<vtkVRMLExporter> vrml = vtkSmartPointer<vtkVRMLExporter>::New();
//			/*vrml->SetInput(m_prenWin);*/
//			vrml->SetInput(renderWindow);
//			vrml->SetFileName( "ExportSample.vrml" );
//			vrml->Write();
//
//			renderer->AddActor(actorC);
//			renderer->AddActor(actorRoof);
//			renderer->SetBackground(1, 1, 1); // Background color white
//			renderWindow->Render();
//			renderWindowInteractor->Start();
//
//
//			return false;
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//	fValidWalls = *fValidWallsPtr;
//
//
//
//	//std::uint64_t count = for_each_permutation(v.begin(),
//	//	v.begin() + r,
//	//	v.end(),
//	//	f(v.size()));
//	//// print out "---" to the correct length for the above output
//	//unsigned e = 3 * r + 2;
//	//if (r < v.size())
//	//	e += 1 + 3 * (v.size() - r);
//	//for (unsigned i = 0; i < e; ++i)
//	//	std::cout << '-';
//	//// print out the permuted vector to show that it has the original order
//	//std::cout << "\n[ ";
//	//display(v.begin(), v.end());
//	//std::cout << " ]\n";
//	//// sanity check
//	//assert(count == count_each_permutation(v.begin(), v.begin() + r, v.end()));
//	//// print out summary of what has happened,
//	////   using 'count' from functor state returned from for_each_permutation algorithm.
//	//std::cout << "Found " << count << " permutations of " << v.size()
//	//	<< " objects taken " << r << " at a time.\n";
//
//	boost::this_thread::sleep(boost::posix_time::seconds(20));
//
//
//
//	return false;
//
//}
//
//void HouseModelBuilder::AddQuadActorToRenderer(const PtCloudDataPtr &fQuadEdges, vtkSmartPointer<vtkActor> &fQuadActor)
//{
//
//	//// Create four points (must be in counter clockwise order)
//	/*double p0[3] = {0.0, 0.0, 0.0};
//	double p1[3] = {1.0, 0.0, 0.0};
//	double p2[3] = {1.0, 1.0, 0.0};
//	double p3[3] = {0.0, 1.0, 0.0};*/
//
//
//	double p0[3] = {fQuadEdges->points[0].x, fQuadEdges->points[0].y, fQuadEdges->points[0].z};
//	double p1[3] = {fQuadEdges->points[1].x, fQuadEdges->points[1].y, fQuadEdges->points[1].z};
//	double p2[3] = {fQuadEdges->points[2].x, fQuadEdges->points[2].y, fQuadEdges->points[2].z};
//	double p3[3] = {fQuadEdges->points[3].x, fQuadEdges->points[3].y, fQuadEdges->points[3].z};
//	
//	// Add the points to a vtkPoints object
//	vtkSmartPointer<vtkPoints> points =
//		vtkSmartPointer<vtkPoints>::New();
//	points->InsertNextPoint(p0);
//	points->InsertNextPoint(p1);
//	points->InsertNextPoint(p2);
//	points->InsertNextPoint(p3);
//
//	// Create a quad on the four points
//	vtkSmartPointer<vtkQuad> quad =
//		vtkSmartPointer<vtkQuad>::New();
//	quad->GetPointIds()->SetId(0,0);
//	quad->GetPointIds()->SetId(1,1);
//	quad->GetPointIds()->SetId(2,2);
//	quad->GetPointIds()->SetId(3,3);
//
//	// Create a cell array to store the quad in
//	vtkSmartPointer<vtkCellArray> quads =
//		vtkSmartPointer<vtkCellArray>::New();
//	quads->InsertNextCell(quad);
//
//	// Create a polydata to store everything in
//	vtkSmartPointer<vtkPolyData> polydata =
//		vtkSmartPointer<vtkPolyData>::New();
//
//	// Add the points and quads to the dataset
//	polydata->SetPoints(points);
//	polydata->SetPolys(quads);
//
//	// Setup actor and mapper
//	vtkSmartPointer<vtkPolyDataMapper> mapper =
//		vtkSmartPointer<vtkPolyDataMapper>::New();
//	mapper->SetInput(polydata);
//
//	vtkSmartPointer<vtkActor> actor =
//		vtkSmartPointer<vtkActor>::New();
//	actor->SetMapper(mapper);
//
//	fQuadActor = actor;
//}
//
//
//void HouseModelBuilder::AddCloudActorToRenderer(const PtCloudDataPtr &fCloud, vtkSmartPointer<vtkActor> &fCloudActor)
//{
//	unsigned int nPointsPresent = fCloud->points.size();
//	vtkSmartPointer<vtkPoints> sPoints = vtkSmartPointer<vtkPoints>::New();
//
//	for (int i=0; i < nPointsPresent; i++)
//	{
//		const float  tmpPoint[3] = {fCloud->points[i].x, fCloud->points[i].y, fCloud->points[i].z}; 
//		sPoints->InsertNextPoint(tmpPoint);
//	}
//
//	vtkSmartPointer<vtkPolyData> sPointsPolyData = vtkSmartPointer<vtkPolyData>::New();
//	sPointsPolyData->SetPoints(sPoints);
//
//	vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter =	vtkSmartPointer<vtkVertexGlyphFilter>::New();
//	vertexFilter->SetInputConnection(sPointsPolyData->GetProducerPort());
//	vertexFilter->Update();
//
//	vtkSmartPointer<vtkPolyData> polydata =	vtkSmartPointer<vtkPolyData>::New();
//	polydata->ShallowCopy(vertexFilter->GetOutput());
//
//	//Visualize
//	vtkSmartPointer<vtkPolyDataMapper> mapper = 
//		vtkSmartPointer<vtkPolyDataMapper>::New();
//	mapper->SetInput(polydata);
//	vtkSmartPointer<vtkActor> actor = 
//		vtkSmartPointer<vtkActor>::New();
//	actor->SetMapper(mapper);
//	actor->GetProperty()->SetColor(0.0,255,0.0);
//	fCloudActor = actor;
//}
//
//
//void HouseModelBuilder::testCongress()
//{
//	//read in the data
//	//const std::vector<SegmentationResultPtr> _m_outPlaneSegments;
//
//	//compute the centroid of the segments
//	//Get all planes intersecting the selected plane
//	PtCloudDataPtr segment_1(new PtCloudData), segment_2(new PtCloudData), segment_3(new PtCloudData);
//	SegmentationResultPtr Plane0 = _m_outPlaneSegments.at(0);
//	SegmentationResultPtr Plane1 = _m_outPlaneSegments.at(1);
//	SegmentationResultPtr Plane2= _m_outPlaneSegments.at(2);
//	SegmentationResultPtr Plane3 = _m_outPlaneSegments.at(3);
//
//	VFRMath::ConcatenatePointClouds(Plane1->get<0>(), Plane0->get<0>(), segment_1);
//	VFRMath::ConcatenatePointClouds(Plane2->get<0>(), segment_1, segment_2);
//	VFRMath::ConcatenatePointClouds(Plane3->get<0>(), segment_2, segment_3);
//
//	//Determine the centroid of the cloud ? or of the segmentation ?
//	Eigen::Vector4f fCentroid;
//	pcl::compute3DCentroid(*segment_3, fCentroid);
//	pcl::PointXYZ cloudcentroid;
//	cloudcentroid.x = fCentroid[0];
//	cloudcentroid.y = fCentroid[1];
//	cloudcentroid.z = fCentroid[2];
//
//	////project on the centroid plane with normal vp,
//	//Eigen::Vector4f fCentroidPlaneCoefficients(0.0030848, -0.999579, -0.0288597, 0);// .getVector4fMap();
//	Eigen::Vector4f fCentroidPlaneCoefficients(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);// .getVector4fMap();
//	
//	Eigen::Vector4f fVPplaneNormal(_m_computedVP[3], _m_computedVP[4], _m_computedVP[5], 0);
//	////project on the centroid plane with normal vp,
//	//Eigen::Vector4f fCentroidPlaneCoefficients(0.00 , 0.0, 1.0, 0);// .getVector4fMap();
//	//Eigen::Vector4f fVPplaneNormal(0.00 , 0.0, 1.0, 0);
//
//	fCentroidPlaneCoefficients[3] = -1*fCentroidPlaneCoefficients.dot(fCentroid);
//
////	pcl::projectPoint(fPointIntersPnt, fCentroidPlaneCoefficients, fPointIntersPntCentroid);
//
//
//
//	//save all intersecting pairs
//	std::vector<fWallLinkagePtr> fLinkageMatrix;
//	float fLineScale = 1.70f;
//	//float fLineScale = 7.0f;
//
//	std::vector<PtCloudDataPtr> fWalls;
//	
//
//	//Get all planes intersecting the selected plane
//	for (unsigned int i = 0; i < _m_outPlaneSegments.size(); i++)
//	{
//		SegmentationResultPtr aPlane = _m_outPlaneSegments.at(i);
//
//		fIntersectorsPtr fIntersectingPlanes(new fIntersectors());
//
//		PtCloudDataPtr fPointLineA(new PtCloudData);
//		fPointLineA->width = 4;
//		fPointLineA->height = 1;
//		fPointLineA->resize(fPointLineA->width * fPointLineA->height);
//		unsigned int nPoint = 0;
//
//		for (unsigned int j = 0; j < _m_outPlaneSegments.size(); j ++)
//		{
//			SegmentationResultPtr bPlane = _m_outPlaneSegments.at(j);
//			Eigen::VectorXf tmpLine;
//			bool ifIntersecting = _PlanePlaneIntersection(aPlane->get<0>(), bPlane->get<0>(), tmpLine);
//
//			if (!ifIntersecting)
//			{
//				continue;
//			}
//
//			std::cout<<"Intersecting!!"<<std::endl;
//
//
//		//	//_Startplot cloud 
//		//	pcl::visualization::PCLVisualizer viewer0 ("3D Viewer");
//
//
//		//	viewer0.setBackgroundColor (0, 0, 0);
//		//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> fOriginalCloudHandle0 (aPlane->get<0>(), 0, 255, 0);
//		//	viewer0.addPointCloud (aPlane->get<0>(), fOriginalCloudHandle0, "Original Cloud");
//		//	viewer0.setBackgroundColor (0.0, 0.0, 0.0);
//		//	viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Original Cloud");
//		//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> EdgesInliersHandler0 (bPlane->get<0>(),0, 0, 255);
//		//	unsigned int fRen = 1;
//		////	*std::string fwall_id = "Edges Cloud" + boost::lexical_cast<std::string>(fRen);*/
//		//	viewer0.addPointCloud (bPlane->get<0>(), EdgesInliersHandler0, "Edges Cloud");
//
//		//	viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.3, "Edges Cloud");
//		//	
//		//	pcl::PointXYZ fLineStart_b;
//		//	fLineStart_b.x = tmpLine[0] - fLineScale* tmpLine[3];
//		//	fLineStart_b.y = tmpLine[1] - fLineScale* tmpLine[4];
//		//	fLineStart_b.z = tmpLine[2] - fLineScale* tmpLine[5];
//
//		//	pcl::PointXYZ fLineEndPoint_b;
//		//	fLineEndPoint_b.x = tmpLine[0] + fLineScale* tmpLine[3];
//		//	fLineEndPoint_b.y = tmpLine[1] + fLineScale* tmpLine[4];
//		//	fLineEndPoint_b.z = tmpLine[2] + fLineScale* tmpLine[5];
//
//
//		//	std::string fCloudId2 = "Line Cloud" + boost::lexical_cast<std::string>(11);
//		//	viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId2, 0); 
//		//	viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, fCloudId2);
//		//	//std::string fCloudId_b = "Line Cloud" + boost::lexical_cast<std::string>(fRen++);
//		//	//viewer0.addLine<pcl::PointXYZ, pcl::PointXYZ>(fLineStart_b, fLineEndPoint_b, 0.0, 255.0, 0.0, fCloudId_b, 0); 
//		//	//viewer0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, fCloudId);
//		//	viewer0.addCoordinateSystem (1.0f);
//
//		//	while (!viewer0.wasStopped () /*&& range_image_widget.isShown ()*/)
//		//	{
//		//		//////	range_image_widget.spinOnce ();
//		//		viewer0.spin();
//		//		//viewer0.spinOnce(1000);
//		//		boost::this_thread::sleep(boost::posix_time::seconds(0));
//		//		//boost::this_thread::sleep (boost::posix_time::microseconds (100000000));
//
//		//	}
//		//	//_Endplot
//
//
//			pcl::PointXYZ fPointIntersPnt, fPointIntersPntCentroid, fPointRef1A, fPointRef2A, fPointRef1B, fPointRef2B;
//			fPointIntersPnt.x = tmpLine[0];
//			fPointIntersPnt.y = tmpLine[1];
//			fPointIntersPnt.z = tmpLine[2];
//
//			pcl::projectPoint(fPointIntersPnt, fCentroidPlaneCoefficients, fPointIntersPntCentroid);
//
//			fPointLineA->points[nPoint].x = fPointIntersPntCentroid.x + 0.7*fLineScale*fVPplaneNormal[0];
//			fPointLineA->points[nPoint].y = fPointIntersPntCentroid.y + 0.7*fLineScale*fVPplaneNormal[1];
//			fPointLineA->points[nPoint].z = fPointIntersPntCentroid.z + 0.7*fLineScale*fVPplaneNormal[2];
//			//fPointLineA->points[nPoint].x = fPointIntersPntCentroid.x + 0.7*fLineScale*fVPplaneNormal[0];
//			//fPointLineA->points[nPoint].y = fPointIntersPntCentroid.y + 0.7*fLineScale*fVPplaneNormal[1];
//			//fPointLineA->points[nPoint].z = fPointIntersPntCentroid.z + 0.7*fLineScale*fVPplaneNormal[2];
//			//01
//			nPoint++;
//			fPointLineA->points[nPoint].x = fPointIntersPntCentroid.x - 0.7*fLineScale*fVPplaneNormal[0];
//			fPointLineA->points[nPoint].y = fPointIntersPntCentroid.y - 0.7*fLineScale*fVPplaneNormal[1];
//			fPointLineA->points[nPoint].z = fPointIntersPntCentroid.z - 0.7*fLineScale*fVPplaneNormal[2];
//			nPoint++;
//		}
//					fWalls.push_back(fPointLineA);
//	}
//
//
//	//compute the hulls and plot
//
//	// Setup actor and mapper
//
//	// Setup render window, renderer, and interactor
//	vtkSmartPointer<vtkRenderer> renderer =
//		vtkSmartPointer<vtkRenderer>::New();
//	vtkSmartPointer<vtkRenderWindow> renderWindow =
//		vtkSmartPointer<vtkRenderWindow>::New();
//	renderWindow->AddRenderer(renderer);
//	vtkSmartPointer<vtkActor> actorC =
//		vtkSmartPointer<vtkActor>::New();
//
//	for (unsigned int i = 0; i < 4; i++)
//	{
//		vtkSmartPointer<vtkActor> actorQ =
//			vtkSmartPointer<vtkActor>::New();
//
//		PtCloudDataPtr mtmpQuad2(new PtCloudData);
//		//mtmpQuad2->width = 4;
//		//mtmpQuad2->height = 1;
//		//mtmpQuad2->resize(mtmpQuad2->width * mtmpQuad2->height);
//
//
//		std::cout<<"Failure Point!"<<std::endl;
//
//		/*std::cout<<fWalls.at(i)->points[0].x<<","<<fWalls.at(i)->points[0].y<<","<<fWalls.at(i)->points[0].z<<std::endl;
//		std::cout<<fWalls.at(i)->points[1].x<<","<<fWalls.at(i)->points[1].y<<","<<fWalls.at(i)->points[1].z<<std::endl;
//		std::cout<<fWalls.at(i)->points[2].x<<","<<fWalls.at(i)->points[2].y<<","<<fWalls.at(i)->points[2].z<<std::endl;
//		std::cout<<fWalls.at(i)->points[3].x<<","<<fWalls.at(i)->points[3].y<<","<<fWalls.at(i)->points[3].z<<std::endl;
//
//		std::cout<<fWalls.at(i+1)->points[0].x<<","<<fWalls.at(i+1)->points[0].y<<","<<fWalls.at(i+1)->points[0].z<<std::endl;
//		std::cout<<fWalls.at(i+1)->points[1].x<<","<<fWalls.at(i+1)->points[1].y<<","<<fWalls.at(i+1)->points[1].z<<std::endl;
//		std::cout<<fWalls.at(i+1)->points[2].x<<","<<fWalls.at(i+1)->points[2].y<<","<<fWalls.at(i+1)->points[2].z<<std::endl;
//		std::cout<<fWalls.at(i+1)->points[3].x<<","<<fWalls.at(i+1)->points[3].y<<","<<fWalls.at(i+1)->points[3].z<<std::endl;
//
//		std::cout<<fWalls.at(i+2)->points[0].x<<","<<fWalls.at(i+2)->points[0].y<<","<<fWalls.at(i+2)->points[0].z<<std::endl;
//		std::cout<<fWalls.at(i+2)->points[1].x<<","<<fWalls.at(i+2)->points[1].y<<","<<fWalls.at(i+2)->points[1].z<<std::endl;
//		std::cout<<fWalls.at(i+2)->points[2].x<<","<<fWalls.at(i+2)->points[2].y<<","<<fWalls.at(i+2)->points[2].z<<std::endl;
//		std::cout<<fWalls.at(i+2)->points[3].x<<","<<fWalls.at(i+2)->points[3].y<<","<<fWalls.at(i+2)->points[3].z<<std::endl;
//		
//		std::cout<<fWalls.at(i+3)->points[0].x<<","<<fWalls.at(i+3)->points[0].y<<","<<fWalls.at(i+3)->points[0].z<<std::endl;
//		std::cout<<fWalls.at(i+3)->points[1].x<<","<<fWalls.at(i+3)->points[1].y<<","<<fWalls.at(i+3)->points[1].z<<std::endl;
//		std::cout<<fWalls.at(i+3)->points[2].x<<","<<fWalls.at(i+3)->points[2].y<<","<<fWalls.at(i+3)->points[2].z<<std::endl;
//		std::cout<<fWalls.at(i+3)->points[3].x<<","<<fWalls.at(i+3)->points[3].y<<","<<fWalls.at(i+3)->points[3].z<<std::endl;*/
//
//		//swap 3 and 2
//		pcl::PointXYZ tmp2, tmp3;
//		tmp3 = fWalls.at(i)->points[3];
//		tmp2 = fWalls.at(i)->points[2];
//
//		fWalls.at(i)->points[2] = tmp3;
//		fWalls.at(i)->points[3] = tmp2;
//		
//		//_FCloudHull(fWalls.at(i), mtmpQuad2);
//		AddQuadActorToRenderer(fWalls.at(i), actorQ);
//
//		double a = rand()/((double) RAND_MAX);
//		double b = rand()/((double) RAND_MAX);
//		double c = rand()/((double) RAND_MAX);
//		actorQ->GetProperty()->SetColor(a,b,c);
//		renderer->AddActor(actorQ);
//	}
//
//	//	sProp->SetOpacity(0.8);
//	//	//sProp->SetOpacity(0.25);
//	//	sProp->SetAmbient(0.5);
//	//	sProp->SetDiffuse(0.6);
//	//	sProp->SetSpecular(1.0);
//	//	//sProp->SetColor(105/255,89/255,205/255);
//	//	sProp->SetSpecularPower(10.0);
//	//	sProp->SetPointSize(2);
//
//	/*vtkSmartPointer<vtkActor> actorRoof =
//		vtkSmartPointer<vtkActor>::New();
//
//	AddQuadActorToRenderer(mRoofQuad, actorRoof);
//	actorRoof->GetProperty()->SetColor(0.3, 0.5, 0.3);*/
//
//
//	//downsampled cloud!
//	// Create the filtering object
//	PtCloudDataPtr cloud_filtered(new PtCloudData);
//	pcl::VoxelGrid<pcl::PointXYZ> sor;
//	sor.setInputCloud(m_InputCloud);
//	sor.setLeafSize (0.1f, 0.1f, 0.1f);
//	sor.filter (*cloud_filtered);
//
//	std::cerr << "PointCloud before filtering: " << m_InputCloud->width * m_InputCloud->height 
//		<< " data points (" << pcl::getFieldsList (*m_InputCloud) << ").";
//
//	std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
//		<< " data points (" << pcl::getFieldsList (*cloud_filtered) << ").";
//
//
//	
//	AddCloudActorToRenderer(cloud_filtered, actorC);
//	actorC->GetProperty()->SetPointSize(4);
//	actorC->GetProperty()->SetColor(.3, .6, .3); // Background color green
//	//actorC->GetProperty()->SetAmbientColor(.1,.1,.1);
//	//actorC->GetProperty()->SetDiffuseColor(.1,.2,.4);
//	//actorC->GetProperty()->SetSpecularColor(1,1,1);
//
//	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
//		vtkSmartPointer<vtkRenderWindowInteractor>::New();
//	renderWindowInteractor->SetRenderWindow(renderWindow);
//
//
//	vtkSmartPointer<vtkVRMLExporter> vrml = vtkSmartPointer<vtkVRMLExporter>::New();
//	/*vrml->SetInput(m_prenWin);*/
//	vrml->SetInput(renderWindow);
//	vrml->SetFileName( "ExportSample.vrml" );
//	vrml->Write();
//
//	renderer->AddActor(actorC);
//	//renderer->AddActor(actorRoof);
//	renderer->SetBackground(1, 1, 1); // Background color white
//	renderWindow->Render();
//	renderWindowInteractor->Start();
//}
