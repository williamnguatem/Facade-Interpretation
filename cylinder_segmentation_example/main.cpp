#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/angles.h>

#include <pcl/visualization/cloud_viewer.h>
#include <boost/lexical_cast.hpp> 



void weightedLeastQuaresFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::VectorXf &optimized_coefficients);
void clusterNDirectionPatches(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, Eigen::VectorXf &robust_main_axis);
void cylinderSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cylinder_inliers, Eigen::VectorXf &cylinder_model_coeffs);



//
//
//template <typename PointT> bool
//	pcl::SampleConsensusModelPlane<PointT>::computeModelCoefficients (
//	const std::vector<int> &samples, Eigen::VectorXf &model_coefficients)
//{
//	// Need 3 samples
//	if (samples.size () != 3)
//	{
//		PCL_ERROR ("[pcl::SampleConsensusModelPlane::computeModelCoefficients] Invalid set of samples given (%zu)!\n", samples.size ());
//		return (false);
//	}
//
//	pcl::Array4fMapConst p0 = input_->points[samples[0]].getArray4fMap ();
//	pcl::Array4fMapConst p1 = input_->points[samples[1]].getArray4fMap ();
//	pcl::Array4fMapConst p2 = input_->points[samples[2]].getArray4fMap ();
//
//	// Compute the segment values (in 3d) between p1 and p0
//	Eigen::Array4f p1p0 = p1 - p0;
//	// Compute the segment values (in 3d) between p2 and p0
//	Eigen::Array4f p2p0 = p2 - p0;
//
//	// Avoid some crashes by checking for collinearity here
//	Eigen::Array4f dy1dy2 = p1p0 / p2p0;
//	if ( (dy1dy2[0] == dy1dy2[1]) && (dy1dy2[2] == dy1dy2[1]) )          // Check for collinearity
//		return (false);
//
//	// Compute the plane coefficients from the 3 given points in a straightforward manner
//	// calculate the plane normal n = (p2-p1) x (p3-p1) = cross (p2-p1, p3-p1)
//	model_coefficients.resize (4);
//	model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
//	model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
//	model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
//	model_coefficients[3] = 0;
//
//	// Normalize
//	model_coefficients.normalize ();
//
//	// ... + d = 0
//	model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));
//
//	return (true);
//}
//
//





void cylinderSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cylinder_inliers, Eigen::VectorXf &cylinder_model_coeffs)
{
	int num_point_canditates = in_cloud->points.size();
	unsigned int num_iterations = 5000;
	float ransac_tolerance = 0.06;

	// Consider using more precise timers such as gettimeofday on *nix or
	// GetTickCount/timeGetTime/QueryPerformanceCounter on Windows.
	boost::mt19937 randGen(std::time(0));

	// Now we set up a distribution. Boost provides a bunch of these as well.
	// This is the preferred way to generate numbers in a certain range.
	// initialize a uniform distribution between 0 and the max=num_point_canditates

	boost::uniform_int<> uInt8Dist(0, num_point_canditates);

	// Finally, declare a variate_generator which maps the random number
	// generator and the distribution together. This variate_generator
	// is usable like a function call.
	boost::variate_generator< boost::mt19937&, boost::uniform_int<> > 
		getRandNumber(randGen, uInt8Dist);

	float best_radius = 0.0f;
	unsigned best_index, max_counter = 0;
	Eigen::Vector4f best_main_axis (0.0, 0.0, 0.0, 0.0);
	Eigen::Vector4f best_center (0.0, 0.0, 0.0, 0.0);

	for (unsigned int i = 0; i < num_iterations; i++)
	{
		// identify 3 different points
		const int first_index = getRandNumber() % num_point_canditates;
		int temp_index;
		do { temp_index = getRandNumber() % num_point_canditates; } while (temp_index == first_index);
		const int second_index = temp_index;
		do { temp_index = getRandNumber() % num_point_canditates; } while (temp_index == first_index || temp_index == second_index);

		pcl::PointXYZ center_c;
		Eigen::Vector4f p0 = in_cloud->points[first_index].getVector4fMap();
		Eigen::Vector4f p1 = in_cloud->points[second_index].getVector4fMap();
		Eigen::Vector4f p2 = in_cloud->points[temp_index].getVector4fMap();

		center_c.x = 1/3*(p0(0) + p1(0) + p2(0));
		center_c.y = 1/3*(p0(1) + p1(1) + p2(1));
		center_c.z = 1/3*(p0(2) + p1(2) + p2(2));;


		// Compute the segment values (in 3d) between p1 and p0
		Eigen::Vector4f p1p0 = p1 - p0;
		// Compute the segment values (in 3d) between p2 and p0
		Eigen::Vector4f p2p0 = p2 - p0;

		//// Avoid some crashes by checking for collinearity here
		//Eigen::Vector4f dy1dy2 = p1p0 / p2p0;
		//if ( (dy1dy2[0] == dy1dy2[1]) && (dy1dy2[2] == dy1dy2[1]) )          // Check for collinearity
		//	return ;

		// Compute the plane coefficients from the 3 given points in a straightforward manner
		// calculate the plane normal n = (p2-p1) x (p3-p1) = cross (p2-p1, p3-p1)
		Eigen::VectorXf tmp_model_coefficients;
		tmp_model_coefficients.resize (4);
		tmp_model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
		tmp_model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
		tmp_model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
		tmp_model_coefficients[3] = 0;

		// Normalize
		tmp_model_coefficients.normalize ();

		// ... + d = 0
	/*	tmp_model_coefficients[3] = -1 * (tmp_model_coefficients.template head<4>().dot (p0.matrix ()));*/
	//	tmp_model_coefficients[3] = -1 * tmp_model_coefficients

		//get centroid of p0,p1,p2
		//get radius of cylinder = distance between 
		float cylinder_radius = (center_c.getVector4fMap()-p0).squaredNorm();

	/*	float cylinder_radius0 = pcl::euclideanDistance(p0, center_c.getVector4fMap());
		float cylinder_radius1 = pcl::euclideanDistance(p2, center_c.getVector4fMap());*/

		//score model
		unsigned int tmp_counter = 0;
		Eigen::Vector4f line_pt (center_c.x, center_c.y, center_c.z, 0); 
		Eigen::Vector4f line_dir(tmp_model_coefficients(0), tmp_model_coefficients(1), tmp_model_coefficients(2), 0);

		for (unsigned int j = 0; j < num_point_canditates; j++)
		{
		 float distances = sqrt ((line_pt - in_cloud->points[j].getVector4fMap ()).cross3 (line_dir).squaredNorm ());
		 if (fabsf( distances - cylinder_radius) < ransac_tolerance)
		 {
		 tmp_counter++;
		 }
		 	}

		//check if temp model is the best
		if(tmp_counter > max_counter)
		{
			max_counter = tmp_counter;
			best_center = line_pt;
			best_main_axis = line_dir;
			best_radius = cylinder_radius;
		}
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers(new pcl::PointCloud<pcl::PointXYZ>);
	cloud_inliers->height = 1;
	cloud_inliers->width = max_counter;
	cloud_inliers->resize(max_counter);

	//filter inliers!
	unsigned int h = 0;
	for (unsigned int j = 0; j < num_point_canditates; j++)
	{
		float distances = sqrt ((best_center - in_cloud->points[j].getVector4fMap ()).cross3 (best_main_axis).squaredNorm ());

		if (fabsf( distances - best_radius) < ransac_tolerance)
		{
			cloud_inliers->points[h] = in_cloud->points[j];
			h++;
		}
	}
	cylinder_model_coeffs.resize(7);
	cylinder_model_coeffs.head<3>() = best_center.head<3>();
	cylinder_model_coeffs.segment<3>(3) = best_main_axis.head<3>();
	cylinder_model_coeffs(6) = best_radius;

	cylinder_inliers = cloud_inliers;

}




void clusterNDirectionPatches(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, Eigen::VectorXf &robust_main_axis)
{
	unsigned int num_of_normals = in_cloud->points.size() * 50/100;
	unsigned int num_of_xproducts = 10000;
	unsigned  int num_of_iterations = 5000;
	unsigned int k = 18;
	double angular_tolerance = 1.0;
	Eigen::Vector4f estimated_main_axis;

	int num_point_canditates = in_cloud->points.size();

	// Consider using more precise timers such as gettimeofday on *nix or
	// GetTickCount/timeGetTime/QueryPerformanceCounter on Windows.
	boost::mt19937 randGen(std::time(0));

	// Now we set up a distribution. Boost provides a bunch of these as well.
	// This is the preferred way to generate numbers in a certain range.
	// initialize a uniform distribution between 0 and the max=num_point_canditates

	boost::uniform_int<> uInt8Dist(0, num_point_canditates);

	// Finally, declare a variate_generator which maps the random number
	// generator and the distribution together. This variate_generator
	// is usable like a function call.
	boost::variate_generator< boost::mt19937&, boost::uniform_int<> > 
		getRandNumber(randGen, uInt8Dist);

	//sample random points and compute their normals
	pcl::IndicesPtr indices(new  std::vector<int>());
	for (unsigned int i = 0; i < num_of_normals; i++)
	{
		indices->push_back(getRandNumber() % num_point_canditates);      
	}

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (in_cloud);
	ne.setSearchMethod (pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
	ne.setKSearch (k);
	//ne.setRadiusSearch (fRadius); //radius search is may be more reliable if the cloud metric scale is known
	ne.setIndices(indices); //compute normals only for the randomly selected indices

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
	ne.compute (*cloud_normals);

	//typedef boost::scoped_ptr<Eigen::Vector4f> fXProds;
	//change to scopeptr ?
	std::vector<boost::shared_ptr<Eigen::Vector4f> > main_axis_hypothesis;

	//sample random points and compute their normals
	for (unsigned int i = 0; i < num_of_xproducts; i++)
	{
		unsigned int index_a = getRandNumber() % num_of_normals;
		unsigned int index_b = getRandNumber() % num_of_normals;

		pcl::Normal &normal_a = cloud_normals->points[index_a];
		pcl::Normal &normal_b = cloud_normals->points[index_b];

		Eigen::Vector4f n_a(normal_a.normal_x, normal_a.normal_y, normal_a.normal_z, 0); 
		Eigen::Vector4f n_b(normal_b.normal_x, normal_b.normal_y, normal_b.normal_z, 0);
		Eigen::Vector4f n_c = n_a.cross3(n_b);

		//change to scopeptr
		boost::shared_ptr<Eigen::Vector4f> tmp_xproducts(new Eigen::Vector4f(n_c));
		main_axis_hypothesis.push_back(tmp_xproducts);
	}

	//the cross products are then the direction we are looking. Find the majority w.r.t a ref direction!
	Eigen::Vector4f ref_direction (1.0, 0.0, 0.0, 0);

	//validate the best using RANSAC w.r.t ref_direction
	unsigned int max_counter = 0;
	int best_index = -1;
	for (unsigned int g = 0; g < num_of_iterations; g++)
	{
		unsigned int h0_index = getRandNumber() % num_of_xproducts;
		Eigen::Vector4f &h0 = *main_axis_hypothesis.at(h0_index);
		double angle_in_radians = pcl::getAngle3D(h0, ref_direction);
		double angle_in_degrees = pcl::rad2deg(angle_in_radians);

		//score the selected hypothesis
		unsigned int h0_counter = 0 ;
		for (unsigned int l = 0; l < num_of_xproducts; l++)
		{
			//const float sAnglesInRadianstmp = Math3d::Angle(*((*crosProds)[l]), sRefDir);
			//const float sAnglesInDegreestmp = sAnglesInRadianstmp * 180.0f / CV_PI;

			Eigen::Vector4f &tmp_h = *main_axis_hypothesis.at(l);
			double tmp_angles_in_radians = pcl::getAngle3D(tmp_h, ref_direction);
			double tmp_angle_in_degrees = pcl::rad2deg(tmp_angles_in_radians);

			if (fabsf(angle_in_degrees - tmp_angle_in_degrees) <= angular_tolerance) //Tolerance of 2 Degress!!!
			{
				h0_counter++;
			}
		}

		if (h0_counter > max_counter)
		{	
			max_counter = h0_counter;
			best_index = h0_index;
		}
	}

	//collect all the inliers and do Weighted Least Squares Fit
	//(best_index == -1) ? return : ; //sanity check
	if (best_index == -1)
	{
		return ;
	}

	Eigen::Vector4f &best_model_hypothesis = *main_axis_hypothesis.at(best_index);
	double best_angle_radians = pcl::getAngle3D(best_model_hypothesis, ref_direction);
	double best_angle_degrees = pcl::rad2deg(best_angle_radians);

	pcl::PointCloud<pcl::PointXYZ>::Ptr model_inliers(new pcl::PointCloud<pcl::PointXYZ>);
	model_inliers->width = max_counter;
	model_inliers->height = 1;
	model_inliers->resize(model_inliers->width * model_inliers->height);
	unsigned int ii = 0;

	for (unsigned int l = 0; l < num_of_xproducts; l++)
	{
		Eigen::Vector4f &tmp_h = *main_axis_hypothesis.at(l);
		double tmp_angles_in_radians = pcl::getAngle3D(tmp_h, ref_direction);
		double tmp_angle_in_degrees = pcl::rad2deg(tmp_angles_in_radians);

		if (fabsf(best_angle_degrees - tmp_angle_in_degrees) <= angular_tolerance) //Tolerance of 2 Degress!!!
		{
			model_inliers->points[ii].x = tmp_h(0);
			model_inliers->points[ii].y = tmp_h(1);
			model_inliers->points[ii].z = tmp_h(2);
			ii++;
		}
	}

	//Robust fitting the inliers using WleastSquears fit
	weightedLeastQuaresFitting(model_inliers, robust_main_axis);
}


void weightedLeastQuaresFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::VectorXf &optimized_coefficients)
{
	optimized_coefficients.resize (6);

	// Compute the 3x3 covariance matrix
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid (*cloud, centroid);

	Eigen::Matrix3f covariance_matrix;
	//pcl::computeCovarianceMatrix(*input_, inliers, centroid, covariance_matrix);
	pcl::computeCovarianceMatrix(*cloud, centroid, covariance_matrix);
	optimized_coefficients[0] = centroid[0];
	optimized_coefficients[1] = centroid[1];
	optimized_coefficients[2] = centroid[2];

	// Extract the eigenvalues and eigenvectors
	EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
	EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

	optimized_coefficients.tail<3> () = eigen_vectors.col (2).normalized ();
}






typedef pcl::PointXYZ PointT;

int
	main (int argc, char** argv)
{
	// read input data
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_inliers(new pcl::PointCloud<pcl::PointXYZ>);
	reader.read ("E:\\21_05_2012_Shot1_DN15_0Grad_Int5.pcd", *cloud);

	float line_scale = 3.0f;

	// Approach 1: Principal Component Analysis (PCA) and get the main Axis  
	//*********************************************************************

	Eigen::VectorXf cloud_main_axis_pca;
	cloud_main_axis_pca.resize(6);

	// Compute the 3x3 covariance matrix of cloud
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid (*cloud, centroid);

	Eigen::Matrix3f covariance_matrix;
	pcl::computeCovarianceMatrix(*cloud, centroid, covariance_matrix);
	cloud_main_axis_pca[0] = centroid[0];
	cloud_main_axis_pca[1] = centroid[1];
	cloud_main_axis_pca[2] = centroid[2];

	// Extract the eigenvalues and eigenvectors
	EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
	EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);
	cloud_main_axis_pca.tail<3> () = eigen_vectors.col (2).normalized ();

	//visualize 
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer0(new pcl::visualization::PCLVisualizer("3D Viewer")) ;
	viewer0->setBackgroundColor (0, 0, 0);

	//std::string name_pca = "results_"+ boost::lexical_cast<std::string>(i)+ "_" +tmp_path.stem().string()+ ".jpg";
	//std::cout<<"result_image_name: "<<result_name<<std::endl;
	//boost::this_thread::sleep (boost::posix_time::microseconds (10000000));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> original_cloud_handler (cloud, 255, 0, 0);
	std::string cloud_id = "org_cloud";
	viewer0->addPointCloud (cloud, original_cloud_handler, cloud_id );
	viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_id);
	viewer0->addCoordinateSystem (1.0f);

	//define endings of the quad
	pcl::PointXYZ line_start;
	line_start.x = centroid[0] - line_scale * cloud_main_axis_pca[3];
	line_start.y = centroid[1] - line_scale * cloud_main_axis_pca[4];
	line_start.z = centroid[2] - line_scale * cloud_main_axis_pca[5];

	pcl::PointXYZ line_end;
	line_end.x = centroid[0] + line_scale * cloud_main_axis_pca[3];;
	line_end.y = centroid[1] + line_scale * cloud_main_axis_pca[4];
	line_end.z = centroid[2] + line_scale * cloud_main_axis_pca[5];
	viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(line_start, line_end, 0.0, 255.0, 0.0, "line_pca", 0); 

	Eigen::Vector4f ref_dir(0.0, 0.0, 1.0, 0.0);
	Eigen::Vector4f pca_dir(cloud_main_axis_pca(3), cloud_main_axis_pca(4), cloud_main_axis_pca(5), 0);
	double tmp_angles_in_radians = pcl::getAngle3D(pca_dir, ref_dir);
	double tmp_angle_in_degrees = pcl::rad2deg(tmp_angles_in_radians);
	std::cout << "PCA approach. Angle to Z-Axis in degrees: "<< tmp_angle_in_degrees<<std::endl;


	//Approach 2: Cluster cross products of point normals

	Eigen::VectorXf cloud_main_axis_clustering;
	cloud_main_axis_clustering.resize(6);
	clusterNDirectionPatches(cloud, cloud_main_axis_clustering);

	//define line endings
	pcl::PointXYZ line_start_2;
	line_start_2.x = centroid[0] - line_scale * cloud_main_axis_clustering[3];
	line_start_2.y = centroid[1] - line_scale * cloud_main_axis_clustering[4];
	line_start_2.z = centroid[2] - line_scale * cloud_main_axis_clustering[5];

	pcl::PointXYZ line_end_2;
	line_end_2.x = centroid[0] + line_scale * cloud_main_axis_clustering[3];;
	line_end_2.y = centroid[1] + line_scale * cloud_main_axis_clustering[4];
	line_end_2.z = centroid[2] + line_scale * cloud_main_axis_clustering[5];

	viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(line_start_2, line_end_2, 255.0, 0.0, 0.0, "line_2", 0); 

	Eigen::Vector4f clustering_dir(cloud_main_axis_clustering(3), cloud_main_axis_clustering(4),cloud_main_axis_clustering(5), 0);
	double tmp_angles_in_radians_0 = pcl::getAngle3D(clustering_dir, ref_dir);
	double tmp_angle_in_degrees_0 = pcl::rad2deg(tmp_angles_in_radians_0);
	std::cout << "Clustering Approach. Angle to Z-Axis in degrees: "<< tmp_angle_in_degrees_0<<std::endl;

	
	
	//Approach 3: Cylinder detection
	//*********************************************************
	Eigen::VectorXf cylinder_model_coeffs;

	pcl::PassThrough<PointT> pass;
	pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
	// Build a passthrough filter to remove spurious NaNs
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.3, 0.9);
	pass.filter (*cloud_filtered2);
	std::cerr << "PointCloud after filtering has: " << cloud_filtered2->points.size () << " data points." << std::endl;


	cylinderSegmentation(cloud_filtered2, cylinder_inliers, cylinder_model_coeffs);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> inliers_cloud_handler (cylinder_inliers, 0.0, 0.0, 255);
	std::string inliers_cloud_id = "inliers_cloud";
	viewer0->addPointCloud (cylinder_inliers, inliers_cloud_handler, inliers_cloud_id);
	viewer0->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, inliers_cloud_id);

	//define line endings
	pcl::PointXYZ line_start_3;
	line_start_3.x = centroid[0] - line_scale * cylinder_model_coeffs[3];
	line_start_3.y = centroid[1] - line_scale * cylinder_model_coeffs[4];
	line_start_3.z = centroid[2] - line_scale * cylinder_model_coeffs[5];

	pcl::PointXYZ line_end_3;
	line_end_3.x = centroid[0] + line_scale * cylinder_model_coeffs[3];;
	line_end_3.y = centroid[1] + line_scale * cylinder_model_coeffs[4];
	line_end_3.z = centroid[2] + line_scale * cylinder_model_coeffs[5];

	viewer0->addLine<pcl::PointXYZ, pcl::PointXYZ>(line_start_3, line_end_3, 0.0, 0.0, 255, "line_3", 0);

	Eigen::Vector4f cylinder_ransac_dir(cylinder_model_coeffs(3), cylinder_model_coeffs(4), cylinder_model_coeffs(5), 0);
	double tmp_angles_in_radians_1 = pcl::getAngle3D(cylinder_ransac_dir, ref_dir);
	double tmp_angle_in_degrees_1 = pcl::rad2deg(tmp_angles_in_radians_1);
	std::cout << "Cylinder detection (RANSAC). Angle to Z-Axis in degrees: "<< tmp_angle_in_degrees_1<<std::endl;



	while (!viewer0->wasStopped ())
	{
		viewer0->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}

	//pcl::PassThrough<PointT> pass;
	//pcl::NormalEstimation<PointT, pcl::Normal> ne;
	//pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
	//pcl::PCDWriter writer;
	//pcl::ExtractIndices<PointT> extract;
	//pcl::ExtractIndices<pcl::Normal> extract_normals;
	//pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

	//// Datasets
	//pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
	//pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
	//pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	//pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
	//pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
	//pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
	//pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);


	//std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;

	//// Build a passthrough filter to remove spurious NaNs
	//pass.setInputCloud (cloud);
	//pass.setFilterFieldName ("z");
	//pass.setFilterLimits (0, 1.5);
	//pass.filter (*cloud_filtered);
	//std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

	//// Estimate point normals
	//ne.setSearchMethod (tree);
	//ne.setInputCloud (cloud_filtered);
	//ne.setKSearch (50);
	//ne.compute (*cloud_normals);

	//// Create the segmentation object for the planar model and set all the parameters
	//seg.setOptimizeCoefficients (true);
	//seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
	//seg.setNormalDistanceWeight (0.1);
	//seg.setMethodType (pcl::SAC_RANSAC);
	//seg.setMaxIterations (100);
	//seg.setDistanceThreshold (0.03);
	//seg.setInputCloud (cloud_filtered);
	//seg.setInputNormals (cloud_normals);
	//// Obtain the plane inliers and coefficients
	//seg.segment (*inliers_plane, *coefficients_plane);
	//std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

	//// Extract the planar inliers from the input cloud
	//extract.setInputCloud (cloud_filtered);
	//extract.setIndices (inliers_plane);
	//extract.setNegative (false);

	//// Write the planar inliers to disk
	//pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
	//extract.filter (*cloud_plane);
	//std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
	//writer.write ("data\\table_scene_mug_stereo_textured_plane.pcd", *cloud_plane, false);

	//// Remove the planar inliers, extract the rest
	//extract.setNegative (true);
	//extract.filter (*cloud_filtered2);
	//extract_normals.setNegative (true);
	//extract_normals.setInputCloud (cloud_normals);
	//extract_normals.setIndices (inliers_plane);
	//extract_normals.filter (*cloud_normals2);

	//// Create the segmentation object for cylinder segmentation and set all the parameters
	//seg.setOptimizeCoefficients (true);
	//seg.setModelType (pcl::SACMODEL_CYLINDER);
	//seg.setMethodType (pcl::SAC_RANSAC);
	//seg.setNormalDistanceWeight (0.1);
	//seg.setMaxIterations (10000);
	//seg.setDistanceThreshold (0.05);
	//seg.setRadiusLimits (0, 0.1);
	//seg.setInputCloud (cloud_filtered2);
	//seg.setInputNormals (cloud_normals2);

	//// Obtain the cylinder inliers and coefficients
	//seg.segment (*inliers_cylinder, *coefficients_cylinder);
	//std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

	//// Write the cylinder inliers to disk
	//extract.setInputCloud (cloud_filtered2);
	//extract.setIndices (inliers_cylinder);
	//extract.setNegative (false);
	//pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
	//extract.filter (*cloud_cylinder);
	//if (cloud_cylinder->points.empty ()) 
	//	std::cerr << "Can't find the cylindrical component." << std::endl;
	//else
	//{
	//	std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
	//	writer.write ("data\\table_scene_mug_stereo_textured_cylinder.pcd", *cloud_cylinder, false);
	//}
	return (0);
}