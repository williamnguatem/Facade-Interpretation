#include "vp_detection_wrapper.h"
#include "fi_config.h"
#include <boost/lexical_cast.hpp> 
#include <boost/regex.hpp>
#include <pcl/common/angles.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>



fi::VPDetectionWrapper::VPDetectionWrapper()
{

}


fi::VPDetectionWrapper::VPDetectionWrapper(const std::vector<std::vector<std::string> > &corresponding_images_filenames, const std::string &out_put_dir)
	: m_corresponding_images_filenames(corresponding_images_filenames), m_out_put_dir(out_put_dir)
{

}

fi::VPDetectionWrapper::~VPDetectionWrapper()
{

}

bool fi::VPDetectionWrapper::collectVanishingPoints(std::vector<std::vector<Eigen::Vector2f> > &sets_of_vanishing_point)
{
	std::vector<std::string> file_names;
	getUniqueFileNames(m_corresponding_images_filenames, file_names);

	unsigned int number_of_images = file_names.size();
	for (unsigned int i = 0; i < number_of_images; i++ )
	{
		computeVanishingPoint(file_names.at(i), m_out_put_dir);
	}


	//collect all the vp outputs and parse
	//boost::filesystem::path slash("/");
	//std::string preferredSlash = slash.make_preferred().native();
	
	boost::filesystem::path p("/"); 
	std::string vp_results_out_dir(m_out_put_dir + p.string() + VP_RESULTS_DIR);
	vp_results_out_dir = correctSlashes(vp_results_out_dir);
	if ( !boost::filesystem::exists( vp_results_out_dir ) ) return false;
	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	boost::regex mapping_filter("vpoint.\*.txt");//used to matched the .cal or .calrd	
	for ( boost::filesystem::directory_iterator itr( vp_results_out_dir );itr != end_itr; ++itr )
	{
		boost::smatch what_to_match;
		std::cout<<"file name: "<<itr->path().string()<<std::endl;

		//if ( boost::filesystem::is_directory(itr->status()) )
		//{
		//	if ( get3D2DTableFile( itr->path(), /*file_name,*/ path_found ) ) return true;
		//}else 
		if ( !boost::filesystem::is_regular_file( itr->status() ) )
			continue; // Skip if not a file 
		if ( !boost::regex_search(itr->path().string(), what_to_match, mapping_filter ) ) 
			continue; // Skip if no match

		std::vector<Eigen::Vector2f> vp_hypothesis;
		parseVPOutputFile(itr->path().string(), vp_hypothesis);
		sets_of_vanishing_point.push_back(vp_hypothesis);
		std::cout<<"file parsed successfully"<<std::endl;
	}

	if (sets_of_vanishing_point.size() == 0)
	{
		return false;
	}
	std::cout<<"All files parsed successfully"<<std::endl;
	return true;
}


void fi::VPDetectionWrapper::getUniqueFileNames(const std::vector<std::vector<std::string> > &corresponding_images_filenames, std::vector<std::string> &file_names)
{
	//collect all the files plus multiples
	unsigned int i = corresponding_images_filenames.size();
	std::vector<std::string>  multiple_file_names;
	for (unsigned int j = 0; j < i; j++)
	{
		std::vector<std::string>  tmp_files = corresponding_images_filenames.at(j);
		unsigned int k = tmp_files.size();
		for (unsigned int h = 0; h < k; h++)
		{
			multiple_file_names.push_back(tmp_files.at(h));
		}
	}

	removeDuplicates(multiple_file_names);
	file_names = multiple_file_names;
}


void fi::VPDetectionWrapper::computeVanishingPoint(const std::string &image_filename, const std::string &vp_out_dir_name)
{
	//create results folder 
	//boost::filesystem::path sep_tor = boost::filesystem::path("/").native();

	boost::filesystem::path slash("/");
	//std::string preferred_slash = slash.make_preferred().native().string();

	std::string vp_results_out_dir(vp_out_dir_name + slash.string() + VP_RESULTS_DIR);
	vp_results_out_dir = correctSlashes(vp_results_out_dir);
	boost::filesystem::create_directory(vp_results_out_dir);
	//std::string vp_tool(VPDETECTION_TOOL);
	char *a = VPDETECTION_TOOL;
	std::string b ( boost::lexical_cast<std::string>(a)+ " --outdir="+ vp_results_out_dir + " "+ image_filename);
	b=correctSlashes(b);
	std::cout <<"command: "<<b<<std::endl;
	//system(b.c_str());  //undo this for new data sets
}


void fi::VPDetectionWrapper::parseVPOutputFile(const std::string &vp_results_file, std::vector<Eigen::Vector2f> &vp_hypothesis)
{
	std::string a_line;
	unsigned int num_lines = 0; // the radial distortion params is on the 6 and 7th line!
	//std::ifstream param_file ("E:\\21\\Result\\UniBW\\10884_param.txt");
	std::ifstream param_file (vp_results_file.c_str());
	if (param_file.is_open())
	{
		while ( param_file.good() )
		{
			if (num_lines > 2)
			{
				break;
			}

			getline (param_file, a_line);
			Eigen::Vector2f tmpVP;
			if (a_line == "")
			{
				tmpVP(0) = 0;
				tmpVP(1) = 0;
				vp_hypothesis.push_back(tmpVP);
				break;
			}

			std::vector< std::string> r_vals;
			boost::split(r_vals, a_line, boost::is_any_of(";"));

			tmpVP(0) = boost::lexical_cast<float>(r_vals.at(0));
			tmpVP(1) = boost::lexical_cast<float>(r_vals.at(1));
			vp_hypothesis.push_back(tmpVP);
			num_lines++;
		}
	}
	else
	{
		return;
	}
}


void fi::VPDetectionWrapper::validateVanishingPoint(const std::vector<std::vector< Eigen::Vector2f> > &computed_vp_hypothesis, const Eigen::Matrix3f &cam_calib, Eigen::Vector3f &final_robust_vp_x, Eigen::Vector3f &final_robust_vp_y)
{
	Eigen::Matrix3f inv_cam_calib = cam_calib.inverse(); 

	//trans from vps to rays through camera axis, see Z+H Chapter 8, more on single view geometry!
	unsigned int num_vps = computed_vp_hypothesis.size();
	std::vector< Eigen::Vector3f> computed_vp_hypothesis_x;
	std::vector< Eigen::Vector3f> computed_vp_hypothesis_y;
	std::vector< Eigen::Vector3f> computed_vp_hypothesis_z;
	for (unsigned int i = 0; i < num_vps; i++)
	{
		std::vector<Eigen::Vector2f> a_vp = computed_vp_hypothesis.at(i);

		Eigen::Vector2f a_x = a_vp.at(0);
		Eigen::Vector3f x_h, n_x;
		x_h(0) = a_x(0);
		x_h(1) = a_x(1);
		x_h(2) = 1;
		n_x = inv_cam_calib * x_h;
		n_x = n_x.normalized();
		computed_vp_hypothesis_x.push_back(n_x);

		Eigen::Vector2f a_y = a_vp.at(1);
		Eigen::Vector3f y_h, n_y;
		y_h(0) = a_y(0);
		y_h(1) = a_y(1);
		y_h(2) = 1;
		n_y = inv_cam_calib * y_h;
		n_y = n_y.normalized();
		computed_vp_hypothesis_y.push_back(n_y);

		Eigen::Vector2f a_z = a_vp.at(2);
		Eigen::Vector3f z_h, n_z;
		z_h(0) = a_z(0);
		z_h(1) = a_z(1);
		z_h(2) = 1;
		n_z = inv_cam_calib * z_h;
		n_z = n_z.normalized();
		computed_vp_hypothesis_z.push_back(n_z);
	}

	std::vector<Eigen::Vector3f> in_liers_x;
	std::vector<Eigen::Vector3f> in_liers_y;
	std::vector<Eigen::Vector3f> in_liers_z;
	bool found_inliers_x = getRansacInliers(computed_vp_hypothesis_x, in_liers_x);
	bool found_inliers_y = getRansacInliers(computed_vp_hypothesis_y, in_liers_y);
	bool found_inliers_z = getRansacInliers(computed_vp_hypothesis_z, in_liers_z);

	Eigen::VectorXf optimized_vp_x;
	Eigen::VectorXf optimized_vp_y;
	Eigen::VectorXf optimized_vp_z;
	leastQuaresVPFitting(in_liers_x, optimized_vp_x);
	leastQuaresVPFitting(in_liers_y, optimized_vp_y);
	leastQuaresVPFitting(in_liers_z, optimized_vp_z);
        std::cout<<"Vanishing Points Validated"<<std::endl;

	//test the angles and see if OK otherwise check again if truelly orthogonal
	Eigen::Vector3f vp_x (optimized_vp_x[3], optimized_vp_x[4], optimized_vp_x[5]);;
	Eigen::Vector3f vp_y (optimized_vp_y[3], optimized_vp_y[4], optimized_vp_y[5]);
	Eigen::Vector3f vp_z (optimized_vp_z[3], optimized_vp_z[4], optimized_vp_z[5]);

	Eigen::Vector3f vp_x_centroid (optimized_vp_x[0], optimized_vp_x[1], optimized_vp_x[2]);
	Eigen::Vector3f vp_y_centroid (optimized_vp_y[0], optimized_vp_y[1], optimized_vp_y[2]);
	Eigen::Vector3f vp_z_centroid (optimized_vp_z[0], optimized_vp_z[1], optimized_vp_z[2]);

	float angle_value_radiens_cxy = angleBetweenVectors(vp_x_centroid, vp_y_centroid);
	float angle_value_degrees_cxy = pcl::rad2deg(angle_value_radiens_cxy);
	float angle_value_radiens_cxz = angleBetweenVectors(vp_x_centroid, vp_z_centroid);
	float angle_value_degrees_cxz = pcl::rad2deg(angle_value_radiens_cxz);
	float angle_value_radiens_cyz = angleBetweenVectors(vp_y_centroid, vp_z_centroid);
	float angle_value_degrees_cyz = pcl::rad2deg(angle_value_radiens_cyz);

	float angle_value_radiens_xy = angleBetweenVectors(vp_x, vp_y);
	float angle_value_degrees_xy = pcl::rad2deg(angle_value_radiens_xy);
	float angle_value_radiens_xz = angleBetweenVectors(vp_x, vp_z);
	float angle_value_degrees_xz = pcl::rad2deg(angle_value_radiens_xz);
	float angle_value_radiens_yz = angleBetweenVectors(vp_y, vp_z);
	float angle_value_degrees_yz = pcl::rad2deg(angle_value_radiens_yz);

	//collect only the mean vps
	final_robust_vp_x = optimized_vp_x.tail<3> ();
	final_robust_vp_y = optimized_vp_y.tail<3> ();

	//final_robust_vp_x = vp_x_centroid;
	//final_robust_vp_y = vp_y_centroid;
}


bool fi::VPDetectionWrapper::getRansacInliers(const std::vector<Eigen::Vector3f> &vp_hypothesis, std::vector<Eigen::Vector3f> &in_liers, float angular_tolerance)
{
	unsigned int num_vps = vp_hypothesis.size();

	unsigned int max_count = 0;
	unsigned int best_index = -1;

	for ( unsigned int i = 0; i < num_vps; i++)
	{
		Eigen::Vector3f tmp_n = vp_hypothesis.at(i);
		unsigned int tmp_count = 0;

		for (unsigned int j = 0; j < num_vps; j++)
		{

			float angle_value_radiens = angleBetweenVectors(tmp_n, vp_hypothesis.at(j));
			float angle_value_degrees = pcl::rad2deg(angle_value_radiens);

			if (angle_value_degrees < angular_tolerance)
			{
				tmp_count++;
			}
		}

		if (tmp_count > max_count)
		{
			max_count = tmp_count;
			best_index = i;
		}
	}

	//collect inliers
	Eigen::Vector3f best_querry = vp_hypothesis.at(best_index);
	for (unsigned int j = 0; j < num_vps; j++)
	{

		float angle_value_radiens = angleBetweenVectors(best_querry, vp_hypothesis.at(j));
		float angle_value_degrees = pcl::rad2deg(angle_value_radiens);

		if (angle_value_degrees < angular_tolerance)
		{
			in_liers.push_back(vp_hypothesis.at(j));
		}
	}

	if (in_liers.size() == 0)
	{
		return false;
	}
	return true;
}


float fi::VPDetectionWrapper::angleBetweenVectors(const Eigen::Vector3f &vector1, const Eigen::Vector3f &vector2)
{
	Eigen::Vector3f vec1 = vector1;
	vec1 = vec1.normalized();

	Eigen::Vector3f vec2 = vector2;
	vec2 = vec2.normalized();

	// added this. In some cases angle was numerically unstable
	float r = vec2.dot(vec1);
	if (r > 1.0) r = 1.0;
	if (r < -1.0) r = -1.0;
	return acosf(r);
}


void fi::VPDetectionWrapper::leastQuaresVPFitting(const std::vector<Eigen::Vector3f> &vp_inliers, Eigen::VectorXf &optimized_vp)
{
	unsigned int num_in_liers = vp_inliers.size();


	//change to double
	//std::vector<Eigen::Vector3d> vp_double(num_in_liers);
	//for ( unsigned int i = 0; i < num_in_liers; i++)
	//{
	//	Eigen::Vector3f a_vp = vp_inliers[i];
	//	vp_double[i](0) = a_vp(0);
	//	vp_double[i](1) = a_vp(1);
	//	vp_double[i](2) = a_vp(2);
	//}

	Eigen::VectorXd opt_double_val;
	opt_double_val.resize(6);

	//porpulate the vps in a cloud just to make things a bit h\E4ndy
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	point_cloud->height = 1;
	point_cloud->width = num_in_liers;
	point_cloud->resize(point_cloud->height * point_cloud->width);

	for (unsigned int i = 0; i < num_in_liers; i++)
	{
		Eigen::Vector3f n_tmp = vp_inliers.at(i);
		point_cloud->points[i].x = n_tmp(0);
		point_cloud->points[i].y = n_tmp(1);
		point_cloud->points[i].z = n_tmp(2);
	}

	optimized_vp.resize (6);

	// Compute the 3x3 covariance matrix
	Eigen::Vector4d centroid;
	//pcl::compute3DCentroid (*point_cloud, centroid);

	Eigen::Matrix3d covariance_matrix;
	//pcl::computeCovarianceMatrix(*input_, inliers, centroid, covariance_matrix);
	//pcl::computeCovarianceMatrix(*point_cloud, centroid, covariance_matrix);
	pcl::computeMeanAndCovarianceMatrix(*point_cloud, covariance_matrix, centroid);
	optimized_vp[0] = centroid[0];
	optimized_vp[1] = centroid[1];
	optimized_vp[2] = centroid[2];
	opt_double_val(0) = centroid[0];
	opt_double_val(1) = centroid[1];
	opt_double_val(2) = centroid[2];

	// Extract the eigenvalues and eigenvectors
	EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
	EIGEN_ALIGN16 Eigen::Matrix3d eigen_vectors;
	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

	//optimized_vp.tail<3> () = eigen_vectors.col (0).normalized ();
	opt_double_val.tail<3> () = eigen_vectors.col (0).normalized ();
	optimized_vp[3] = static_cast<float> (opt_double_val[3]);
	optimized_vp[4] = static_cast<float> (opt_double_val[4]);
	optimized_vp[5] = static_cast<float> (opt_double_val[5]);
}


std::string fi::VPDetectionWrapper::correctSlashes(std::string &s)
{
	std::string::iterator p = s.begin();
	for (p; p!=s.end(); p++)
	{
#ifdef WIN32
		if(*p =='/') *p = '/\\';
#else
		if (*p=="\\/") *p = '/';
#endif
	}
	return s;
}