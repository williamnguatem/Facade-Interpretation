#include <iostream>
#include <cmath>
#include "mshr_3D2D_table_parser.h"
#include "mshr_cam_calib_parser.h"
#include "unibw_cam_pose_parser.h"
#include "file_io.h"



fi::MSHRFileIO::MSHRFileIO()
{

}

fi::MSHRFileIO::MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unibw_params_dir)
	:m_mshr_result_dir(mshr_result_dir), m_image_data_dir(image_data_dir), m_image_extension(image_extension), m_unibw_params_dir(unibw_params_dir)
{

}


fi::MSHRFileIO::MSHRFileIO(const std::string &input_image_folder, const std::string &image_extension)
	:m_image_data_dir(input_image_folder), m_image_extension(image_extension)
{

}


//
//fi::MSHRFileIO::MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &dlr_params_dir)
//	:m_mshr_result_dir(mshr_result_dir), m_image_data_dir(image_data_dir), m_image_extension(image_extension), m_dlr_params_dir(dlr_params_dir)
//{
//
//}
//
//
//fi::MSHRFileIO::MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unibw_params_dir, const std::string &mshr_params_file)
//	:m_mshr_result_dir(mshr_result_dir), m_image_data_dir(image_data_dir), m_image_extension(image_extension), m_unibw_params_dir(unibw_params_dir), m_mshr_params_file(m_mshr_params_file)
//{
//
//}
//
//
//fi::MSHRFileIO::MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &dlr_params_dir, const std::string &mshr_params_file)
//	:m_mshr_result_dir(mshr_result_dir), m_image_data_dir(image_data_dir), m_image_extension(image_extension), m_dlr_params_dir(dlr_params_dir), m_mshr_params_file(m_mshr_params_file)
//{
//
//}


fi::MSHRFileIO::~MSHRFileIO()
{

}


void fi::MSHRFileIO::get3DTo2DProjectionsUniBW()
{
	//collect the points and images
	fi::MSHR3D2DTableParser mshr_results(m_mshr_result_dir, m_image_data_dir, m_image_extension);
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	std::vector<std::vector<unsigned int> > corresponding_images; //image names used to extract the corresponding poses
	std::vector<std::vector<std::string> > mapping_table; 
	bool parse_data = mshr_results.parseInput(point_cloud, corresponding_images, mapping_table);
	
	//get the full paths to the images
	std::vector<std::vector<std::string> > corresponding_images_filenames;
	bool image_present = mshr_results.getCorrespondingImageFiles(corresponding_images, corresponding_images_filenames);

	//get the calibration data file
	fi::CamCalibParser m_cam(m_mshr_result_dir);
	Eigen::Matrix3f cam_calib;
	m_cam.getCameraCalib(cam_calib);
	std::cout<<cam_calib<<std::endl;

	//get the pose
	std::vector<std::vector<Eigen::Matrix3f> > rotation_matrix;
	std::vector<std::vector<Eigen::Vector3f> > translation_vectors;
	std::vector<std::vector<float> > radial_distortion_params_k1;
	std::vector<std::vector<float> > radial_distortion_params_k2;
	std::vector<std::vector<unsigned int> > image_width;
	std::vector<std::vector<unsigned int> > image_height;
	std::string unibw_params_folder = m_unibw_params_dir;
	
	fi::CamPOSEParser data_pose(m_unibw_params_dir, corresponding_images);
	bool extracted = data_pose.getCamPose(rotation_matrix, translation_vectors, radial_distortion_params_k1, radial_distortion_params_k2, image_width, image_height);
   
	//project to image
	unsigned int num_pnts = translation_vectors.size();
	for ( unsigned int i = 0; i < num_pnts; i++)
	{
		//collect for the point all the values of the pose
		std::vector<Eigen::Matrix3f> pnt_rotation_matrix = rotation_matrix.at(i);
		std::vector<Eigen::Vector3f> pnt_translation_vectors = translation_vectors.at(i);
		std::vector<float> pnt_radial_distortion_params_k1 = radial_distortion_params_k1.at(i);
		std::vector<float> pnt_radial_distortion_params_k2 = radial_distortion_params_k2.at(i);
		std::vector<unsigned int> pnt_image_width = image_width.at(i);
		std::vector<unsigned int> pnt_image_height = image_height.at(i);
		
		//
		unsigned int pnt_num_images = pnt_radial_distortion_params_k1.size();
		std::vector<Eigen::Vector2f> pnt_img_points;

		for(unsigned int j = 0; j < pnt_num_images; j ++)
		{
			//for the selected image compute all the corresponding values
				const Eigen::Matrix3f &rot_mat = pnt_rotation_matrix.at(j); 
				const Eigen::Vector3f &trans_vec = pnt_translation_vectors.at(j);
				float undist1 = pnt_radial_distortion_params_k1.at(j);
				float undist2 = pnt_radial_distortion_params_k2.at(j);
				unsigned int img_w = pnt_image_width.at(j);
				unsigned int img_h = pnt_image_height.at(j);
				Eigen::Vector2f an_image_point;

				getProjectionPointOnImage(point_cloud->points[i], cam_calib, rot_mat, trans_vec, undist1, undist2, img_w, img_h, an_image_point);
				pnt_img_points.push_back(an_image_point);
		}

		//collect all corresponding points
		m_image_points2D.push_back(pnt_img_points);
	}

	m_cloud = point_cloud;
	m_corresponding_images_filenames = corresponding_images_filenames;
	m_mapping_table = mapping_table;
}


void fi::MSHRFileIO::getProjectionPointOnImage(const pcl::PointXYZ &reconstructed_3D_point,
	const Eigen::Matrix3f &cam_calib,
	const Eigen::Matrix3f &rotations_matrix, 
	const Eigen::Vector3f &translation_vector,
	float radial_dist1, 
	float radial_dist2,
	unsigned int img_width,
	unsigned int img_height,
	Eigen::Vector2f &image_point)
{

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

	Eigen::Vector3f point_homgenoues_img = cam_calib*point_xy1;

	//undistortion
	float r = sqrtf(point_xy1(0)*point_xy1(0) + point_xy1(1)*point_xy1(1)); 

	float xa = point_xy1(0)*(1 + radial_dist1*(r*r) + radial_dist2*(r*r*r*r));
	float ya = point_xy1(1)*(1 + radial_dist1*(r*r) + radial_dist2*(r*r*r*r));

	//rescale
	float img_scale = (img_width + img_height)/2;
	float step_x = img_width/2;
	float step_y = img_height/2;

	float pixel_x = xa * img_scale + step_x;
	float pixel_y = ya * img_scale + step_y;

	image_point(0) = pixel_x;
	image_point(1) = pixel_y;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr fi::MSHRFileIO::getCloud()
{
	return m_cloud;
}


std::vector <std::vector<Eigen::Vector2f> >  fi::MSHRFileIO::getImagePoints()
{
	return m_image_points2D;
}


std::vector<std::vector<std::string> > fi::MSHRFileIO::getCorrespondingImageFileNames()
{
	return m_corresponding_images_filenames;
}


std::vector<std::vector<std::string> > fi::MSHRFileIO::getMappingTable ()
{
	return m_mapping_table;
}


bool fi::MSHRFileIO::getImageFileNames(std::vector<std::string> &images_filenames)
{
	if ( !boost::filesystem::exists( m_image_data_dir ) )
	{
		std::cout<<"directory not found!"<<std::endl;
		return false;
	}

	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	//boost::regex mapping_filter("3D2Dtable");//used to matched the 3D2Dtable	

	for ( boost::filesystem::directory_iterator itr( m_image_data_dir );itr != end_itr; ++itr )
	{
		if ( !boost::filesystem::is_regular_file( itr->status() ) )
			continue; // Skip if not a file 
		if (! (itr->path().extension().string() == m_image_extension)) 
			continue; // Skip if no match
		images_filenames.push_back(itr->path().string());
	}

	if (images_filenames.size() < 2)
	{
		std::cout<<"At least not 2 images are needed for matching !"<<std::endl;
		return false;
	}

	return true;
}