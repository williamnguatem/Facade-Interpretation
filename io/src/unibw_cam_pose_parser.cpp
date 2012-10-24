#include "unibw_cam_pose_parser.h"
#include <iostream> 
#include <boost/regex.hpp> 
#include <locale> 
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp> 



fi::CamPOSEParser::CamPOSEParser()
{

}

fi::CamPOSEParser::CamPOSEParser(const std::string &mshr_pose_dir, const std::vector<std::vector<unsigned int> > image_names)
	:m_mshr_pose_dir(mshr_pose_dir), m_image_names(image_names)
{

}


fi::CamPOSEParser::~CamPOSEParser()
{

}


bool fi::CamPOSEParser::getCamPose(std::vector<std::vector<Eigen::Matrix3f> > &rotation_matrix, 
	std::vector<std::vector<Eigen::Vector3f> > &translation_vector, 
	std::vector<std::vector<float> > &radial_distortion_params_k1, 
	std::vector<std::vector<float> > &radial_distortion_param_k2,
	std::vector<std::vector<unsigned int> > &image_width,
	std::vector<std::vector<unsigned int> > &image_height)
{
	const boost::filesystem::path dir_path( m_mshr_pose_dir);

	if ( !boost::filesystem::exists( dir_path ) ) return false;
	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	unsigned int num_images = m_image_names.size();

	if (num_images == 0)
	{
		PCL_ERROR(" No input image found in directory %s ", m_mshr_pose_dir );
		return false;
	}


	//search for all camera poses
	std::vector<std::string> pose_param_files(num_images); //preallocate for speed
	for (unsigned int i = 0; i < num_images; i++)
	{
		std::vector<unsigned int> imgs_for_point_i = m_image_names[i];
		unsigned int i_size = imgs_for_point_i.size();
		std::vector<Eigen::Matrix3f> tmp_rotation_matrix(i_size );
		std::vector<Eigen::Vector3f> tmp_translation_vector(i_size );
		std::vector<float> tmp_radial_distortion_params_k1(i_size );
		std::vector<float> tmp_radial_distortion_param_k2(i_size );
		std::vector<unsigned int> tmp_image_width;
		std::vector<unsigned int> tmp_image_height;

		//extract the pose for this point and store accordingly
		for (unsigned int k = 0; k < i_size; k ++ )
		{
			std::string an_image_name = boost::lexical_cast<std::string>(imgs_for_point_i[k]);
			boost::filesystem::path path_found; //path to the mapping table file
			bool file_found = false;

			boost::regex mapping_filter(an_image_name);//used to matched the image pose_params in the unibw/dlr folder
			for ( boost::filesystem::directory_iterator itr( dir_path );itr != end_itr; ++itr )
			{
				boost::smatch what_to_match;
				//std::cout<<"file name: "<<itr->path().filename().string()<<std::endl;

				//if ( boost::filesystem::is_directory(itr->status()) )
				//{
				//	if ( get3D2DTableFile( itr->path(), /*file_name,*/ path_found ) ) return true;
				//}else 
				if ( !boost::filesystem::is_regular_file( itr->status() ) )
					continue; // Skip if not a file 
				if ( !boost::regex_search(itr->path().filename().string(), what_to_match, mapping_filter ) ) 
					continue; // Skip if no match
				path_found = itr->path();
				file_found = true;
				break;
			}
			if (file_found)
			{
				Eigen::Matrix3f an_r_matrix;
				Eigen::Vector3f a_t_vector;
				float k1_val, k2_val;
				getCamRotationMatrix(path_found, an_r_matrix);
				getCamTranslationVector(path_found, a_t_vector);
				getRadialDistortionParams(path_found, k1_val, k2_val);
				unsigned int img_h = getImageHeight(path_found);
				unsigned int img_w = getImageWidth(path_found);
				tmp_image_height.push_back(img_h);
				tmp_image_width.push_back(img_w);
				tmp_rotation_matrix[k] = an_r_matrix;
				tmp_translation_vector[k] = a_t_vector;
				tmp_radial_distortion_params_k1[k] = k1_val;
				tmp_radial_distortion_param_k2[k] = k2_val;
			}
		}
		rotation_matrix.push_back(tmp_rotation_matrix);
		translation_vector.push_back(tmp_translation_vector);
		radial_distortion_params_k1.push_back(tmp_radial_distortion_params_k1);
		radial_distortion_param_k2.push_back(tmp_radial_distortion_param_k2);
		image_width.push_back(tmp_image_width);
		image_height.push_back(tmp_image_height);

		if (i == 10)
		{
			break;
		}
	}

	if (radial_distortion_param_k2.size() == 0||radial_distortion_params_k1.size() == 0)
	{
		std::cout<< "Something went wrong with the computation of [R|T]"<<std::endl;
		return false;
	}
	return true;
}


//ToDo: Move this to private since its only needed within this class!
void fi::CamPOSEParser::getCamRotationMatrix(const boost::filesystem::path &pose_file, Eigen::Matrix3f &rotation_matrix)
{
	std::string a_line;
	unsigned int num_lines = 0; // the rotation matrix is on the 3rd line!
	//std::ifstream param_file ("E:\\21\\Result\\UniBW\\10884_param.txt");
	std::ifstream param_file (pose_file.string().c_str());
	if (param_file.is_open())
	{
		while ( param_file.good() )
		{
			getline (param_file, a_line);	
			std::vector<unsigned int> images_names;

			if (num_lines == 2)
			{
				std::vector< std::string> r_vals0, r_vals1, r_vals2, rows_of_r, tmp_remove_begin, tmp_remove_end;
				boost::split(tmp_remove_begin, a_line, boost::is_any_of("["));
				boost::split(tmp_remove_end, tmp_remove_begin[1], boost::is_any_of("]"));
				boost::split(rows_of_r, tmp_remove_end[0], boost::is_any_of(";"));

				//first row
				boost::split(r_vals0, rows_of_r[0], boost::is_any_of(" "));

				//second row //terminating null in the first position
				boost::split(r_vals1, rows_of_r[1], boost::is_any_of(" "));
				//third row //terminating null in the first position
				boost::split(r_vals2, rows_of_r[2], boost::is_any_of(" "));

				rotation_matrix<< boost::lexical_cast<float>(r_vals0[0]), boost::lexical_cast<float>(r_vals0[1]), boost::lexical_cast<float>(r_vals0[2]),
					boost::lexical_cast<float>(r_vals1[1]), boost::lexical_cast<float>(r_vals1[2]), boost::lexical_cast<float>(r_vals1[3]),
					boost::lexical_cast<float>(r_vals2[1]), boost::lexical_cast<float>(r_vals2[2]), boost::lexical_cast<float>(r_vals2[3]);
				break;
			}
			num_lines++;
		}
	}
	//std::cout<<rotation_matrix<<std::endl;
}


void fi::CamPOSEParser::getCamTranslationVector(const boost::filesystem::path &pose_file, Eigen::Vector3f &translation_vector)
{

	std::string a_line;
	unsigned int num_lines = 0; // the rotation matrix is on the 4th line!
	//std::ifstream param_file ("E:\\21\\Result\\UniBW\\10884_param.txt");
	std::ifstream param_file (pose_file.string().c_str());
	if (param_file.is_open())
	{
		while ( param_file.good() )
		{
			getline (param_file, a_line);	
			std::vector<unsigned int> images_names;

			if (num_lines == 3)
			{
				std::vector< std::string> r_vals, tmp_remove_begin, tmp_remove_end;
				boost::split(tmp_remove_begin, a_line, boost::is_any_of("["));
				boost::split(tmp_remove_end, tmp_remove_begin[1], boost::is_any_of("]"));
				boost::split(r_vals, tmp_remove_end[0], boost::is_any_of(" "));

				translation_vector(0) = boost::lexical_cast<float>(r_vals[0]);
				translation_vector(1) = boost::lexical_cast<float>(r_vals[1]);
				translation_vector(2) = boost::lexical_cast<float>(r_vals[2]);
				break;
			}
			num_lines++;
		}
	}
	//std::cout<<translation_vector<<std::endl;
}


void fi::CamPOSEParser::getRadialDistortionParams(const boost::filesystem::path &pose_file, float &radial_distortion_param_k1, float &radial_distortion_param_k2)
{
	std::string a_line;
	unsigned int num_lines = 0; // the radial distortion params is on the 6 and 7th line!
	//std::ifstream param_file ("E:\\21\\Result\\UniBW\\10884_param.txt");
	std::ifstream param_file (pose_file.string().c_str());
	if (param_file.is_open())
	{
		while ( param_file.good() )
		{
			getline (param_file, a_line);	
			std::vector<unsigned int> images_names;

			if (num_lines == 5)
			{
				std::vector< std::string> r_vals, tmp_remove_begin, tmp_remove_end;
				boost::split(tmp_remove_begin, a_line, boost::is_any_of("="));
				/*	boost::split(tmp_remove_end, tmp_remove_begin.at(1), boost::is_any_of("]"));
				boost::split(r_vals, tmp_remove_end.at(0), boost::is_any_of(" "));*/

				radial_distortion_param_k1 = boost::lexical_cast<float>(tmp_remove_begin[1]);
			}

			if (num_lines == 6)
			{
				std::vector< std::string> r_vals, tmp_remove_begin, tmp_remove_end;
				boost::split(tmp_remove_begin, a_line, boost::is_any_of("="));
				/*	boost::split(tmp_remove_end, tmp_remove_begin.at(1), boost::is_any_of("]"));
				boost::split(r_vals, tmp_remove_end.at(0), boost::is_any_of(" "));*/

				radial_distortion_param_k2 = boost::lexical_cast<float>(tmp_remove_begin[1]);
				break;
			}
			num_lines++;
		}
	}
	//std::cout<<radial_distortion_param_k1<<std::endl;
	//std::cout<<radial_distortion_param_k2<<std::endl;
}


unsigned int fi::CamPOSEParser::getImageWidth(const boost::filesystem::path &pose_file)
{
	std::string a_line;
	unsigned int num_lines = 0; // the radial distortion params is on the 6 and 7th line!
	//std::ifstream param_file ("E:\\21\\Result\\UniBW\\10884_param.txt");
	std::ifstream param_file (pose_file.string().c_str());
	if (param_file.is_open())
	{
		while ( param_file.good() )
		{
			getline (param_file, a_line);	
			std::vector<unsigned int> images_names;

			if (num_lines == 8)
			{
				std::vector< std::string> r_vals, tmp_remove_begin, tmp_remove_end;
				boost::split(tmp_remove_begin, a_line, boost::is_any_of("="));
				return boost::lexical_cast<unsigned int>(tmp_remove_begin[1]);
			}
			num_lines++;
		}
	}
	else
	{
		return (0);
	}
}


unsigned int fi::CamPOSEParser::getImageHeight(const boost::filesystem::path &pose_file)
{
	std::string a_line;
	unsigned int num_lines = 0; // the radial distortion params is on the 6 and 7th line!
	//std::ifstream param_file ("E:\\21\\Result\\UniBW\\10884_param.txt");
	std::ifstream param_file (pose_file.string().c_str());
	if (param_file.is_open())
	{
		while ( param_file.good() )
		{
			getline (param_file, a_line);	
			std::vector<unsigned int> images_names;

			if (num_lines == 4)
			{
				std::vector< std::string> r_vals, tmp_remove_begin, tmp_remove_end;
				boost::split(tmp_remove_begin, a_line, boost::is_any_of("="));
				return boost::lexical_cast<unsigned int>(tmp_remove_begin[1]);
			}
			num_lines++;
		}
	}
	else
	{
		return (0);
	}
}

//
////do the actual parsing of the file and save to an Eigen Matrix
//bool fi::CamCalibParser::getCameraCalib(Eigen::Matrix3f &calib_cam)
//{
//	const boost::filesystem::path dir_path( m_mshr_result_dir);
//	boost::filesystem::path path_found; 
//	bool found_calib_file = getCameraCalibrationFile( dir_path, path_found );
//
//	if (found_calib_file)
//	{
//		std::string a_line;
//		std::ifstream amy_file (path_found.string().c_str());
//		if (amy_file.is_open())
//		{
//			unsigned int k_row_lenght = 0;
//			Eigen::Matrix3f cam_k;
//			cam_k;
//			Eigen::VectorXf tmp_val;
//			tmp_val.resize(9);
//			while ( amy_file.good() )
//			{
//				getline (amy_file, a_line);	
//				std::vector<std::string> strs;
//
//				if (k_row_lenght < 3 )
//				{
//					boost::split(strs, a_line, boost::is_any_of("\t "));
//
//					tmp_val(0+k_row_lenght*3) = atof(strs.at(0).c_str());
//					tmp_val[1+k_row_lenght*3] = atof(strs.at(1).c_str());
//					tmp_val[2+k_row_lenght*3] = atof(strs.at(2).c_str());
//					std::cout<< " k is: "<<tmp_val<<std::endl;
//				}
//				else{
//					cam_k <<tmp_val[0], tmp_val[1], tmp_val[2],
//						tmp_val[3], tmp_val[4], tmp_val[5],
//						tmp_val[6], tmp_val[7], tmp_val[8];
//					std::cout<< " calibration file: "<<"\n"<<cam_k<<std::endl;
//					calib_cam = cam_k;
//					break;
//				}
//
//				k_row_lenght++;
//			}
//			amy_file.close();
//			return true;
//		}
//		else
//		{
//			PCL_ERROR("Unable to open the file! ");
//			return false;
//		}
//	}
//	else
//	{
//		return false;
//		PCL_ERROR("Calibration file not found! ");
//	}
//}
