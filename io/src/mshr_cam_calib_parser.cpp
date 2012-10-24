#include "mshr_cam_calib_parser.h"
#include <locale> 
#include <iostream> 
#include <boost/regex.hpp> 
#include <boost/regex.hpp> 
#include <locale> 
#include <iostream> 



fi::CamCalibParser::CamCalibParser()
{

}


fi::CamCalibParser::CamCalibParser(const std::string &mshr_result_dir)
	:m_mshr_result_dir(mshr_result_dir)
{

}

fi::CamCalibParser::~CamCalibParser()
{

}


bool fi::CamCalibParser::getCameraCalibrationFile(const boost::filesystem::path & dir_path, boost::filesystem::path & path_found )	
{
	if ( !boost::filesystem::exists( dir_path ) ) return false;
	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	boost::regex mapping_filter(".cal");//used to matched the .cal or .calrd	
	for ( boost::filesystem::directory_iterator itr( dir_path );itr != end_itr; ++itr )
	{
		boost::smatch what_to_match;
		std::cout<<"file name: "<<itr->path().filename().extension().string()<<std::endl;

		//if ( boost::filesystem::is_directory(itr->status()) )
		//{
		//	if ( get3D2DTableFile( itr->path(), /*file_name,*/ path_found ) ) return true;
		//}else 
		if ( !boost::filesystem::is_regular_file( itr->status() ) )
			continue; // Skip if not a file 
		//if ( !boost::regex_search(itr->path().filename().extension().string(), what_to_match, mapping_filter ) ) 
		if (itr->path().filename().extension().string() == ".cal")
		{
			path_found = itr->path();
		}else
			continue; // Skip if no match

			return true;
	}
	return false;
}

//do the actual parsing of the file and save to an Eigen Matrix
bool fi::CamCalibParser::getCameraCalib(Eigen::Matrix3f &calib_cam)
{
	const boost::filesystem::path dir_path( m_mshr_result_dir);
	boost::filesystem::path path_found; 
	bool found_calib_file = getCameraCalibrationFile( dir_path, path_found );

	if (found_calib_file)
	{
		std::string a_line;
		std::ifstream amy_file (path_found.string().c_str());
		if (amy_file.is_open())
		{
			unsigned int k_row_lenght = 0;
			Eigen::Matrix3f cam_k;
			cam_k;
			Eigen::VectorXf tmp_val;
			tmp_val.resize(9);
			while ( amy_file.good() )
			{
				getline (amy_file, a_line);	
				std::vector<std::string> strs;

				if (k_row_lenght < 3 )
				{
					boost::split(strs, a_line, boost::is_any_of("\t "));

					tmp_val(0+k_row_lenght*3) = atof(strs.at(0).c_str());
					tmp_val[1+k_row_lenght*3] = atof(strs.at(1).c_str());
					tmp_val[2+k_row_lenght*3] = atof(strs.at(2).c_str());
					std::cout<< " k is: "<<tmp_val<<std::endl;
				}
				else{
					cam_k <<tmp_val[0], tmp_val[1], tmp_val[2],
						tmp_val[3], tmp_val[4], tmp_val[5],
						tmp_val[6], tmp_val[7], tmp_val[8];
					std::cout<< " calibration file: "<<"\n"<<cam_k<<std::endl;
					calib_cam = cam_k;
					break;
				}

				k_row_lenght++;
			}
			amy_file.close();
			return true;
		}
		else
		{
			PCL_ERROR("Unable to open the file! ");
			return false;
		}
	}
	else
	{
		return false;
		PCL_ERROR("Calibration file not found! ");
	}
}