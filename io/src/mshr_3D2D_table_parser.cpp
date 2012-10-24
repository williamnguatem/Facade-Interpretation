#include "mshr_3D2D_table_parser.h"
#include <locale> 
#include <iostream> 
#include <boost/regex.hpp> 
#include <boost/lexical_cast.hpp> 
#include <locale> 
#include <iostream> 
 
fi::MSHR3D2DTableParser::MSHR3D2DTableParser()
{

}


fi::MSHR3D2DTableParser::MSHR3D2DTableParser(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension) 
	:m_mshr_result_dir(mshr_result_dir), m_image_data_dir(image_data_dir), m_image_extension(image_extension)
{

}

fi::MSHR3D2DTableParser::~MSHR3D2DTableParser()
{

}

bool fi::MSHR3D2DTableParser::parseInput(pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, std::vector<std::vector <unsigned int> > &corresponding_images, std::vector<std::vector<std::string> > &mapping_table)
{

	const boost::filesystem::path dir_path(m_mshr_result_dir);
	boost::filesystem::path path_found; //path to the mapping table file 
	bool file_found = get3D2DTableFile(	dir_path, path_found); 
	//assert(file_found, true);

	if (!file_found)
	{
		return false;
	}

	//std::string sPath(cloud_file_data);
	//InputParser sData;
	//char sVertex[2];
	float x_temp = 0, y_temp = 0, z_temp = 0;
	int s_num_temp = 0;

	unsigned int num_lines = 0;
	std::string line;
	std::ifstream mapping_file (path_found.string().c_str());

	if (mapping_file.is_open())
	{
		while ( mapping_file.good() )
		{
			getline (mapping_file, line);
			//std::cout << line << std::endl;
			num_lines++;
		}
		mapping_file.close();
	}
	else
	{
		std::cout << "Unable to open file"<<std::endl; 
	}

	////creat cloud with known size
	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	in_cloud->height = 1;
	in_cloud->width = num_lines - 2; //the first line and the last line are removed. see file 3D2Dtable format
	in_cloud->resize(in_cloud->height * in_cloud->width);

	num_lines = -1;
	std::string a_line;
	std::ifstream amyfile (path_found.string().c_str());
	if (amyfile.is_open())
	{
		while ( amyfile.good() )
		{
			getline (amyfile, line);	
			std::vector<std::string> strs;
			std::vector<unsigned int> images_names;

			if (num_lines !=-1 && num_lines < in_cloud->points.size())
			{
				boost::split(strs, line, boost::is_any_of("\t "));
				x_temp = in_cloud->points[num_lines].x = atof(strs.at(0).c_str());
				y_temp = in_cloud->points[num_lines].y = atof(strs.at(1).c_str());
				z_temp = in_cloud->points[num_lines].z = atof(strs.at(2).c_str());

				unsigned int found_point_in = strs.size() - 4; //strs[0], strs[1] and strs[2] are x, y, z, 4 and not 3 because of the null terminated string!
				for (unsigned int j = 0; j < found_point_in; j++)
				{
					images_names.push_back(atoi(strs.at(j + 3).c_str()));
				}

				//collect the parsed data and create a full assignment table
				mapping_table.push_back(strs);
				corresponding_images.push_back(images_names);
			}

			num_lines++;
		}
		amyfile.close();
		point_cloud = in_cloud;
		return true;
	}
	else
	{
		PCL_ERROR("Unable to open the file! ");
		return false;
	}
}

bool fi::MSHR3D2DTableParser::get3D2DTableFile(	const boost::filesystem::path & dir_path, boost::filesystem::path & path_found )	
{
	if ( !boost::filesystem::exists( dir_path ) ) return false;
	boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end

	boost::regex mapping_filter("3D2Dtable");//used to matched the 3D2Dtable	
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
		return true;
	}
	return false;
}


bool fi::MSHR3D2DTableParser::getCorrespondingImageFiles(const std::vector<std::vector <unsigned int> > &corresponding_images, std::vector<std::vector<std::string> > &corresponding_images_filenames )
{
	unsigned int num_points = corresponding_images.size();

	for (unsigned int i = 0; i < num_points; i++)
	{
		std::vector <unsigned int> h_images = corresponding_images.at(i);
		unsigned int h = h_images.size();
		
		std::vector<std::string> pnt_on_images(h);
		for ( unsigned int j = 0; j < h; j++)
		{
boost::filesystem::path path_separator = boost::filesystem::path("/").native();

			std::string tmp_filename = m_image_data_dir + path_separator.string()+ boost::lexical_cast<std::string>(h_images.at(j)) + m_image_extension; 
			pnt_on_images.at(j) = correctSlashes(tmp_filename);

		}
		corresponding_images_filenames.push_back(pnt_on_images);
	}

	if (corresponding_images_filenames.size() == 0)
	{
		PCL_ERROR( "No input image present within the image folder");
		return false;
	}
	else
		return true;
}


std::string fi::MSHR3D2DTableParser::correctSlashes(std::string &s)
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