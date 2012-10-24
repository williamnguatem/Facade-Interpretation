#ifndef _CAM_CALIB_PARSER_H_
#define _CAM_CALIB_PARSER_H_

#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

namespace fi
{

	class CamCalibParser 
	{
	
	public:

		CamCalibParser();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		CamCalibParser(const std::string &mshr_result_dir);

		//Destructor
		virtual ~CamCalibParser();

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the vtkpolydata into point cloud format for easy access of xyz values
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void setCameraCalib(const Eigen::Matrix3f &calib_cam);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the point clouds parsed from the 3D2D-table datei
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool getCameraCalib(Eigen::Matrix3f &calib_cam);

		
		bool getCameraCalibrationFile(
			const boost::filesystem::path & dir_path,         // in this directory,
		//	const std::string & file_name, // search for this name,
			boost::filesystem::path & path_found );		 // camera calibration file


	private:

		//3D Model
		std::string m_mshr_result_dir;
		Eigen::Matrix3f m_calib_camera;
	};
}//fi

#endif//_CAM_CALIB_PARSER_H_
