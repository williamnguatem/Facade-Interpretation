#ifndef _CAM_POSE_PARSER_H_
#define _CAM_POSE_PARSER_H_

#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

namespace fi
{

	class CamPOSEParser 
	{
	
	public:

		CamPOSEParser();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		CamPOSEParser(const std::string &mshr_pose_dir, const std::vector<std::vector<unsigned int> > image_names);

		CamPOSEParser(const std::string &mshr_pose_dir, const std::vector<std::string> image_names);

		//Destructor
		virtual ~CamPOSEParser();

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the vtkpolydata into point cloud format for easy access of xyz values
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool getCamPose(std::vector<std::vector<Eigen::Matrix3f> > &rotation_matrix, 
			std::vector<std::vector<Eigen::Vector3f> > &translation_vector, 
			std::vector<std::vector<float> > &radial_distortion_params_k1, 
			std::vector<std::vector<float> > &radial_distortion_param_k2,
			std::vector<std::vector<unsigned int> > &image_width,
			std::vector<std::vector<unsigned int> > &image_height);

		//ToDo: Move this to private since its only needed within this class!
		void getCamRotationMatrix(const boost::filesystem::path &pose_file, Eigen::Matrix3f &rotation_matrix);
		void getCamTranslationVector(const boost::filesystem::path &pose_file, Eigen::Vector3f &translation_vector);
		void getRadialDistortionParams(const boost::filesystem::path &pose_file, float &radial_distortion_params_k1, float &radial_distortion_param_k2);
		unsigned int getImageWidth(const boost::filesystem::path &pose_file);
		unsigned int getImageHeight(const boost::filesystem::path &pose_file);




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
		std::string m_mshr_pose_dir;
		std::vector<std::vector<unsigned int> > m_image_names;
		//std::vector<std::string> m_image_names; //at the moment mshr needs but numbers to save the images maybe the'll change things to strings
	};
}//fi

#endif//_CAM_POSE_PARSER_H_
