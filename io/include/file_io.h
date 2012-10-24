#ifndef _MSHR_FILE_IO_H_
#define _MSHR_FILE_IO_H_

#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

namespace fi
{

	class MSHRFileIO 
	{
		typedef std::pair<pcl::PointXYZ, std::vector<unsigned int> > MapingType;

	public:

		MSHRFileIO();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unbw_params_dir);

		MSHRFileIO(const std::string &input_image_folder, const std::string &image_extension);
		/*MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &dlr_params_dir);

		
		MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unbw_params_dir, const std::string &mshr_params_file);

		MSHRFileIO(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &dlr_params_dir, const std::string &mshr_params_file);
	*/

		//Destructor
		virtual ~MSHRFileIO();

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the vtkpolydata into point cloud format for easy access of xyz values
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		//void setMSHRResultDir(const std::string &mshr_result_dir);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the point clouds parsed from the 3D2D-table datei
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void get3DTo2DProjectionsUniBW();

		void get3DTo2DProjectionsDLR(std::vector <std::vector<Eigen::Vector2f> > &pnt_projectionstion);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the point clouds parsed from the 3D2D-table datei
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void get3DPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief get cloud from ply data
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void get3DPointsFromPLY(const std::string &input_ply_file, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief get cloud from obj data
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void get3DPointsFromOBJ(const std::string &input_OBJ_file, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief get the list of images processed by the input folder
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void getIMGs(std::vector<std::string> &input_images);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief get the list of images processed by the input folder
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool getImageFileNames(std::vector<std::string> &images_filenames);


		//
		//void GetDataPcl(vtkPolyData* sInPolyData, PtCloudData &sOutpclData);



		//////////////////////////////////////////////////////////////////////////////////
		///** \brief Loads a 3D point cloud from a given *.xyz fileName also known as a delimited text , and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the *.xyz dataset
		//*/
		//vtkPolyData* loadDelimTextasDataset (const std::string &nFilename);



		//////////////////////////////////////////////////////////////////////////////////
		///** \brief Loads a 3D point cloud from a given STL fileName, and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the STL dataset
		//*/
		//vtkPolyData* loadSTLasDataset (const std::string &nFilename);



		//////////////////////////////////////////////////////////////////////////////////
		///** \brief Loads a 3D point cloud from a given VRML fileName, and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the VRML dataset
		//*/
		//vtkPolyData* loadVRMLasDataset(const std::string &nFilename);



		//////////////////////////////////////////////////////////////////////////////////
		///** \brief Loads a 3D point cloud from a given OBJ fileName, and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the OBJ dataset
		//*/
		//vtkPolyData* loadOBJasDataset (const std::string &nFilename);


		//////////////////////////////////////////////////////////////////////////////////
		///** \brief 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the XYZ dataset
		//* \return give actually the number of distinct image files present!
		//*/
		//vtkPolyData* loadXYZasDataset (const std::string &nFilename);
		bool getCorrespondingImageFiles(const std::vector<std::vector <unsigned int> > &corresponding_images, std::vector<std::vector<std::string> > &corresponding_images_filenames);


		//////////////////////////////////////////////////////////////////////////////////
		///** \brief Loads a 3D point cloud from a given PCD file, and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the XYZ dataset
		//*/
		//vtkPolyData* loadPCDasDataset(const std::string &nFilename);
		bool parseInput(pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, std::vector<std::vector <unsigned int> > &corresponding_images, std::vector<std::vector<std::string> > &mapping_table);


		//////////////////////////////////////////////////////////////////////////////////
		///** \brief Loads a 3D point cloud from a given PLY fileName, and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the PLY dataset
		//*/
		//vtkPolyData* loadPLYasDataset (const std::string &nFilename);


		//void ComputeCentroid(const PtCloudData &inCloud, Vec3d &inCloudCentoid );


		//bool CSaveToFile(const CordPoints &sPointCandidates, const char *pFilePath);

		bool get3D2DTableFile(
			const boost::filesystem::path & dir_path,         // in this directory,
		//	const std::string & file_name, // search for this name,
			boost::filesystem::path & path_found );		 // placing path here if found);


		//////////////////////////////////////////////////////////////////////////////////
		///** \brief compute the 2D point of the reconstructed 3D point using the camera pose
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the XYZ dataset
		//*/
		//vtkPolyData* loadPCDasDataset(const std::string &nFilename);
		void getProjectionPointOnImage(const pcl::PointXYZ &reconstructed_3D_point,
			const Eigen::Matrix3f &cam_calib,
			const Eigen::Matrix3f &rotations_matrix, 
			const Eigen::Vector3f &translation_vector,
			float radial_dist1, 
			float radial_dist2,
			unsigned int img_width,
			unsigned int img_height,
			Eigen::Vector2f &image_point);

		//getters
		pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud ();

		std::vector <std::vector<Eigen::Vector2f> >  getImagePoints ();

		std::vector<std::vector<std::string> > getCorrespondingImageFileNames ();

		std::vector<std::vector<std::string> > getMappingTable ();

	private:

		//helper function to search for 2D3D-table file

		//3D Model
		std::string m_mshr_result_dir;
		std::string m_image_data_dir;
		std::string m_image_extension;  //this can be gotten from the params file
		std::string m_unibw_params_dir;
		std::string m_dlr_params_dir;
		std::string m_mshr_params_file;

		//give this out through interface such that one would have access to the data
		pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
		std::vector <std::vector<Eigen::Vector2f> > m_image_points2D;
		std::vector<std::vector<std::string> > m_corresponding_images_filenames;
		std::vector<std::vector<std::string> > m_mapping_table;
	};
}//fi

#endif//_MSHR_FILE_IO_H_
