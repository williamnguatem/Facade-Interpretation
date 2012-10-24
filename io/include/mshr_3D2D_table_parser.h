#ifndef _MSHR_3D2D_TABLE_PARSER_H_
#define _MSHR_3D2D_TABLE_PARSER_H_

#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

namespace fi
{

	class MSHR3D2DTableParser 
	{
		typedef std::pair<pcl::PointXYZ, std::vector<unsigned int> > MapingType;

	public:

		MSHR3D2DTableParser();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		MSHR3D2DTableParser(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension);

		//Destructor
		virtual ~MSHR3D2DTableParser();

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the vtkpolydata into point cloud format for easy access of xyz values
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void setMSHRResultDir(const std::string &mshr_result_dir);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the point clouds parsed from the 3D2D-table datei
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void get2D3DMappings(std::vector<MapingType> &mapping_table);

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
		///** \brief std::vector<std::vector<std::string> > &corresponding_images_filenames is the input to vp detection for all images
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
		///** \brief Loads a 3D point cloud from a given PCD file, and returns: a 
		//* vtkDataSet object containing the point cloud.
		//* \param file_name the name of the file containing the XYZ dataset
		//*/
		//vtkPolyData* loadPCDasDataset(const std::string &nFilename);
		std::string correctSlashes(std::string &s);
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


	private:

		//helper function to search for 2D3D-table file

		//3D Model
		std::string m_mshr_result_dir;
		std::string m_image_data_dir;
		std::string m_image_extension;  //this can be gotten from the params file
		std::string m_unibw_params_dir;
		std::string m_dlr_params_dir;
		std::string m_mshr_params_file;
	};
}//fi

#endif//_MSHR_3D2D_TABLE_PARSER_H_
