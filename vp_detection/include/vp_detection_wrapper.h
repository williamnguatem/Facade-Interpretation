#ifndef _VP_DETECTION_WRAPPER_H_
#define _VP_DETECTION_WRAPPER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>

namespace fi
{
	//just a helper function
	template<typename T>
	void removeDuplicates(std::vector<T>& vec)
	{
		std::sort(vec.begin(), vec.end());
		vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
	}

	const std::string VP_RESULTS_DIR("vp_results_dir");

	class VPDetectionWrapper //for now I'll just wrap my code around the binaries of uni koblenz or Helmuts code based on Halkon 
	{
		typedef std::pair<pcl::PointXYZ, std::vector<unsigned int> > MapingType;

	public:

		VPDetectionWrapper();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		VPDetectionWrapper(const std::string &image_data_dir, const std::string &out_put_dir);

		VPDetectionWrapper(const std::vector<std::string> &image_filenames, const std::string &out_put_dir);//, const std::string &image_extension, const std::string &unbw_params_dir)

		VPDetectionWrapper(const std::vector<std::vector<std::string> > &corresponding_images_filenames, const std::string &out_put_dir);

		//VPDetection(const std::string &image_filename, const std::string &out_put_dir, const Eigen::Matrix3f &cam_calib);//, const std::string &image_extension, const std::string &unbw_params_dir)

		VPDetectionWrapper(const std::string &image_data_dir, const std::string &out_put_dir, const Eigen::Matrix3f &cam_calib);


		//Destructor
		virtual ~VPDetectionWrapper();


		////////////////////////////////////////////////////////////////////////////////
		/** \brief do a least squares fit on the vp and us ransac  to elliminate bad vps, confirm results using calib matrix
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void validateVanishingPoint(const std::vector<std::vector< Eigen::Vector2f> > &computed_vp_hypothesis, const Eigen::Matrix3f &cam_calib, Eigen::Vector3f &final_robust_vp_x, Eigen::Vector3f &final_robust_vp_y);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief angle between two 3D vectors
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/

		float angleBetweenVectors(const Eigen::Vector3f &vector1, const Eigen::Vector3f &vector2);
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
		bool getVanishingPoint(std::vector<Eigen::Vector2f> &vanishing_pointp);

		bool getVanishingPoint(std::vector<cv::Point2f> &vanishing_pointp);

		bool getVanishingPoint(std::vector<cv::Point3f> &vanishing_pointp);

		bool getVanishingPoint(std::vector<Eigen::Vector3f> &vanishing_pointp);

		bool validateVanishingPoint(const std::vector<Eigen::Vector3f> &vanishing_pointp); //actually one could use a 3x3 matrix

		bool validateVanishingPoint(const std::vector<cv::Point3f> &vanishing_pointp);

		bool validateVanishingPoint(const std::vector<cv::Point2f> &vanishing_point);

		bool validateVanishingPoint(const std::vector<Eigen::Vector2f> &vanishing_pointp);

		void robustlyAdjustVanishingPoint(const std::vector<std::vector<Eigen::Vector2f> > &vanishing_point );

		void robustlyAdjustVanishingPoint(const std::vector<std::vector<Eigen::Vector3f> > &vanishing_point );

	/** \brief collect the set of inliers of the vanishing point prior to least squares fit
	  * \param[in] vp hypothesis
	  * \param[out] Line mode Coeffs where {optimized_coefficients[0], optimized_coefficients[1], optimized_coefficients[2]}=centroid and,
	  *	{optimized_coefficients[3], optimized_coefficients[4], optimized_coefficients[5]}= direction
	  * \return true if successful:
	  */
		bool getRansacInliers(const std::vector<Eigen::Vector3f> &vp_hypothesis, std::vector<Eigen::Vector3f> &in_liers, float angular_tolerance = 5 /* in degrees*/);

	/** \brief Fits an line through a set of points using ordinary least squares
	  * \param[in] Line inliers
	  * \param[out] Line mode Coeffs where {optimized_coefficients[0], optimized_coefficients[1], optimized_coefficients[2]}=centroid and,
	  *	{optimized_coefficients[3], optimized_coefficients[4], optimized_coefficients[5]}= direction
	  * \return true if successful:
	  */
	void leastQuaresVPFitting(const std::vector<Eigen::Vector3f> &vp_inliers, Eigen::VectorXf &optimized_vp);

	/** \brief Fits a plane through a set of points using ordinary least squares
	  * \param[in] Plane inliers
	  * \param[out] Plane model Coefficients in the form Ax + By +Cz + D = 0, where {A = optimized_coefficients[0]
	  *	B = optimized_coefficients[1], C = optimized_coefficients[2] and D = optimized_coefficients[3]
	  * \return true if successful:
	  */
	void leastQuaresLineFitting(const std::vector<Eigen::Vector3f> &inliers, Eigen::VectorXf &optimized_coefficients);

        
		////////////////////////////////////////////////////////////////////////////////
		/** \brief Parse all the files produced by tool and save the vps getting it ready for validation
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool collectVanishingPoints(std::vector<std::vector<Eigen::Vector2f> > &sets_of_vanishing_point);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief computes the vanishing point of an inpute image
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void computeVanishingPoint(const std::string &image_filename, const std::string &vp_out_dir_name);

		
		////////////////////////////////////////////////////////////////////////////////
		/** \brief parse only the vp results file and store the values in Eigen::Vector2f
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void parseVPOutputFile(const std::string &vp_results_file, std::vector<Eigen::Vector2f> &vp_hypothesis);


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

		std::string correctSlashes(std::string &s);


	private:

		//helper function to search for 2D3D-table file
		void getUniqueFileNames(const std::vector<std::vector<std::string> > &corresponding_images_filenames, std::vector<std::string> &image_filenames);

		//3D Model
		std::string m_image_filename;
		std::vector<std::vector<std::string> > m_corresponding_images_filenames; // this just a hack using the mapping table directly
		std::string m_out_put_dir;
		std::string m_image_data_dir;
		std::string m_image_extension;  //this can be gotten from the params file
		std::string m_unibw_params_dir;
		std::string m_dlr_params_dir;
		std::string m_mshr_params_file;
		std::vector<std::string> m_image_filenames; // this should be used actually rathar than the mapping table
	};

}//fi

#endif//_VP_DETECTION_WRAPPER_H_
