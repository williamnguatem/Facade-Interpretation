#ifndef _ESTIMATE_FUNDAMENTAL_MATRIX_H_
#define _ESTIMATE_FUNDAMENTAL_MATRIX_H_

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <boost/filesystem.hpp>
#include <Eigen/Dense>



namespace sfm
{
	//typedef PointPair2D std::pair<Eigen::Vector2f, Eigen::Vector2f>;

	class RansacFundamentalMatrix //for now I'll just wrap my code around the binaries of uni koblenz or Helmuts code based on Halkon 
	{

	public:

		RansacFundamentalMatrix();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		RansacFundamentalMatrix(const std::vector<Eigen::Vector2d> &point_set_a , const std::vector<Eigen::Vector2d> &point_set_b);


		//RansacFundamentalMatrix(const std::vector<Eigen::Vector2d> &point_set_a , const std::vector<Eigen::Vector2d> &point_set_b);


		//Destructor
		virtual ~RansacFundamentalMatrix();


		////////////////////////////////////////////////////////////////////////////////
		/** \brief do a least squares fit on the vp and us ransac  to elliminate bad vps, confirm results using calib matrix
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool computeRansacFundamentalMatrix(Eigen::Matrix3d &best_homography_mat, unsigned int num_iterations, float ransac_threshold, std::vector<Eigen::Vector2d> &match_point_set_a, std::vector<Eigen::Vector2d> &match_point_set_b);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Apply the homography on an input data set
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void applyFundamentalMatrix(const Eigen::Matrix3d &a_mat, const Eigen::Vector2d &p, Eigen::Vector2d &result);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief distance between 2 2D vectors
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		float distanceBetween2DVectors(const Eigen::Vector2d &vector1, const Eigen::Vector2d &vector2);

		bool determineFundamentalMatrix(const std::vector<Eigen::Vector2d> &source_points, const std::vector<Eigen::Vector2d> &target_points, int num_points, Eigen::Matrix3d &transform_matrix, bool bUseSVD);

	private:

		//3D Model
		std::vector<Eigen::Vector2d> m_point_set_a;
		std::vector<Eigen::Vector2d> m_point_set_b;
		//std::vector<std::vector<std::string> > m_corresponding_images_filenames; // this just a hack using the mapping table directly
		//std::string m_out_put_dir;
		//std::string m_image_data_dir;
		//std::string m_image_extension;  //this can be gotten from the params file
		//std::string m_unibw_params_dir;
		//std::string m_dlr_params_dir;
		//std::string m_mshr_params_file;
		//std::vector<std::string> m_image_filenames; // this should be used actually rathar than the mapping table
	};

}//fi

#endif//_ESTIMATE_FUNDAMENTAL_MATRIX_H_
