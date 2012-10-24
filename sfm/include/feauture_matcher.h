#ifndef _FEATURE_MATCHER_H_
#define _FEATURE_MATCHER_H_

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "SiftGPU.h"
#include <boost/shared_ptr.hpp>

const int MIN_NUM_MATCHES = 9;    //min number of points accepted to say there was a match


//Type of matching done from default opencv too siftgpu
typedef enum { MATCHER0_CPU, MATCHER1_CPU, MATCHER2_CPU, MATCHER3_CPU, MATCHER4_CPU, 
	MATCHER0_GPU, MATCHER2_GPU, MATCHER_ADVANCED_MULTIGPU0, MATCHER_ADVANCED_MULTIGPU1} MatcherType;

//Type of method used to compute FM or TrifocalTensor
typedef enum { FM_OPENCV_RANSAC, TT_NISTER, FMGPU} FM_Type;

typedef std::vector<float > SiftDescriptors;
typedef std::vector<SiftGPU::SiftKeypoint> SiftKeys;

typedef boost::shared_ptr<SiftDescriptors> SiftDescriptorsPtr;
typedef boost::shared_ptr<SiftKeys> SiftKeysPtr;


namespace sfm
{
	//typedef PointPair2D std::pair<Eigen::Vector2f, Eigen::Vector2f>;

	class FeatureMatcher //for now I'll just wrap my code around the binaries of uni koblenz or Helmuts code based on Halkon 
	{

	public:

		FeatureMatcher();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		FeatureMatcher(const std::vector<Eigen::Vector2d> &point_set_a , const std::vector<Eigen::Vector2d> &point_set_b);

		FeatureMatcher(const std::string &input_image_folder, const std::string &image_extension);


		bool initMatcher(const MatcherType  &matching_type);
		//RansacFundamentalMatrix(const std::vector<Eigen::Vector2d> &point_set_a , const std::vector<Eigen::Vector2d> &point_set_b);


		//Destructor
		virtual ~FeatureMatcher();


		////////////////////////////////////////////////////////////////////////////////
		/** \brief do a least squares fit on the vp and us ransac  to elliminate bad vps, confirm results using calib matrix
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool computeRansacFundamentalMatrix(Eigen::Matrix3d &best_homography_mat, unsigned int num_iterations, float ransac_threshold, std::vector<Eigen::Vector2d> &match_point_set_a, std::vector<Eigen::Vector2d> &match_point_set_b);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief get the names of the input files from the given directory
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool getImageFileNames(std::vector<std::string> &images_filenames);

		////////////////////////////////////////////////////////////////////////////////
		/** \brief Apply the homography on an input data set
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		void applyFundamentalMatrix(const Eigen::Matrix3d &a_mat, const Eigen::Vector2d &p, Eigen::Vector2d &result);


		unsigned int matchtKeyPoints();


		///////////////////////////////////////////////////////////////////////////////
		/** \brief Compute the fundamental matrix using opencv's RANSAC based method
		* \input at least MIN_NUM_MATCHES point correspondences
		* \param 
		*/
		void computeFundamentalMatrixCPU0(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, Eigen::Matrix3d &fundamental_matrix);


		///////////////////////////////////////////////////////////////////////////////
		/** \brief Compute the triangulation using weighted least squares to the get the points in 3D space
		* \input at least MIN_NUM_MATCHES point correspondences
		* \param 
		*/
		void robustTriangulationCPU0(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, Eigen::Matrix3d &fundamental_matrix);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief distance between 2 2D vectors
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		float distanceBetween2DVectors(const Eigen::Vector2d &vector1, const Eigen::Vector2d &vector2);

		bool determineFundamentalMatrix(const std::vector<Eigen::Vector2d> &source_points, const std::vector<Eigen::Vector2d> &target_points, int num_points, Eigen::Matrix3d &transform_matrix, bool bUseSVD);

	private:
		
		//define all the various matchers
		int cpumatcher0();
		int cpumatcher1();
		int cpumatcher2();
		int cpumatcher3();
		int cpumatcher4();
		int gpumatcher0();

		//3D Model
		std::vector<Eigen::Vector2d> m_point_set_a;
		std::vector<Eigen::Vector2d> m_point_set_b;
		std::string m_image_data_dir;
		std::string m_image_extension;
		std::vector<std::string> m_image_filenames;
		MatcherType m_matching_type;
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

#endif//_FEATURE_MATCHER_H_
