#ifndef _SLIDING_WINDOW_H_
#define _SLIDING_WINDOW_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <string>
#include <vector>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <boost/filesystem.hpp>
//#include <Eigen/Dense>



namespace fi
{
	//typedef PointPair2D std::pair<Eigen::Vector2f, Eigen::Vector2f>;

	class SlidingWindow //for now I'll just wrap my code around the binaries of uni koblenz or Helmuts code based on Halkon 
	{

	public:

		SlidingWindow();

		//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		SlidingWindow(const cv::Mat &input_image, const int window_width, const int window_height, const int sliding_dx, const int sliding_dy);

		//Destructor
		virtual ~SlidingWindow();


		////////////////////////////////////////////////////////////////////////////////
		/** \brief slide window over image capturing all resulting sub-images
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool slide_over_image(std::vector<cv::Mat> &sud_images);

	private:

		cv::Mat m_input_image;
		int m_window_width;
		int m_window_height;
		int m_sliding_dx;
		int m_sliding_dy;

	};

}//fi

#endif//_SLIDING_WINDOW_H_
