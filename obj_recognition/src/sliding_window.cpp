#include "sliding_window.h"


fi::SlidingWindow::SlidingWindow(){};

fi::SlidingWindow::SlidingWindow(const cv::Mat &input_image, const int window_width, const int window_height, const int sliding_dx, const int sliding_dy)
	:m_input_image(input_image), m_window_width(window_width), m_window_height(window_height), m_sliding_dx(sliding_step), m_sliding_dy(sliding_dy)
{

}



bool fi::SlidingWindow::slide_over_image(std::vector<cv::Mat> &sud_images)
{

	//image = imread('image.png');
	//imageWidth = size(image, 2);
	//imageHeight = size(image, 1);

	//windowWidth = 32;
	//windowHeight = 32;

	//for j = 1:imageHeight - windowHeight + 1
	//	for i = 1:imageWidth - windowWidth + 1
	//		window = image(j:j + windowHeight - 1, i:i + windowWidth - 1, :);
	//% do stuff with subimage
	//	end
	//	end
	//cv::Mat imageROI= image(cv::Range(270,270+logo.rows),
	//	cv::Range(385,385+logo.cols))

	return true;
}
