#include "ransac_homography.h"


fi::RansacHomography::RansacHomography(const std::vector<Eigen::Vector2d> &point_set_a, const std::vector<Eigen::Vector2d> &point_set_b)
	:m_point_set_a(point_set_a), m_point_set_b(point_set_b)
{


}


bool fi::RansacHomography::computeRansacHomography(Eigen::Matrix3d &best_transform, unsigned int num_iterations, float ransac_threshold, std::vector<Eigen::Vector2d> &match_point_set_a, std::vector<Eigen::Vector2d> &match_point_set_b)
{
	const int num_match_candidates = m_point_set_a.size();
	if (num_match_candidates < 4)
	{
		printf("error: at least 4 match candidates must be provided for RansacHomography::computeRansacHomography (%i provided)\n", num_match_candidates);
		return false;
	}

	Eigen::Matrix3d best_B;
	int i, max_support = 0;

	for (i = 0; i < num_match_candidates; i++)
	{
		// identify 4 different points
		const int nFirstIndex = rand() % num_match_candidates;

		int nTempIndex;

		do { nTempIndex = rand() % num_match_candidates; } while (nTempIndex == nFirstIndex);
		const int nSecondIndex = nTempIndex;

		do { nTempIndex = rand() % num_match_candidates; } while (nTempIndex == nFirstIndex || nTempIndex == nSecondIndex);
		const int nThirdIndex = nTempIndex;

		do { nTempIndex = rand() % num_match_candidates; } while (nTempIndex == nFirstIndex || nTempIndex == nSecondIndex || nTempIndex == nThirdIndex);

		std::vector<Eigen::Vector2d> p_features_a(4);
		std::vector<Eigen::Vector2d> p_features_b(4);

		p_features_a.at(0) = m_point_set_a.at(nFirstIndex); 
		p_features_a.at(1) = m_point_set_a.at(nSecondIndex);
		p_features_a.at(2) = m_point_set_a.at(nThirdIndex);
		p_features_a.at(3) = m_point_set_a.at(nTempIndex);

		p_features_b.at(0) = m_point_set_b.at(nFirstIndex); 
		p_features_b.at(1) = m_point_set_b.at(nSecondIndex);
		p_features_b.at(2) = m_point_set_b.at(nThirdIndex);
		p_features_b.at(3) = m_point_set_b.at(nTempIndex);

		// compute an affine transformation for these points
		Eigen::Matrix3d transform_matrix;
		bool found_homography = determineHomography(p_features_a, p_features_b, 4, transform_matrix, false);

		// count support
		int nSupport = 0;
		for (int j = 0; j < num_match_candidates; j++)
		{
			Eigen::Vector2d test_point;
			applyHomography(transform_matrix, m_point_set_a.at(j), test_point);

			const float distance = distanceBetween2DVectors(test_point, m_point_set_b.at(j)); 

			if (distance < ransac_threshold)
				nSupport++;
		}

		// store if it is the current maximum
		if (nSupport > max_support)
		{
			max_support = nSupport;
			best_B = transform_matrix;
		}
	}

	// filter inliers
	for (i = 0; i < num_match_candidates; i++)
	{

		Eigen::Vector2d test_point;
		applyHomography(best_B, m_point_set_a.at(i), test_point);

		const float distance = distanceBetween2DVectors(test_point, m_point_set_b.at(i)); 

		if (distance < ransac_threshold)
		{
			match_point_set_a.push_back(match_point_set_a.at(i));
			match_point_set_b.push_back(match_point_set_b.at(i));
		}
	}
	best_transform = best_B;
	return true;
}




bool fi::RansacHomography::determineHomography(const std::vector<Eigen::Vector2d> &source_points, const std::vector<Eigen::Vector2d> &target_points, int num_points, Eigen::Matrix3d &transform_matrix, bool make_accurate)
{
	if (num_points < 4)
	{
		printf("error: not enough input point pairs for LinearAlgebra::DetermineHomography (must be at least 4)\n");
		return false;
	}

	// this least squares problem becomes numerically instable when
	// using float instead of double!!
	Eigen::MatrixXd A_mat(num_points, 8);
	Eigen::VectorXd b_vector;
	b_vector.resize(num_points);

	//populate matrix with values
	for ( unsigned int i = 0; i < num_points; i++)
	{

		Eigen::Vector2d s_point;
		s_point(0) = source_points.at(i)(0);
		s_point(1) = source_points.at(i)(1);

		Eigen::Vector2d t_point;
		t_point(0) = target_points.at(i)(0);
		t_point(1) = target_points.at(i)(1);

		if (1%2 == 0)
		{
			A_mat.row(i) << 1, 0, s_point(0), s_point(1),  0, 0, -s_point(0)*t_point(0), -s_point(1)*t_point(0) ;
			b_vector(i) = t_point(0); 
		}else
			A_mat.row(i) << 0, 1, 0, 0, s_point(0), s_point(1),  0, 0, -s_point(0)*t_point(1), -s_point(1)*t_point(1) ;
		b_vector(i) = t_point(1);
	}
	Eigen::VectorXd homography_values;
	homography_values.resize(num_points);

	if (make_accurate)
	{
		homography_values = A_mat.fullPivHouseholderQr().solve(b_vector); //slow but accurate
	}else
		homography_values = A_mat.colPivHouseholderQr().solve(b_vector);  //not too slow and not too accurate

	transform_matrix << homography_values(0), homography_values(1), homography_values(2),
		homography_values(3), homography_values(4), homography_values(5),
		homography_values(6), homography_values(7), 1.0;
	return true;
}


void fi::RansacHomography::applyHomography(const Eigen::Matrix3d &a_mat, const Eigen::Vector2d &p, Eigen::Vector2d &result)
{
	Eigen::Vector3d test_point;
	test_point(0) = p(0);
	test_point(1) = p(1);
	test_point(1) = 1;

	Eigen::Vector3d result_homogenous = a_mat * test_point;
	result(0) = result_homogenous(0)/result_homogenous(2);
	result(1) = result_homogenous(1)/result_homogenous(2);
}


float fi::RansacHomography::distanceBetween2DVectors(const Eigen::Vector2d &vector1, const Eigen::Vector2d &vector2)
{
	const float x1 = vector1(0) - vector2(0);
	const float x2 = vector1(1) - vector2(1);
	return sqrtf(x1 * x1 + x2 * x2);
}
