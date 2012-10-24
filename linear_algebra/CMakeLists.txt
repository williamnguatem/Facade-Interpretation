PROJECT(linear_algebra)

SET(linear_algebra_HEADERS
  include/determine_homography.h
  include/ransac_homography.h
  include/least_squares_fitting.h
)
  
SET(linear_algebra_SOURCES
   src/ransac_homography.cpp
  src/determine_homography.cpp
  src/least_squares_fitting.cpp	
)

#FIND_PACKAGE( OpenCV REQUIRED )

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS( -DANN_STATIC_LIB )
ENDIF()

ADD_LIBRARY(linear_algebra ${linear_algebra_HEADERS} ${linear_algebra_SOURCES}
				# ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} 
				# ${COMMON_LIB}/pcl_common-gd.lib 
				# ${COMMON_LIB}/pcl_filters-gd.lib 
				# ${CMINPACK}/cminpack_s-gd.lib	
				# ${COMMON_LIB}/pcl_sample_consensus-gd.lib 
				# ${COMMON_LIB}/pcl_segmentation-gd.lib
				# ${COMMON_LIB}/pcl_kdtree-gd.lib
				# ${COMMON_LIB}/pcl_visualization-gd.lib
				# ${COMMON_LIB}/pcl_surface-gd.lib
				# ${COMMON_LIB}/pcl_features-gd.lib
				# ${FLANN}/flann_cpp_s-gd.lib
				# ${BOOST_LIB}/libboost_date_time-vc100-mt-gd-1_46_1.lib
				# ${BOOST_LIB}/libboost_thread-vc100-mt-gd-1_46_1.lib
)

#TESTING_CREATE_MODULE_CONF(ann
 # INCLUDE_DIRS include
#)
