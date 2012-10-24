PROJECT(io)

SET(io_HEADERS
  include/sgm_paramfile_parser.h
  include/sgm_error_ellipse_parser.h
  include/mshr_paramfile_parser.h
  include/mshr_3D2D_table_parser.h
  include/mshr_cov_matrix_parser.h
  include/unibw_cam_pose_parser.h
   include/dlr_cam_pose_parser.h
  include/mshr_cam_calib_parser.h
  include/file_io.h
)
  
SET(io_SOURCES
	src/sgm_paramfile_parser.cpp
	src/sgm_error_ellipse_parser.cpp
	src/mshr_paramfile_parser.cpp
	src/mshr_3D2D_table_parser.cpp
	src/mshr_cov_matrix_parser.cpp
	src/unibw_cam_pose_parser.cpp
	src/dlr_cam_pose_parser.cpp
	src/mshr_cam_calib_parser.cpp
	src/file_io.cpp
)

#FIND_PACKAGE( OpenCV REQUIRED )

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS( -DANN_STATIC_LIB )
ENDIF()

ADD_LIBRARY(io ${io_HEADERS} ${io_SOURCES}
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
