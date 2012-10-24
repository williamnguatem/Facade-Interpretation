PROJECT(vp_detection)

#set modulname or set sub_system_name ${FI_INCLUDE_DIR}
SET(vp_detection_HEADERS
  include/vp_detection_wrapper.h
  include/vp_detection.h
  include/vp_detection_cloud.h
  include/vp_detection_context.h
  include/vp_detection_image.h
)
  
SET(vp_detection_SOURCES
  src/vp_detection_wrapper.cpp
  src/vp_detection.cpp
  src/vp_detection_cloud.cpp
  src/vp_detection_context.cpp
  src/vp_detection_image.cpp
 )

#FIND_PACKAGE( OpenCV REQUIRED )

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS( -DANN_STATIC_LIB )
ENDIF()

ADD_LIBRARY(vp_detection ${vp_detection_HEADERS} ${vp_detection_SOURCES} 
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
