cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cylinder_segmentation_example)


#set modulname or set sub_system_name ${FI_INCLUDE_DIR}
SET(cylinder_segmentation_HEADERS
  include/cylinder_segmentation.h
  #include/fi_algorithms.h
  #include/fi_features.h
)
  
SET(cylinder_segmentation_SOURCES
  src/cylinder_segmentation.cpp
  #src/fi_features.cpp
  #src/fi_algorithms.cpp
  main.cpp
 )

 IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS( -DANN_STATIC_LIB )
ENDIF()

# find_package(PCL 1.2 REQUIRED)

# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})


################################################################################################
#				VTK LIBRARY 						#
################################################################################################
# IF(NOT VTK_BINARY_DIR)
# FIND_PACKAGE(VTK)
# IF(NOT VTK_DIR)
  # MESSAGE(FATAL_ERROR "Please set VTK_DIR.")
# ENDIF(NOT VTK_DIR)
# INCLUDE(${VTK_USE_FILE})
# ENDIF(NOT VTK_BINARY_DIR)


add_executable (cylinder_segmentation_example ${cylinder_segmentation_SOURCES} ${cylinder_segmentation_HEADERS})

target_link_libraries (cylinder_segmentation_example ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES}  ${PCL_FILTERS_LIBRARIES} ${PCL_KDTREE_LIBRARIES} 
${PCL_SURFACE_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES} ${PCL_SEARCH_LIBRARIES}
${PCL_SEGMENTATION_LIBRARIES})