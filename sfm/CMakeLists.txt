cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(sfm)


#set modulname or set sub_system_name ${FI_INCLUDE_DIR}
SET(sfm_HEADERS
  include/estimate_fundamental_matrix.h
  include/feauture_matcher.h
  #include/fi_features.h
)
  
SET(sfm_SOURCES
  src/estimate_fundamental_matrix.cpp
  src/feature_matcher.cpp
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


IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set (SIFTGPU_INCLUDE_DIR "C:/williamnguatem/projects/LIBS/SiftGPU-V400/SiftGPU/src/SiftGPU")
INCLUDE_DIRECTORIES(${SIFTGPU_INCLUDE_DIR})



set (SIFTGPU_LIB_RELEASE "C:/williamnguatem/projects/LIBS/SiftGPU-V400/SiftGPU/lib/SIFTGPU.lib")
set (SIFTGPU_LIB_DEBUG "C:/williamnguatem/projects/LIBS/SiftGPU-V400/SiftGPU/lib/SIFTGPU.lib")
set (DEVIL_LIB_RELEASE "C:/williamnguatem/projects/LIBS/SiftGPU-V400/SiftGPU/lib/DevIL64.lib") 




if(WIN32)
set( sfm_release_libs 
		optimized ${Boost_LIBRARIES} 
		optimized ${SIFTGPU_LIB_RELEASE}
	#	optimized ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/io_Release
)
		#optimized ${DEVIL_LIB_RELEASE}
set( sfm_debug_libs 
		debug ${Boost_LIBRARIES}
		debug ${SIFTGPU_LIB_DEBUG}
)
#		debug ${PCL_LIBRARIES}
endif(WIN32)
#		debug ${DEVIL_LIB_RELEASE}


if(UNIX)
set( sfm_release_libs 
		optimized ${Boost_LIBRARIES}
		optimized ${SIFTGPU_LIB_RELEASE}
		optimized ${DEVIL_LIB_RELEASE}
)
#optimized ${PCL_LIBRARIES} 
#optimized ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/fi_core_Release 
set( sfm_debug_libs  
		debug ${Boost_LIBRARIES}
		debug ${SIFTGPU_LIB_DEBUG}
		debug ${DEVIL_LIB_RELEASE}
)
#		debug ${PCL_LIBRARIES}
endif(UNIX)


add_executable (sfm ${sfm_SOURCES} ${sfm_HEADERS})

target_link_libraries (sfm ${sfm_release_libs} ${sfm_debug_libs} ${OpenCV_LIBS})
