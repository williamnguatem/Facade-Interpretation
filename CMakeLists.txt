CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

#set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "possible configurations" FORCE)
# In case the user does not setup CMAKE_BUILD_TYPE, assume it's RelWithDebInfo
#if("${CMAKE_BUILD_TYPE}" STREQUAL "")
 # set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "build type default to RelWithDebInfo, set to Release to improve performance" FORCE)
 #set(CMAKE_BUILD_TYPE Debug;Release CACHE STRING "build type default to RelWithDebInfo, set to Release to improve performance" FORCE)
#endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

 if(CMAKE_CONFIGURATION_TYPES)
   set(CMAKE_CONFIGURATION_TYPES Debug Release)
   set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
     "Reset the configurations to what we need"
     FORCE)
 endif()



PROJECT(fi)
INCLUDE_REGULAR_EXPRESSION("^([Nn]rrd|itk|vtk|vnl|vcl|vxl|v3p|f2c|netlib|ce|itpack|DICOM|meta|png|dbh|tif|jpeg|zlib|gdcm).*$")
SOURCE_GROUP("XML Files" REGULAR_EXPRESSION "[.]xml$")

# On Visual Studio 8 MS deprecated C. This removes all 1.276E1265 security
# warnings
IF(WIN32)
  IF(NOT BORLAND)
    IF(NOT CYGWIN)
      IF(NOT MINGW)
        IF(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
          ADD_DEFINITIONS(
            -D_CRT_FAR_MAPPINGS_NO_DEPRECATE
            -D_CRT_IS_WCTYPE_NO_DEPRECATE
            -D_CRT_MANAGED_FP_NO_DEPRECATE
            -D_CRT_NONSTDC_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE_GLOBALS
            -D_CRT_SETERRORMODE_BEEP_SLEEP_NO_DEPRECATE
            -D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
            -D_CRT_VCCLRIT_NO_DEPRECATE
            -D_SCL_SECURE_NO_DEPRECATE
            )
        ENDIF(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
      ENDIF(NOT MINGW)
    ENDIF(NOT CYGWIN)
  ENDIF(NOT BORLAND)
ENDIF(WIN32)

# If CMake is the version we're expecting, don't show the
# CMAKE_BACKWARDS_COMPATIBILITY option. If the version is higher than the
# minimal version required, then show the backward compatibility option.
IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" MATCHES "^2\\.4$")
  MARK_AS_ADVANCED(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" MATCHES "^2\\.4$")
  MARK_AS_ADVANCED(CLEAR CMAKE_BACKWARDS_COMPATIBILITY)
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" MATCHES "^2\\.4$")

#SET(CMAKE_BUILD_TYPE Debug)
#-----------------------------------------------------------------------------
# ITK version number.  An even minor number corresponds to releases.
SET(fi_VERSION_MAJOR 1)
SET(fi_VERSION_MINOR 10)
#SET(fi_VERSION_PATCH "0")

# Version string should not include patch level.  The major.minor is
# enough to distinguish available features of the toolkit.
SET(fi_VERSION_STRING "${VisualFacadeReconstructor_VERSION_MAJOR}.${VisualFacadeReconstructor_VERSION_MINOR}")

#Set build directory for the out of source build
SET(PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
SET (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH})

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
#find_package(PCL 1.0 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})


SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH} )
#SET(${CMAKE_RUNTIME_OUTPUT_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/Debug) #if debug, if release set a new one
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_DEBUG_POSTFIX "_Debug" CACHE STRING "Add a postfix, usually d on windows")
SET(CMAKE_RELEASE_POSTFIX "_Release" CACHE STRING "Add a postfix, usually d on windows")


#Not yet working!!!!!!!!!!
SET(TESTING_MODULES_CONF_DIR ${TESTING_BINARY_DIR}/modulesConf CACHE INTERNAL "Modules Conf")
#ADD_SUBDIRECTORY(cmake)


#############################Compiler check for 64-bit#####################
if(WIN32)
	if(MSVC)
		if(MSVC10)
			MESSAGE(STATUS "MSVS Version 10")
			Set(HAVE_C++0X 1)
		endif()
		if(${CMAKE_CL_64})
			MESSAGE(STATUS "Compiling on x64" )
		endif()
	endif()
endif()





#CUDA
set(WITH_CUDA ON CACHE BOOL "Include NVidia Cuda Runtime support")

############################### CUDA ################################
if(WITH_CUDA)
    find_package(CUDA 4.0)
    
    if(CUDA_FOUND)
        set(HAVE_CUDA 1)
        message(STATUS "CUDA detected: " ${CUDA_VERSION})

        set(CUDA_ARCH_BIN "1.1 1.2 1.3 2.0 2.1(2.0) 3.0" CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
        set(CUDA_ARCH_PTX "2.0" CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
        
        string(REGEX REPLACE "\\." "" ARCH_BIN_NO_POINTS "${CUDA_ARCH_BIN}")
        string(REGEX REPLACE "\\." "" ARCH_PTX_NO_POINTS "${CUDA_ARCH_PTX}")
        
        # Ckeck if user specified 1.0 compute capability: we don't support it
        string(REGEX MATCH "1.0" HAS_ARCH_10 "${CUDA_ARCH_BIN} ${CUDA_ARCH_PTX}")
        set(CUDA_ARCH_BIN_OR_PTX_10 0)
        if(NOT ${HAS_ARCH_10} STREQUAL "")
            set(CUDA_ARCH_BIN_OR_PTX_10 1)
        endif()
        
        # NVCC flags to be set
        set(NVCC_FLAGS_EXTRA "")      
        
        # These vars will be passed into the templates
        #set(OPENCV_CUDA_ARCH_BIN "") 
        #set(OPENCV_CUDA_ARCH_PTX "") 
        #set(OPENCV_CUDA_ARCH_FEATURES "")        
        
        # Tell NVCC to add binaries for the specified GPUs
        string(REGEX MATCHALL "[0-9()]+" ARCH_LIST "${ARCH_BIN_NO_POINTS}")
        foreach(ARCH IN LISTS ARCH_LIST)
            if (ARCH MATCHES "([0-9]+)\\(([0-9]+)\\)")
                # User explicitly specified PTX for the concrete BIN
                set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
                #set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${CMAKE_MATCH_1}")
                #set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${CMAKE_MATCH_2}")
            else()
                # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN                
                set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=sm_${ARCH})
                #set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${ARCH}")
                #set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
            endif()
        endforeach()
        
        # Tell NVCC to add PTX intermediate code for the specified architectures
        string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
        foreach(ARCH IN LISTS ARCH_LIST)
            set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=compute_${ARCH})
            #set(OPENCV_CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX} ${ARCH}")
            #set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
        endforeach()
        
        # These vars will be processed in other scripts
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})        
        #set(OpenCV_CUDA_CC "${NVCC_FLAGS_EXTRA}")
        
        message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")
    else()
        unset(CUDA_ARCH_BIN CACHE)
        unset(CUDA_ARCH_PTX CACHE)  
    endif() 
endif()


SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH} )

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})




#################################################################################################
# 				OPENCV LIBRARY 						#
#################################################################################################
FIND_PACKAGE( OpenCV REQUIRED )




#################################################################################################
# 				VTK LIBRARY 						#
#################################################################################################
IF(NOT VTK_BINARY_DIR)
FIND_PACKAGE(VTK)
IF(NOT VTK_DIR)
  MESSAGE(FATAL_ERROR "Please set VTK_DIR.")
ENDIF(NOT VTK_DIR)
INCLUDE(${VTK_USE_FILE})
ENDIF(NOT VTK_BINARY_DIR)


# #################################################################################################
# # 				QT LIBRARY 						#
# #################################################################################################
# SET(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
# SET(QT_MOC_EXECUTABLE ${VTK_QT_MOC_EXECUTABLE} CACHE FILEPATH "")
# SET(QT_UIC_EXECUTABLE ${VTK_QT_UIC_EXECUTABLE} CACHE FILEPATH "")
# FIND_PACKAGE(Qt4)
# IF(QT_USE_FILE)
  # INCLUDE(${QT_USE_FILE})
# ELSE(QT_USE_FILE)
  # SET(QT_LIBRARIES   ${QT_QT_LIBRARY})
# ENDIF(QT_USE_FILE)

# QT4_WRAP_UI(UI_SRCS MainApp.ui)
# QT4_WRAP_CPP(MOC_SRCS GUI4.h)



#############################################################################################
#	Boost LIB
#############################################################################################
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/find_boost.cmake)

#############################################################################################
#	FLANN
###############################################################
# FLANN (optional)
#find_package(Flann REQUIRED)
#find_package(Flann)
#include_directories(${FLANN_INCLUDE_DIRS})

##############################################################################################
# Qhull
###########################
#find_package(Qhull)
#include_directories(${QHULL_INCLUDE_DIRS})


#############################################################################################
#                 EIGEN
######################################
find_package(EIGEN)
include_directories(${EIGEN_INCLUDE_DIRS})


#OPTION( OPTION_VAR \93description string\94 [initial value] )
OPTION(ENABLE_MULTITHREADING "Make extensive use of Boost Threads" ON)
IF(${ENABLE_MULTITHREADING} STREQUAL "ON")
SET(USE_MULTITHREADING 1)
MESSAGE(STATUS "Make sure you're on a multicore platform for an extensive thread usage!")
ENDIF()


#OPTION( VPTOOL " " [initial value] )

#set(VPDETECTION_TOOL "C:/williamnguatem/projects/LIBS/vpointssource/vpoints-win-bin")

SET(VPDETECTION_TOOL CACHE STRING "Vanishing Point Detection Tool from koblenz")


CONFIGURE_FILE(
		"${PROJECT_SOURCE_DIR}/fi_config.h.in"
		"${PROJECT_SOURCE_DIR}/fi_config.h"
		[ESCAPE_QUOTES]
		)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}")

##############################################################################################
#	More Libs //TODO: use a for loop to navigate and add all these different projects
##############################################################################################
#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/Utilities/FacadeDataTypes/include")
#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/Utilities/CuboidSegmentation/include")
#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/Utilities/CuboidDetection/include")
#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/Utilities/FacadeIO/include")

set (COMMON_INCLUDE_DIR "common/include")
set (VP_DETECTION_INCLUDE_DIR "vp_detection/include")
set (UTILS_INCLUDE_DIR "utils/include")
set (IO_INCLUDE_DIR "io/include")
set (FI_CORE_INCLUDE_DIR "fi_core/include")
set (FI_MODEL_SEGMENTATION_INCLUDE_DIR "model_segmentation/include")
set (FI_LINEAR_ALGEBRA_INCLUDE_DIR "linear_algebra/include")
set (SFM_INCLUDE_DIR "sfm/include")
set (OBJ_RECOGNITION_DIR "obj_recognition/include")


INCLUDE_DIRECTORIES(${COMMON_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${VP_DETECTION_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${UTILS_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${IO_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${FI_CORE_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${FI_MODEL_SEGMENTATION_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${FI_LINEAR_ALGEBRA_INCLUDE_DIR}) 
INCLUDE_DIRECTORIES(${SFM_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OBJ_RECOGNITION_DIR})

ADD_SUBDIRECTORY(logs)
ADD_SUBDIRECTORY(common)
ADD_SUBDIRECTORY(obj_recognition)
ADD_SUBDIRECTORY(vp_detection)
ADD_SUBDIRECTORY(model_segmentation)
ADD_SUBDIRECTORY(window_door_detector)
ADD_SUBDIRECTORY(utils)
ADD_SUBDIRECTORY(io)
ADD_SUBDIRECTORY(fi_core)
ADD_SUBDIRECTORY(sand_box)
ADD_SUBDIRECTORY(test_fi)
ADD_SUBDIRECTORY(test_vp)
ADD_SUBDIRECTORY(validate_fi)
ADD_SUBDIRECTORY(validate_vp)
ADD_SUBDIRECTORY(tools)
ADD_SUBDIRECTORY(linear_algebra)
ADD_SUBDIRECTORY(cylinder_segmentation_example)
ADD_SUBDIRECTORY(sfm)




#target_link_libraries (fi ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS})
