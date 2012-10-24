///******************************************************************************
//\par		VisualFacadeReconstructor - Intelligent Visual Facade Reconstructor
//\file		house_model_builder.h
//\author		William Nguatem
//\note		Copyright (C) 
//\note		Bundeswehr University Munich
//\note		Institute of Applied Computer Science
//\note		Chair of Photogrammetry and Remote Sensing
//\note		Neubiberg, Germany
//\since		2012/02/07 
//******************************************************************************/

//#pragma once;

#ifndef _HOUSE_MODEL_BUILDER_H_
#define _HOUSE_MODEL_BUILDER_H_

#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <pcl/ModelCoefficients.h>

//visualizer
#include "vtkVertexGlyphFilter.h"
#include <vtkVersion.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkJPEGReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTexture.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkFloatArray.h>
#include <vtkPolygon.h>


//parse input
#include "file_io.h"
#include "mshr_3D2D_table_parser.h"
#include "mshr_cam_calib_parser.h"
#include "unibw_cam_pose_parser.h"
#include "dlr_cam_pose_parser.h"
#include "model_segmentation.h"
#include "plane_model.h"
#include "model_context.h"

//detect vp
#include "vp_detection.h"
#include "vp_detection_cloud.h"
#include "vp_detection_context.h"
#include "vp_detection_wrapper.h"

//plane segmentation
#include "model_segmentation.h"
#include "plane_model.h"
#include "model_context.h"

namespace fi
{
	typedef enum {
		CLOUD,
		IMAGE,
	} VPDetectionType;

	typedef boost::tuple< Eigen::VectorXf, unsigned int, unsigned int > Intersectors;
	typedef boost::tuple<unsigned int, std::vector<unsigned int> > PlaneLines;

	class HouseModelBuilder
	{
	
	public:

		HouseModelBuilder();

		HouseModelBuilder(const std::string &mshr_result_dir, const std::string &image_data_dir, const std::string &image_extension, const std::string &unbw_params_dir);

				//////////////////////////////////////////////////////////////////////////////////
		/** \brief set the results folder from mshr
		* \vtkDataSet object containing the point cloud.
		* \param sVtkpolydata the name of the file containing the vtkpolydata, sOutpclData the output pcl data
		*/
		HouseModelBuilder(const std::string &mshr_pose_dir, const std::vector<std::vector<unsigned int> > image_names);

		HouseModelBuilder(const std::string &mshr_pose_dir, const std::vector<std::string> image_names);

		//Destructor
		virtual ~HouseModelBuilder();


		void initModel(const VPDetectionType &vp_detection_type);


		bool getIntersectionLines(const pcl::ModelCoefficients::Ptr &model_coefficients_a, const pcl::ModelCoefficients::Ptr &model_coefficients_b, Eigen::VectorXf &intersection_line );


		void reconstructModel();

		void projectPointOnLine(const pcl::PointXYZ &pPoint, const Eigen::VectorXf &fOptimLinesCoefs, pcl::PointXYZ &qPoint);

		bool pointIsOnPlane(const pcl::ModelCoefficients::Ptr &model_coefficients_a, const Eigen::Vector3f &pPoint );


		void findQuads(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
			const std::vector<SegmentationResultPtr> &segments_results,  
			const	std::vector<Intersectors> &test_polys, 
			const std::vector<PlaneLines> &plane_outlines, 
			const std::vector<std::vector<std::string> > &corresponding_images_filenames,
			const Eigen::Vector3f &vanishing_point_x,
			const Eigen::Vector3f &vanishing_point_y,
			const Eigen::Matrix3f &cam_calib,
			const std::string &unibw_result_dir
			);


		void getProjectionPointOnImage(const pcl::PointXYZ &reconstructed_3D_point,	
			Eigen::Vector2f &image_point, 
			const std::string &unibw_result_dir,
			const std::string &image_file_name,
			const Eigen::Matrix3f &cam_calib);

		void addLineToRenderer(const pcl::PointXYZ &fPA, const pcl::PointXYZ &fPB, vtkSmartPointer<vtkActor> &fLineActor);

		void addCloudActorToRenderer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, vtkSmartPointer<vtkActor> &fCloudActor);

		//get the edges to plot
		void GetEdges(const unsigned int fTopPercent, const Eigen::Vector3f &fVP, std::vector<int> &fEdges);

		//reorder the hull points of the quad
		void getCloudHull(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, pcl::PointCloud<pcl::PointXYZ>::Ptr &ordered_hull_points);

		//add quad to actor
		void addQuadActorToRenderer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &fQuadEdges, vtkSmartPointer<vtkActor> &fQuadActor);

		//add textured quad to actor
		void addTexturedQuadActorToRenderer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, const std::string &image_name_jpeg, vtkSmartPointer<vtkActor> &textured_quad_actor);

		void addTexturedQuadActorToRendererR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, const std::string &image_name_jpeg, vtkSmartPointer<vtkActor> &textured_quad_actor);
		void addTexturedQuadActorToRendererRR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &quad_points, const std::string &image_name_jpeg, vtkSmartPointer<vtkActor> &textured_quad_actor);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the vtkpolydata into point cloud format for easy access of xyz values
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool getCamPose(std::vector<std::vector<Eigen::Matrix3f> > &rotation_matrix, 
			std::vector<std::vector<Eigen::Vector3f> > &translation_vector, 
			std::vector<std::vector<float> > &radial_distortion_params_k1, 
			std::vector<std::vector<float> > &radial_distortion_param_k2,
			std::vector<std::vector<unsigned int> > &image_width,
			std::vector<std::vector<unsigned int> > &image_height);

		//ToDo: Move this to private since its only needed within this class!
		void getCamRotationMatrix(const boost::filesystem::path &pose_file, Eigen::Matrix3f &rotation_matrix);
		void getCamTranslationVector(const boost::filesystem::path &pose_file, Eigen::Vector3f &translation_vector);
		void getRadialDistortionParams(const boost::filesystem::path &pose_file, float &radial_distortion_params_k1, float &radial_distortion_param_k2);
		unsigned int getImageWidth(const boost::filesystem::path &pose_file);
		unsigned int getImageHeight(const boost::filesystem::path &pose_file);


		////////////////////////////////////////////////////////////////////////////////
		/** \brief Get the point clouds parsed from the 3D2D-table datei
		* vtkDataSet object containing the point cloud.
		* \param sInPolyData the name of the file containing the vtkpolydata, sOutCVecData the output pcl data
		*/
		bool getCameraCalib(Eigen::Matrix3f &calib_cam);

		
		bool getCameraCalibrationFile(
			const boost::filesystem::path & dir_path,         // in this directory,
		//	const std::string & file_name, // search for this name,
			boost::filesystem::path & path_found );		 // camera calibration file


	private:

		//3D Model
		//3D Model
		std::string m_mshr_result_dir;
		std::string m_image_data_dir;
		std::string m_image_extension;  //this can be gotten from the params file
		std::string m_unibw_params_dir;
		std::string m_dlr_params_dir;
		std::string m_mshr_params_file;

		//original input cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;

		//computed from the vp
		Eigen::Vector3f m_horizon_direction;
		Eigen::Vector3f m_zenith_direction;

		//get this after have parse mshr_results folder
		Eigen::Matrix3f m_cam_calib;
		std::vector<std::vector<std::string> > m_corresponding_images_filenames;
		std::vector<std::vector<std::string> > m_mapping_table;


		std::string m_mshr_pose_dir;
		std::vector<std::vector<unsigned int> > m_image_names;
		//std::vector<std::string> m_image_names; //at the moment mshr needs but numbers to save the images maybe the'll change things to strings
	};
}//fi

#endif//_HOUSE_MODEL_BUILDER_H_





















//
////Delete_me
//#include <vtkPolyData.h>
//#include <vtkPointData.h>
//#include <vtkCellArray.h>
//#include <vtkUnsignedCharArray.h>
//#include <vtkRenderWindowInteractor.h>
//#include <vtkVertexGlyphFilter.h>
//#include <vtkProperty.h>
//#include <vtkCleanPolyData.h>
//#include <vtkSphereSource.h>
//#include <vtkMath.h>
//#include <vtkGlyph3D.h>
//#include <vtkArrowSource.h>
//#include <vtkBrownianPoints.h>
//#include <vtkTransform.h>
//#include <vtkTransformPolyDataFilter.h>
//#include <vtkMath.h>
//#include <vtkImageData.h>
//#include <vtkImageImport.h>
//#include <vtkDataSetMapper.h>
//#include <vtkViewTheme.h>
//#include <vtkOrientationMarkerWidget.h>
//#include <vtkAxesActor.h>
//#include <vtkPropAssembly.h>
//#include <vtkDataSetMapper.h>
//#include <vtkLookupTable.h>
//#include <vtkOutlineFilter.h>
//#include <vtkCellArray.h>
//#include <vtkPoints.h>
//#include <vtkHexahedron.h>
//#include <vtkUnstructuredGrid.h>
//#include <vtkTriangle.h>
//
//#include <vtkCellData.h>
//#include <vtkProperty2D.h>
//#include <vtkMapper2D.h>
//#include <vtkLeaderActor2D.h>
//
//#include <vtkSphereSource.h>
//#include <vtkDoubleArray.h>
//#include <vtkFieldData.h>
//#include <vtkXYPlotActor.h>
//
//#include <vtkCellArray.h>
//#include <vtkPoints.h>
//#include <vtkQuad.h>
//#include <vtkPolyData.h>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <vtkImageActor.h>
//#include <vtkWindowToImageFilter.h>
//#include <vtkPNGWriter.h>
//#include <vtkJPEGReader.h>
//#include <vtkSimplePointsReader.h>
//#include <vtkSmartPointer.h>
//#include <vtkVertexGlyphFilter.h>
//#include <vtkPoints.h>
//#include <vtkProperty.h>
//#include "vtkRenderWindow.h"
//#include "vtkRenderer.h"
//#include "vtkCommand.h"
//#include "vtkEventQtSlotConnect.h"
//#include "vtkConeSource.h"
//#include "vtkSphereSource.h"
//#include <vtkPolyDataMapper.h>
//#include "vtkActor.h"
//#include "vtkInteractorStyle.h"
//#include "vtkTDxInteractorStyleCamera.h"
//#include "vtkTDxInteractorStyleSettings.h"
//#include "CUBOIDSEGMENTATION/InputParser.h"
//#include "vtkCamera.h"
//
//#include <vtkActor.h>
//#include <vtkCellArray.h>
//#include <vtkInteractorStyleTrackballActor.h>
//#include <vtkObjectFactory.h>
//#include <vtkCubeSource.h>
//#include <vtkSphereSource.h>
//#include <vtkPoints.h>
//#include <vtkPolyData.h>
//#include <vtkPolyDataMapper.h>
//#include <vtkPropPicker.h>
//#include <vtkRenderWindow.h>
//#include <vtkRenderWindowInteractor.h>
//#include <vtkRenderer.h>
//
////-end
//
//#include <string>
//#include "FacadeDataTypes/FacadeDataTypes.h"
//
//
///*!
//\brief 		Main Class for building an LOD model
//\ingroup 	ModelSegmentation
//*/
//
//
//class HouseModelBuilder
//{
//public:
//
//	~HouseModelBuilder();
//
//	/** \brief Set the name of the input cloud
//	* \param[in] Input Cloud
//	* \param[out] :ToDo: If needed specify the name of the output file
//	* \return void 
//	*/
//	void setInputCloud(const PtCloudDataPtr &fInputcloud);
//
//	/** \brief Set the desired model parameters
//	* \param[in] Parameters defining the model
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void setModelParameter(const PolygonModelParamsPtr  &fModelParams); 
//
//	void buildModel(std::vector<VerticalEdgeLinePtr> &fVPlines, PtCloudDataPtr &fCloudTopPoints, PtCloudDataPtr &fCloudBottomPoints, std::vector<fWallLinkagePtr> &m_LinkageMatrix, std::vector<fWallLinkagePtr> &ValidatedLinkageMatrix);
//
//	void ExtractModel(std::vector<Eigen::Vector4f> &fHousModelWalls);
//
//	void getHouseWallSegemenst(std::vector<SegmentationResultPtr> &_HouseWallSegements);
//
//	void getVP(Eigen::VectorXf &_emtimatedVP);
//
//
//
//
//	
//	
//
//	
//
//	//void setSegmentAllModels(const bool &_segmentAll); 
//	//bool getSegmentAllModels(); 
//
//	///** \brief Additional criteria if many models are to be segmented
//	//  * \param[in] The maximum number of models to be segmented from the cloud
//	//  * \param[out] :ToDo: 
//	//  * \return void 
//	//  */
//	//void setMaxNumOfModelsToSegment(const unsigned  int &_maxNumModels = 1);  //Default always the best
//	//unsigned int getMaxNumOfModelsToSegment();  //Default always the best, i.e. = 1
//
//	///** \brief Additional criteria if many models are to be segmented
//	//  * \param[in] set the min number of per model
//	//  * \param[out] :ToDo: 
//	//  * \return void 
//	//  */
//	//void setMinModelInliers(const unsigned  int &_minModelInliers = 2); //default is 2 for the case that the model is a line
//	//unsigned int getMinModelInliers(); 
//
//	///** \brief Determine if the output should be printed to a file
//	//  * \param[in] true if the output should be printed to a file els retain the output in the system for further processing
//	//  * \param[out] :ToDo: If needed specify the name of the output file
//	//  * \return void 
//	//  */
//	//void setOutFile(const bool &_f = true );
//
//	///** \brief Extract the segmentation results into the 3 tuple
//	//  * \param[in] true if the output should be printed to a file els retain the output in the system for further processing
//	//  * \param[out] Model Segments in the form |inliers|ModelCoefficients|Indices
//	//  * \return void 
//	//  */
//	//void _extractSegments(std::vector<SegmentationResultPtr> &outSegments);
//
//	///** \brief Do the conversion by inspecting the file extension
//	//  * \param[in] true if the output should be printed to a file els retain the output in the system for further processing
//	//  * \param[out] :ToDo: If needed specify the name of the output file
//	//  * \return void 
//	//  */
//	//virtual void ExecuteSegmentation() = 0;
//
//private:
//
//	/** \brief Get all the edges of the cloud using the curvature values
//	* \param[in] true = segment all models else chose only the best
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _GetCloudEdges(const PtCloudDataPtr &fInCloud, const unsigned int fTopnPercent, const int kNeigbouhood, PtCloudDataPtr &fEdgesCloud);
//
//
//	/** \brief Compute the average interpoint distance from an randomly sampled number of pnts
//	* \param[in] input cloud
//	* \param[in] number of sampling points considered
//	* \param[out] :ToDo: 
//	* \return mean distance 
//	*/
//	float _meanInterPntDistance(const PtCloudDataPtr &fInCloud, unsigned int fNumPntsConsidered);
//
//
//	/**\brief Get all strong lines on the curvature cloud
//	* \param[in] strong curvature cloud
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _GetAllLines(const PtCloudDataPtr &fInCloud, std::vector<VerticalEdgeLinePtr> &fAllLineOptimized, const int fNumIterations, const int fMinSupport);
//
//	/**\brief Ordinary least squares line fitting
//	* \param[in] 
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _LSLineFitting(const PtCloudDataPtr &fInClouds, Eigen::VectorXf &optimized_coefficients);
//
//	/**\brief Project points onto a line model
//	* \param[in] 
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _ProjectPointOnLine(const pcl::PointXYZ &pPoint, const Eigen::VectorXf fOptimLinesCoefs, pcl::PointXYZ &qPoint);
//
//	/**\brief Determin the intersection point of a line and a plane
//	* \param[in] 
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _PlaneLineIntersection(const Eigen::Vector4f &fOptimPlaneCoefs, const Eigen::VectorXf &fOptimLinesCoefs, pcl::PointXYZ &fPntOfIntersection);
//
//
//	/**\brief Determine the line of intersection of two planes using langranges multipliers
//	* \param[in] Coefs of plane A
//	* \param[in] Coefs of plane B
//	* \param[out] : Coefs of Line
//	* \return true if planes aren'nt parallel otherwise false. The point on the line is choosen as the closest to the mid-point of the centroid of both planes 
//	*/
//	bool _PlanePlaneIntersection1(const Eigen::Vector4f &PlaneA, const Eigen::Vector4f &PlaneB, Eigen::VectorXf &LineCoefs);
//
//	/**\brief filter roughly isolated intersection lines
//	* \param[in] ToDo
//	* \param[in] ToDo
//	* \param[out] : 
//	* \return true if planes aren'nt parallel otherwise false. The point on the line is choosen as the closest to the mid-point of the centroid of both planes 
//	*/
//	void _RoughlyValidateIntersection(const std::vector<fWallLinkagePtr> &fInLinkageMatrix, std::vector<fWallLinkagePtr> &fPreprocessedLinkageMatrix, const float &flengthOfIntersectionLine, PtCloudDataPtr &fOutinterPnts, unsigned int fCounterMin = 20);
//
//	/**\brief filter all planar segments between any two VP lines
//	* \param[in] ToDo
//	* \param[in] ToDo
//	* \param[out] : 
//	* \return true ToDo
//	*/
//	void _FilterConnectingSegments(const std::vector<fWallLinkagePtr> &fInLinkageMatrix, std::vector<EdgeConnectorPtr> &fConnectingEdges);
//
//
//	/**\brief Print the footprint and the connections of the various validated segments
//	* \param[in] ToDo
//	* \param[in] ToDo
//	* \param[out] : 
//	* \return ToDo
//	*/
//	void _Print2DProjections(const std::vector<fWallLinkagePtr> &fInLinkageMatrix);
//
//
//	/**\brief Parse all intersecting lines and convert the intersection points to graph edges by taking 2 points each
//	* and filtering all the inliers between these two points
//	* \param[in] ToDo
//	* \param[in] ToDo
//	* \param[out] : 
//	* \return ToDo
//	*/
//	void convertIntersectionPointsToGraphEdges(const std::vector<fWallLinkagePtr> &fInLinkageMatrix);
//
//
//
//	/**\brief Validate lines using thier mean and std
//	* \param[in] 
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _ValidateLines(const std::vector<VerticalEdgeLinePtr> &fLinesOptimized, std::vector<VerticalEdgeLinePtr> &ffLinesOptimizedValidated, float fStdMult);
//
//
//	void _ValidateLines(const std::vector<VerticalEdgeLinePtr> &fLinesOptimized, const std::vector<fWallLinkagePtr> &fInLinkageMatrix, std::vector<fWallLinkagePtr> &fOutLinkageMatrix, double fConnectionTolerrance);
//
//   /**\brief validate house model using BIC/GRIC/AIC/GIC
//	* \param[in] 
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	bool _ValidateSegments(const std::vector<EdgeConnectorPtr> &fConnectingEdges, std::vector<EdgeConnectorPtr> &fValidWalls);
//
//
//	/**\brief Determine the intersection of two planes using langranges multipliers
//	* \param[in] inliers of the plane
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	bool _PlanePlaneIntersection(const PtCloudDataPtr &PlaneA, const PtCloudDataPtr &PlaneB, Eigen::VectorXf &LineParams);
//
//	bool _PlanePlaneIntersection2(const Eigen::Vector4f &fPlaneA, const Eigen::Vector4f &fPlaneB, Eigen::VectorXf &fLineCoefs);
//
//	/**\brief Do ordinary least square plane fitting
//	* \param[in] inliers of the plane
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _LSPlaneFitting(const PtCloudDataPtr &fInClouds, Eigen::Vector4f &optimized_coefficients);
//
//	/**\brief Do ordinary least square plane fitting
//	* \param[in] inliers of the plane
//	* \param[out] :ToDo: 
//	* \return void 
//	*/
//	void _InterpreteHull(const std::vector<fWallLinkagePtr> &fInLinkageMatrix, PtCloudDataPtr &foutHullPnts);
//
//
//
//	void _DeterminCentroidPoints(const Vec3dPtr &fVPOptimized, const std::vector<VerticalEdgeLinePtr> &fOptimLinesCoefs, PtCloudDataPtr &fIntersectionCloud);
//
//	//void _PlaneLineIntersection(const Eigen::VectorXf &fOptimPlaneCoefs, const Eigen::VectorXf &fOptimLinesCoefs, pcl::PointXYZ &fPntOfIntersection);
//
//	void _DetermineQuadPoints(const std::vector<VerticalEdgeLinePtr> &fVPBestLines, const Vec3dPtr &fVPOptimized, PtCloudDataPtr &fCloudTopPoints, PtCloudDataPtr &fCloudBottomPoints);
//
//	void _GetNBestVerticalLines(const std::vector<VerticalEdgeLinePtr> &fVPLineOptimizedInliers, std::vector<VerticalEdgeLinePtr> &fVPBestNLines, unsigned int fTopNPercent);
//
//	void _FCloudHull(const PtCloudDataPtr &fPlaneCloud, PtCloudDataPtr &fHullCloud);
//
//
//	float _MaxWideOfCloud(const PtCloudDataPtr &fLineCloud, std::pair<pcl::PointXYZ, pcl::PointXYZ> &fLineEndingPnts, PtCloudDataPtr &fCloudProjectedOnLine);
//
//	float _MaxWideOfCloud(const PtCloudDataPtr &fLineCloud);
//
//	void _GetLinesParallelToVector(const std::vector<VerticalEdgeLinePtr> &fAllLineOptimized, std::vector<VerticalEdgeLinePtr> &fAllLinesParallelToVector, pcl::PointXYZ &fDirectionVector, float fAngularTolerance = 2.0);
//
//	void AddQuadActorToRenderer(const PtCloudDataPtr &fQuadEdges, vtkSmartPointer<vtkActor> &fQuadActor);
//
//	void AddCloudActorToRenderer(const PtCloudDataPtr &fCloud, vtkSmartPointer<vtkActor> &fCloudActor);
//
//	void _RobustlyEstimateVP(PtCloudDataPtr &voxelgridFilteredCloud);
//	Eigen::VectorXf _m_computedVP;
//
//	void _RobustlySegmentaPlanes();
//
//	void testCongress();
//
//	std::vector<SegmentationResultPtr> _m_outPlaneSegments;
//
//	PolygonModelParamsPtr m_ModelParams;
//	std::vector<Eigen::Vector4f> m_HousModelWalls;
//	PtCloudDataPtr m_InputCloud;
//	std::string m_InputFile;
//
//	/*bool m_SegmentAll;*/
//	/*unsigned int m_MaxNumModels;*/
//	/*unsigned int m_MinModelInliers;*/
//
//};
//
//#endif// _HOUSE_MODEL_BUILDER_H_