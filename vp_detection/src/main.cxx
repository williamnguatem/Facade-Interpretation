#include <VanishingPointDetection/VPDetection.h>
#include <VanishingPointDetection/VPDetectionCloud.h>
#include <VanishingPointDetection/VPDetectionContext.h>
#include <boost/algorithm/string.hpp>
//#include <VanishingPointDetection/VPDetection.h>
#include <iostream>
#include <fstream>


//without streaming
void load_cloud(const std::string &cloud_file_data, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<unsigned int> &mapping_table, const std::string &save_to_file)
{
	//std::string sPath(cloud_file_data);
	//InputParser sData;
	//char sVertex[2];
	float xTemp = 0, yTemp = 0, zTemp = 0;
	int sNumTemp = 0;
	//Vec3d sVec3Temp;
	FILE *sFile;
	//sFile = fopen(IN_3DPOINTS,"r");
	sFile = fopen(cloud_file_data.c_str(),"r");

	std::vector<std::string> strs;

	unsigned int num_lines = 0;
	std::string line;
	std::ifstream myfile (cloud_file_data);
	if (myfile.is_open())
	{
		while ( myfile.good() )
		{
			getline (myfile,line);
			//std::cout << line << std::endl;
			num_lines++;
		}
		myfile.close();
	}

	else std::cout << "Unable to open file"; 


	//creat cloud with known size
	 pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	in_cloud->height = 1;
	in_cloud->width = num_lines - 2; //the first line and the last line are removed. see file 3D2Dtable format
	in_cloud->resize(in_cloud->height * in_cloud->width);

	num_lines = -1;
	std::string aline;
	std::ifstream amyfile (cloud_file_data);
	if (amyfile.is_open())
	{
		while ( amyfile.good() )
		{
			getline (amyfile,line);
			//std::cout << line << std::endl;
			/*scanf("%f\n", &xTemp, line.c_str());*/
			if (num_lines !=-1 && num_lines < in_cloud->points.size())
			{
				boost::split(strs, line, boost::is_any_of("\t "));
				xTemp = in_cloud->points[num_lines].x = atof(strs.at(0).c_str());
				yTemp = in_cloud->points[num_lines].y = atof(strs.at(1).c_str());
				zTemp = in_cloud->points[num_lines].z = atof(strs.at(2).c_str());
			}
			num_lines++;
		}
		amyfile.close();
	}

	else std::cout << "Unable to open file";




	//PtCloudDataPtr fConvertedData(new PtCloudData());
	//fConvertedData->height = 1;
	//fConvertedData->width = sNumTemp;
	//fConvertedData->resize(fConvertedData->height * fConvertedData->width);

	//sFile = fopen(cloud_file_data.c_str(),"r");
	//sNumTemp = 0;
	////mapping_table.resize(sNumTemp); //just in case

	//unsigned int img_indices;
	//while(fscanf(sFile,"%f %f %f %d\n",&xTemp, &yTemp, &zTemp, &img_indices) !=EOF){

	//	//ignore header infos
	//	//if(sNumTemp == 0)
	//	//	continue;

	//	fConvertedData->points[sNumTemp].x = xTemp;
	//	fConvertedData->points[sNumTemp].y = yTemp;
	//	fConvertedData->points[sNumTemp].z = zTemp;
	//	mapping_table.push_back(img_indices);

	//	if (sNumTemp % 15000 == 0)
	//	{
	//		std::cout<<"Converted Line "<<sNumTemp<<std::endl; 
	//	}
	//	sNumTemp++;
	//}





	//while(fscanf(sFile,"%f %f %f\n", &xTemp, &yTemp, &zTemp ) !=EOF){
	//	////while(fscanf(sFile,"%f %f %f\n",&xTemp, &yTemp, &zTemp) !=EOF){
	//	//sVec3Temp.x =xTemp;
	//	//sVec3Temp.y =yTemp;
	//	//sVec3Temp.z =zTemp;
	//	//CSaveToFile(sVec3Temp, SaveToFile.c_str());
	//	////		sInData.AddElement(sVec3Temp);
	//	//std::cout<<xTemp<<std::endl;
	//	//std::cout<<yTemp<<std::endl;
	//	//std::cout<<zTemp<<std::endl;
	//	if (sNumTemp%50000 == 0)
	//	{
	//		std::cout<<"Read Line "<<sNumTemp<<std::endl; 
	//	}
	//	sNumTemp++;
	//}

	//fclose(sFile);

	//PtCloudDataPtr fConvertedData(new PtCloudData());
	//fConvertedData->height = 1;
	//fConvertedData->width = sNumTemp;
	//fConvertedData->resize(fConvertedData->height * fConvertedData->width);

	//sFile = fopen(cloud_file_data.c_str(),"r");
	//sNumTemp = 0;
	////mapping_table.resize(sNumTemp); //just in case

	//unsigned int img_indices;
	//while(fscanf(sFile,"%f %f %f %d\n",&xTemp, &yTemp, &zTemp, &img_indices) !=EOF){

	//	//ignore header infos
	//	//if(sNumTemp == 0)
	//	//	continue;

	//	fConvertedData->points[sNumTemp].x = xTemp;
	//	fConvertedData->points[sNumTemp].y = yTemp;
	//	fConvertedData->points[sNumTemp].z = zTemp;
	//	mapping_table.push_back(img_indices);

	//	if (sNumTemp % 15000 == 0)
	//	{
	//		std::cout<<"Converted Line "<<sNumTemp<<std::endl; 
	//	}
	//	sNumTemp++;
	//}

	//save to disc als binary
	std::string fFilename(save_to_file);
	pcl::PCDWriter writer;
	writer.writeASCII(fFilename, *in_cloud);

//	fFilename = "Haus51.pcd";
//	writer.write(fFilename, *fConvertedData, true);
	std::cout<<"Number of Lines scanned "<<sNumTemp<<std::endl; 
}




void get_cloud_vp(const  pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud_ptr )
{
	Eigen::VectorXf fResults;
	//Extract Model Parameters from GUI Elements
	VPModelParamsPtr fModelParams(new VPModelParams);
	fModelParams->fNumOfIterations = 2000;
	// collect Voxel grid params
//	//ToDo: Check to see which controller is active before collecting voxel grid value
	fModelParams->fVoxGridSize(0) = 0.001; //doubleSpinBox_8->value();
	fModelParams->fVoxGridSize(1) = 0.001; //doubleSpinBox_9->value();
	fModelParams->fVoxGridSize(2) = 0.001; //doubleSpinBox_10->value();
	// collect Spatial partitioning params, default use KD-tree
	fModelParams->fRadiusSearched = 0.0f; //doubleSpinBox_3->value();//
	fModelParams->m_k = 10; //spinBox_5->value();
	//collect EPS
	fModelParams->fEPSInDegrees = 3;
	//collect percentage of the number of clouds to compute normals
	fModelParams->fPercentageOfNormals = 40;
	//collect the number of cross products
	fModelParams->fNumOfCrossProducts = 1000;
	//always validate results
	fModelParams->paramValidateResults = true;
	fModelParams->fNumberOfValidationRounds = 20;
	
	//Do the actual work here
	VPDetectionCloud *vp = new VPDetectionCloud;
	VPDetectionContext vpContext(vp);
	vpContext.setInputCloud(input_cloud_ptr);
	vpContext.setModelParams(fModelParams);
	vpContext.validateResults(fModelParams->paramValidateResults);
	vpContext.ComputeVPDetection();
	vpContext.extractVPs(fResults);
	std::cout<<fResults<<std::endl;
	delete vp;
}



int main(int argc, char** argv)
{

  //if (argc < 2)
  //{
  //  throw std::runtime_error ("Required arguments: filename.pcd");
  //}

  //std::string fileName = argv[1];
  std::string fileName = "Haus51.pcd";
  std::cout << "Reading " << fileName << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (fileName, *cloud) == -1) // load the file
  {
    PCL_ERROR ("Couldn't read file");
    return (-1);
  }

  std::cout << "Loaded " << cloud->points.size () << " points." << std::endl;


  //Load and parse the cloud
  const std::string get_from_file("Bauhaus-3D2Dtable.txt"/*"G:\\DataSet\\Images\\Haus51_multi_adv\\Result\\p10-3D2Dtable.txt"*/);
  std::vector<unsigned int> mapping_table;
  std::string save_to_file("haus512.pcd");
  
  load_cloud(get_from_file, cloud, mapping_table, save_to_file);

  get_cloud_vp(cloud);
//std::cout"Tue nix"<<std::endl;

//get the input data
//test_vp_cloud()

return 0;
}
