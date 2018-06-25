#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;

#include <string>
#include "H5Cpp.h"
using namespace H5;

const H5std_string FILE_NAME( "myfile.h5" );
const int NX = 1;
const int NY = 13;

int main(void)  {
   int curr_nearest[13];
   float center[2];

   int rank;
   hsize_t     dims[4]; /* memory space dimensions */

   H5File *file = NULL;
   Group *group = NULL;

  
   /*
    * Try block to detect exceptions raised
    */   
    try {
    
       /*
       * Turn off the auto-printing when failure occurs so that we can
       * handle the errors appropriately
       */
      Exception::dontPrint();
      
      file = new H5File(FILE_NAME, H5F_ACC_RDWR);
      group = new Group(file->openGroup("g1") );   

 
     /*
      * 12_nearestIDS
      */
      DataSet dataset = group->openDataSet("12_nearestIDs");     
      DataSpace dataspace = dataset.getSpace();//dataspace???
      rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
      DataSpace memspace( rank, dims );     
      dataset.read(curr_nearest, PredType::NATIVE_INT, memspace, dataspace ); 
    


     /*
      * center
      */
      dataset = group->openDataSet("center");     
      dataspace = dataset.getSpace();//dataspace???
      rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
      memspace.setExtentSimple( rank, dims );
      dataset.read(center, PredType::NATIVE_FLOAT, memspace, dataspace ); 
   

     /*
      * gaze
      */
      dataset = group->openDataSet("gaze");     
      dataspace = dataset.getSpace();//dataspace???
      rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
      memspace.setExtentSimple( rank, dims );


     /*
      * headpose
      */
      dataset = group->openDataSet("headpose");     
      dataspace = dataset.getSpace();//dataspace???
      rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
      memspace.setExtentSimple( rank, dims );


      /*
      * data
      */
      dataset = group->openDataSet("data");     
      dataspace = dataset.getSpace();//dataspace???
      rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
      memspace.setExtentSimple( rank, dims );//24x1x9x15






     /*
      * Close the group and file
      */         
      delete group;
      delete file;
       
    } 
    catch(  FileIException error)  {
       error.printErrorStack();    
       return -1;
    }
    
    
    


   return 0;
}

/*
#include<iostream>
#include<vector>
#include<H5Cpp.h>


const std::string& filename = "mydata.h5";

std::vector<double> load( const std::string& dataset) {
  
  H5::DataSet dset = fp.openDataSet(dataset.c_str());
  H5::DataSpace dspace = dset.getSpace();
  hsize_t rank;
  hsize_t dims[2];  
  rank = dspace.getSimpleExtentDims(dims, nullptr);
  std::vector<double> data;
  data.resize(dims[0]);
  dset.read(data.data(), H5::PredType::NATIVE_DOUBLE, dspace);
  
  return data;
}

int main() {

  H5::H5File fp(filename.c_str(), H5F_ACC_RDONLY);
  std::vector<double> data = {1, 2};

  auto data_read = load(fp, "my dataset");
  for(auto item: data_read) {
    std::cout<<item<<" ";
  }
  std::cout<<std::endl;
  fp.close();


  return 0;
}
*/
