#define MAX_GRP_SIZE 500
#define MAX_TREES 1000
#define HEIGHT 9
#define WIDTH 15


#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;

// C,C++ Libraries 
#include <string>
#include <stdlib.h>
#include <random>
#include <math.h>
#include "H5Cpp.h"

using namespace H5;

const H5std_string FILE_NAME( "myfile.h5" );


void print_dims(int rank,  hsize_t *dims)  {
   cout << "dims are: ";
   for (int i = 0; i < rank; i++)  {
      cout  << dims[i] << ", ";
   }
   cout << "\n";

   return ;
}

int main(void)  {
  const int R = 10;
  /*

  /*
   * tree-data
   */
   unsigned char treeImgs[MAX_TREES][HEIGHT][WIDTH][1000];
   double treeGazes[MAX_TREES][1000][2];
   double treePoses[MAX_TREES][1000][2];///////////


  /*
   * data
   */
   int curr_nearest[13];
   double curr_center[2];
   double curr_gazes[MAX_GRP_SIZE][2];
   double curr_poses[MAX_GRP_SIZE][2];
   unsigned char curr_imgs[MAX_GRP_SIZE][HEIGHT][WIDTH];
   int *samplesInTree = NULL;

  /*
   * hdf5 staff
   */
   int rank;
   hsize_t     dims[4]; /* memory space dimensions */
   H5File *file = NULL;
   Group *group = NULL;
   Group *group_nearest = NULL;
  /*
   * temp staff
   */
   char grpName[10]; 
   int numOfGrps, grpContribution;
   int i,j;
   int randNum;
   
   
   //define randomization   
   std::random_device rd; // obtain a random number from hardware
   std::mt19937 eng(rd()); // seed the generator


  

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
      //group = new Group(file->openGroup("g1") );
   
      group = new Group(file->openGroup("/"));
      numOfGrps = group->getNumObjs();
      delete group;
 

      samplesInTree = (int *)calloc( numOfGrps , sizeof(int) );
      if (samplesInTree == NULL) {
         cout << "Error allocating NULL memory. Terminating\n";
	 return -1;
      }



      //for every group
      for (int i = 0; i < numOfGrps; i++ )  {
	sprintf(grpName, "g%d", i+1); 
        group = new Group(file->openGroup( grpName ) );
        
        /*
         * 12_nearestIDS
         */ 
         DataSet dataset = group->openDataSet("nearestIDs");     
         DataSpace dataspace = dataset.getSpace();//dataspace???
         rank = dataspace.getSimpleExtentDims( dims );// get rank =    numOfDims
         DataSpace memspace( rank, dims );     
         dataset.read(curr_nearest, PredType::NATIVE_INT, memspace, dataspace ); 
           
        /*
         * center
         */
         dataset = group->openDataSet("center");     
         dataspace = dataset.getSpace();//dataspace???
         rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
         memspace.setExtentSimple( rank, dims );
         dataset.read(curr_center, PredType::NATIVE_DOUBLE, memspace, dataspace ); 
   
        /*
         * gaze
         */
         dataset = group->openDataSet("gaze");     
         dataspace = dataset.getSpace();//dataspace???
         rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
         memspace.setExtentSimple( rank, dims );
         dataset.read(curr_gazes, PredType::NATIVE_DOUBLE, memspace, dataspace ); 
	 
        /*
         * headpose
         */
         dataset = group->openDataSet("headpose");     
         dataspace = dataset.getSpace();//dataspace???
         rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
         memspace.setExtentSimple( rank, dims );
         dataset.read(curr_poses, PredType::NATIVE_DOUBLE, memspace, dataspace ); 

        /*
         * data
         */
         dataset = group->openDataSet("data");     
         dataspace = dataset.getSpace();//dataspace???
         rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
         memspace.setExtentSimple( rank, dims );//24x1x9x15 
         //print_dims(rank, (hsize_t *)dims);     
         dataset.read(curr_imgs, PredType::C_S1, memspace, dataspace );


	
         grpContribution = sqrt( dims[0]);//dims[0] is the numOfSamples in group1

	 /*
	  * main Group
	  */
	 for (j = 0; j < grpContribution; j++)  {
	    
	    std::uniform_int_distribution<> distr(0, dims[0]-1); // range
	    randNum = distr(eng);

	    //copy img
	    for (int k = 0; k < HEIGHT; k++)  {
               for (int l = 0; l < WIDTH; l++)   {
	          treeImgs[i][k][l][ samplesInTree[i] ] = curr_imgs[randNum][k][l];
               }
	    } 

	    //copy gaze
	    treeGazes[i][samplesInTree[i]][0] = curr_gazes[randNum][0];
	    treeGazes[i][samplesInTree[i]][1] = curr_gazes[randNum][1];


	    //copy pose
	    treePoses[i][samplesInTree[i]][0] = curr_poses[randNum][0];
	    treePoses[i][samplesInTree[i]][1] = curr_poses[randNum][0];


	    samplesInTree[i]++;
	 }

	 dataspace.close();
         dataset.close();
         memspace.close();
	 delete group;


	 /*
	  * R-nearest
	  */
	 for (int r = 0; r < R; r++)  {
	   
	    sprintf(grpName, "g%d", curr_nearest[r] ); 
            group_nearest = new Group(file->openGroup( grpName ) );

	   /*
            * gaze
            */
            DataSet dataset = group->openDataSet("gaze");  
            DataSpace dataspace = dataset.getSpace();//dataspace???
            rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
            DataSpace memspace( rank, dims );
            dataset.read(curr_gazes, PredType::NATIVE_DOUBLE, memspace, dataspace );   
	  
           /*
            * headpose
            */
            dataset = group->openDataSet("headpose");     
            dataspace = dataset.getSpace();//dataspace???
            rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
            memspace.setExtentSimple( rank, dims );
            dataset.read(curr_poses, PredType::NATIVE_DOUBLE, memspace, dataspace ); 

           /*
            * data
            */
            dataset = group->openDataSet("data");     
            dataspace = dataset.getSpace();//dataspace???
            rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
            memspace.setExtentSimple( rank, dims );//24x1x9x15 
            dataset.read(curr_imgs, PredType::C_S1, memspace, dataspace );

	    grpContribution = sqrt( dims[0]);//dims[0] is the numOfSamples in group1
	    for (j= 0; j < grpContribution; j++)  {
	       std::uniform_int_distribution<> distr(0, dims[0]-1); // range
	       randNum = distr(eng);

	       //copy img
	       for (int k = 0; k < HEIGHT; k++)  {
                  for (int l = 0; l < WIDTH; l++)   {
	             treeImgs[i][k][l][ samplesInTree[i] ] = curr_imgs[randNum][k][l];
                  }
	       } 

	       //copy gaze
	       treeGazes[i][samplesInTree[i]][0] = curr_gazes[randNum][0];
	       treeGazes[i][samplesInTree[i]][1] = curr_gazes[randNum][1];


	       //copy pose
	       treePoses[i][samplesInTree[i]][0] = curr_poses[randNum][0];
	       treePoses[i][samplesInTree[i]][1] = curr_poses[randNum][0];


	       samplesInTree[i]++;
	     }

	    dataspace.close();
            dataset.close();
            memspace.close();
	    delete group_nearest;

	 }        
	cout << "Samples(" << i << ") = " << samplesInTree[i] << "\n";
        
      }//for i


    }//try 
    catch(  FileIException error)  {
       error.printErrorStack();    
       return -1;
     }
    
   
    


   return 0;
}
