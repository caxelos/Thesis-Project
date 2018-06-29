#define NUM_OF_TREES 514
#define MAX_SAMPLES_PER_TREE 1000
#define MAX_GRP_SIZE 500
#define HEIGHT 9
#define WIDTH 15
#define LEFT 1
#define RIGHT 2

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

struct tree {
   double mean;	
   struct tree *left;
   struct tree *right;
   unsigned short thres;
};
typedef struct tree treeT;


/**************** FUNCTIONS *************************************/

void print_dims(int rank,  hsize_t *dims)  {
   cout << "dims are: ";
   for (int i = 0; i < rank; i++)  {
      cout  << dims[i] << ", ";
   }
   cout << "\n";

   return ;
}

//treeT **buildRegressionTree(unsigned int numOfGrps, unsigned int *fatherSize, unsigned char treeImgs[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][HEIGHT][WIDTH], double treeGazes[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2], double treePoses[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2]);
treeT **buildRegressionTree(unsigned int numOfGrps, unsigned int *fatherSize, unsigned char treeImgs[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][HEIGHT][WIDTH], double **treeGazes, double treePoses[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2]); 


/*************** MAIN ********************************************/
int main(void)  {
  const int R = 10;

 


  /*
   * data
   */
   int curr_nearest[13];
   double curr_center[2];
   double curr_gazes[MAX_GRP_SIZE][2];
   double curr_poses[MAX_GRP_SIZE][2];
   unsigned char curr_imgs[MAX_GRP_SIZE][HEIGHT][WIDTH];
   unsigned int *samplesInTree = NULL;

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
   unsigned int numOfGrps, grpContribution;
   unsigned int i,j;
   unsigned randNum;


  /*
   * tree-data
   */
   //unsigned char treeImgs[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][HEIGHT][WIDTH];
   unsigned char **treeImgs;
   double **treeGazes;
   double **treePoses;
   treeImgs = (unsigned char **)malloc( NUM_OF_TREES * sizeof(unsigned char *) );
   if (treeImgs == NULL)  {
      cout << "Error allocating memory\n";
      return -1;
   }
  
   treeGazes = (double **)malloc( NUM_OF_TREES * sizeof(double*) ); //double treeGazes[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2];
   if (treeGazes == NULL)  {
      cout << "Error allocating memory\n";
      return -1;
   }

   treePoses = (double **)malloc( NUM_OF_TREES * sizeof(double*) );//double treePoses[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2]; 
   if (treePoses == NULL)  {
      cout << "Error allocating memory\n";
      return -1;
   }


   treeT **trees;
   
   
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
   
      group = new Group(file->openGroup("/"));
      numOfGrps = group->getNumObjs();
      delete group;
 

      samplesInTree = (unsigned int *)calloc( numOfGrps , sizeof(int) );
      if (samplesInTree == NULL) {
         cout << "Error allocating NULL memory. Terminating\n";
	 return -1;
      }



      //for every group
      for (i = 0; i < numOfGrps; i++ )  {
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
	 treeGazes[i] = (double *)malloc( 2 * grpContribution * sizeof(double) );
	 if (treeGazes[i] == NULL)  {
	    cout << "Error allocating sqrt doubles. Exiting\n";
            return -1;  
	 }
	 treePoses[i] = (double *)malloc( 2 * grpContribution * sizeof(double) );
	 if (treePoses[i] == NULL)  {
	    cout << "Error allocating sqrt doubles. Exiting\n";
            return -1;  
	 }
	 treeImgs[i] = (unsigned char *)malloc( (WIDTH*HEIGHT*grpContribution) * sizeof( unsigned char ) );
	 if (treeImgs[i] == NULL)  {
	    cout << "Error allocating sqrt doubles. Exiting\n";
            return -1;    
	 }
	 /*
	  * main Group
	  */
	 for (j = 0; j < grpContribution; j++)  {
	    
	    std::uniform_int_distribution<> distr(0, dims[0]-1); // range
	    randNum = distr(eng);

	    //copy img
	    for (unsigned int k = 0; k < HEIGHT; k++)  {
               for (unsigned int l = 0; l < WIDTH; l++)   {
	          treeImgs[i][ samplesInTree[i]*WIDTH*HEIGHT + k*WIDTH+l ] = curr_imgs[randNum][k][l];
               }
	    } 

	    //copy gaze
	    //treeGazes[i][samplesInTree[i]][0] = curr_gazes[randNum][0];
	    //treeGazes[i][samplesInTree[i]][1] = curr_gazes[randNum][1];
	    treeGazes[i][2*samplesInTree[i]] = curr_gazes[randNum][0];
	    treeGazes[i][2*samplesInTree[i]+1 ] = curr_gazes[randNum][1];

	    //copy pose
	    treePoses[i][2*samplesInTree[i]] = curr_poses[randNum][0];
	    treePoses[i][2*samplesInTree[i]+1] = curr_poses[randNum][0];


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
            if (grpContribution != 0) { 

	       treeGazes[i] = (double *)realloc(treeGazes[i],  (2 * (samplesInTree[i] + grpContribution) )*sizeof(double) );
	       if (treeGazes[i] == NULL)  {
	          cout << "Error allocating sqrt doubles(2). Exiting\n";
                  return -1;  
	       }
	       treeImgs[i] = (unsigned char *)realloc(treeImgs[i], WIDTH*HEIGHT*(samplesInTree[i]+grpContribution) * sizeof( unsigned char ) );
	       if (treeImgs[i] == NULL)  {
	          cout << "Error allocating sqrt doubles(2). Exiting\n";
                  return -1;  
	       }
	       treePoses[i] = (double *)realloc(treePoses[i],  (2 * (samplesInTree[i] + grpContribution) )*sizeof(double) );
	       if (treePoses[i] == NULL)  {
	          cout << "Error allocating sqrt doubles(2). Exiting\n";
                  return -1;  
	       }
	    }

	    for (j= 0; j < grpContribution; j++)  {
	       std::uniform_int_distribution<> distr(0, dims[0]-1); // range
	       randNum = distr(eng);

	       //copy img
	       for (int k = 0; k < HEIGHT; k++)  {
                  for (int l = 0; l < WIDTH; l++)   {
		     treeImgs[i][ samplesInTree[i]*WIDTH*HEIGHT + k*WIDTH+l ] = curr_imgs[randNum][k][l];
                  }
	       } 

	       //copy gaze
	       //treeGazes[i][samplesInTree[i]][0] = curr_gazes[randNum][0];
	       //treeGazes[i][samplesInTree[i]][1] = curr_gazes[randNum][1];
	       treeGazes[i][2*samplesInTree[i]   ] = curr_gazes[randNum][0];
	       treeGazes[i][2*samplesInTree[i]+1 ] = curr_gazes[randNum][1];


	       //copy pose
	       treePoses[i][2*samplesInTree[i]   ] = curr_poses[randNum][0];
	       treePoses[i][2*samplesInTree[i]+1 ] = curr_poses[randNum][0];


	       samplesInTree[i]++;
	     }

	    dataspace.close();
            dataset.close();
            memspace.close();
	    delete group_nearest;

	  }//for r            
      }//for i

     // trees = buildRegressionTree( numOfGrps, samplesInTree, treeImgs, treeGazes, treePoses);
      //free(trees);  

    }//try 
    catch(  FileIException error)  {
       error.printErrorStack();    
       return -1;
     }
    

   for (i = 0; i < NUM_OF_TREES; i++)  {
      free( treeGazes[i] );
      free( treeImgs[i]  );
      free( treePoses[i] );
   }
   free( treeGazes );
   free( treeImgs  );
   free( treePoses );
       
   return 0;
}


treeT **buildRegressionTree(unsigned int numOfGrps, unsigned int *fatherSize, unsigned char treeImgs[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][HEIGHT][WIDTH], double **treeGazes, double treePoses[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2])  {   
//treeT **buildRegressionTree(unsigned int numOfGrps, unsigned int *fatherSize, unsigned char treeImgs[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][HEIGHT][WIDTH], double treeGazes[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2], double treePoses[NUM_OF_TREES][MAX_SAMPLES_PER_TREE][2])  {
 
   treeT **trees = NULL;   

   unsigned int i,j;//,stackindex, state, node_i;
  
   cout << "numOfGrps is " << numOfGrps << endl; 
   trees = (treeT **)malloc( numOfGrps * sizeof(treeT *) );
   for (i = 0; i < numOfGrps; i++)  {
      trees[i] = (treeT *)malloc( sizeof(treeT) );
   }

   if (trees == NULL)  {
     cout << "Error allocating tree memory. Exiting\n" << endl;
     return NULL;
  }
   	
  
 
   for (i = 0; i < numOfGrps; i++ )  {
    

     // stackindex = 0;
     // state = 1;
     // node_i = 1;
      //trees[i] = (treeT *)malloc( sizeof(treeT) );
   //   for (j = 0; j < /*fatherSize[i]*/5; j++)  {
        // currPtrs[j] = j;//initialization	
   //   }
         

   }//end for i
/* 


   for (i = 0; i < numOfGrps; i++) {
     // free(trees[i]);
   }
   
*/
  
// free(trees);

   return NULL;
}




