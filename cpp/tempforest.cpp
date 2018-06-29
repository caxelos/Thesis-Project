
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


treeT **buildRegressionTree(unsigned int *fatherSize,unsigned char **treeImgs,double **treeGazes,double**treePoses);


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
   unsigned int grpContribution;
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
   
      samplesInTree = (unsigned int *)calloc( NUM_OF_TREES , sizeof(int) );
      if (samplesInTree == NULL) {
         cout << "Error allocating NULL memory. Terminating\n";
	 return -1;
      }



      //for every group
      for (i = 0; i < NUM_OF_TREES; i++ )  {
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

      trees = buildRegressionTree(samplesInTree, treeImgs, treeGazes, treePoses);

    }//try 
    catch(  FileIException error)  {
       error.printErrorStack();    
       return -1;
     }
    

   for (i = 0; i < NUM_OF_TREES; i++)  {
      free( treeGazes[i] );
      free( treeImgs[i]  );
      free( treePoses[i] );
      free( trees[i]     );
      
   }
   free( treeGazes );
   free( treeImgs  );
   free( treePoses );
   free( trees     );    
   return 0;
}



/*
 * falloc = forest allocation
 */
tree **falloc(void)   {

   treeT **trees = (treeT **)malloc( NUM_OF_TREES * sizeof(treeT *) );
   if (trees == NULL)  {
     cout << "Error allocating tree memory at falloc(1). Exiting\n" << endl;
     return NULL;
   }

   for (unsigned i = 0; i < NUM_OF_TREES; i++)  {
      trees[i] = (treeT *)malloc( sizeof(treeT) );
      if (trees[i] == NULL)  {
         cout << "Error allocating tree memory at falloc(1). Exiting\n" << endl;
         return NULL;
      }
   }
   
   return trees;
}

treeT **buildRegressionTree(unsigned int *fatherSize,unsigned char **treeImgs,double **treeGazes,double**treePoses) {
   
   treeT **trees = NULL; 
   treeT *currNode = NULL;
   unsigned int *currPtrs = NULL;  
   unsigned int *l_r_fl_fr_ptrs = NULL;
   unsigned int i,j,l,r,stackindex, state, node_i;
   unsigned int minSquareError;
   unsigned short minPx1_vert, minPx1_hor, minPx2_vert, minPx2_hor, bestThres;
   unsigned short px1_hor, px1_vert, px2_hor, px2_vert, thres;
   unsigned int counter;
   double meanLeftGaze[2];
   double meanRightGaze[2]; 
  /*
   * caching big arrays
   */
   unsigned char *cache_treeImgs; 
   

  /*
   * allocate **trees memory
   */
   trees = falloc();
   if (trees == NULL) {
      return NULL;
   }
 
   	
   
   for (i = 0; i < NUM_OF_TREES; i++ )  {
      stackindex = 0;
      state = 1;
      node_i = 1;
      currNode = trees[i];


      currPtrs = (unsigned int *)malloc(fatherSize[i] * sizeof( unsigned int) );
      if (currPtrs == NULL)  {
         cout << "Error allocating \"currPtrs\". Exiting\n";
         return NULL;
      }
      cache_treeImgs = (unsigned char *)malloc( fatherSize[i]*2*sizeof(unsigned char) );
      if (cache_treeImgs == NULL) {
         cout << "error allocating memory for caching. Exiting\n"; 
         return NULL;
      } 
	
      
      l_r_fl_fr_ptrs = (unsigned int *)malloc( 4*fatherSize[i]*sizeof(unsigned int) ); 
      if (l_r_fl_fr_ptrs == NULL) {
         cout << "error allocating memory for ptrs2. Exiting\n";
	 return NULL;
      }


      while (state != 2) {
         minSquareError = 10000;//a huge value
	 minPx1_vert =    10000;//again the same
	 minPx1_hor =     10000;//also here
	 minPx2_vert=     10000;//and here..
	 minPx2_hor =     10000;//and here 
	 bestThres  =     10000;//ah, and here
 
         counter = 0;//threadID here
	 while (counter < WIDTH*HEIGHT )  {
            px1_vert = counter/WIDTH;   
	    px1_hor = counter%WIDTH;

	    for (px2_vert=px1_vert+(px1_hor+1)/WIDTH; px2_vert<HEIGHT; px2_vert++)  {
               for (px2_hor=(px1_hor+1)%WIDTH; px2_hor < WIDTH; px2_hor++)  {
	          if  ( sqrt( pow(px1_vert -px2_vert,2) + pow(px1_hor-px2_hor,2) ) < 6.5 )  {  
		     

                     for (j = 0; j < fatherSize[i]; j++)  {
		        cache_treeImgs[2*j    ] = treeImgs[i][currPtrs[j]*WIDTH*HEIGHT + px1_vert*WIDTH + px1_hor];  //treeImgs(i, px1_vert,px1_hor, currPtrs( j)  );
	      	        cache_treeImgs[2*j + 1] = treeImgs[i][currPtrs[j]*WIDTH*HEIGHT + px2_vert*WIDTH + px2_hor];
                     }


		     for (thres = 25; thres <= 40/*50*/; thres++) {
			l = 0;
			r = 0;
			meanLeftGaze[0]  = 0;
			meanLeftGaze[1]  = 0;
			meanRightGaze[0] = 0;
			meanRightGaze[1] = 0;

			for (j = 0; j < fatherSize[i]; j++)  {
			   if ( abs(cache_treeImgs[j,0]-cache_treeImgs[j,1])< thres )  {

//4*fatherSize(i)				   


			      //left child
			      l_r_fl_fr_ptrs[0 + l] = currPtrs[j];
			      l++;

			      meanLeftGaze[0] = meanLeftGaze[0] + treeGazes[i][currPtrs[j]*2];
			      meanLeftGaze[1] = meanLeftGaze[1] + treeGazes[i][currPtrs[j]*2 + 1];			       
			   }
			   else {

			      //right child
			      l_r_fl_fr_ptrs[1*fatherSize[i]+r] = currPtrs[j];
  			      r++;	   
  
			      meanRightGaze[0] = meanRightGaze[0] + treeGazes[i][currPtrs[j]*2];
			      meanRightGaze[1] = meanRightGaze[1] + treeGazes[i][currPtrs[j]*2 + 1];			      
			   }
		        }
		     }
		  }
               }
            }
         }
      }
   }//end for i

   return trees;
}




