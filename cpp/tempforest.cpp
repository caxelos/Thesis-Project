#define PARALLEL
#define STDEV_CALC

#ifdef PARALLEL
   #define NUM_OF_THREADS 4
#endif


#define NUM_OF_TREES 690//690//940//
#define MAX_SAMPLES_PER_TREE 1000
#define MAX_RECURSION_DEPTH 15
#define MAX_GRP_SIZE 500


#define HEIGHT 9
#define WIDTH 15
#define LEFT 1
#define RIGHT 2

#define RADIUS 5


#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif

using std::cout;
using std::endl;
using std::ofstream; 
// C,C++ Libraries 
#include <string>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <math.h>
#include <unistd.h>
#include "H5Cpp.h"
#include <omp.h>
using namespace H5;

const H5std_string FILE_NAME( "myfile.h5" );

struct tree {
   double mean[2];
   double stdev[2];
   double mse;
   struct tree *left;
   struct tree *right;
   unsigned int *ptrs;
   unsigned int numOfPtrs;
   unsigned short thres;
   unsigned short minPx1_hor;
   unsigned short minPx2_hor;
   unsigned short minPx1_vert;
   unsigned short minPx2_vert;
};
typedef struct tree treeT;

ofstream outputFile;
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
treeT *testSampleInTree(treeT *currNode, unsigned char *test_img, double *test_pose, int j); 

/*************** MAIN ********************************************/
int main(int argc, char *argv[])  {

 
  /*
   * read hdf5 data
   */
   int curr_nearest[RADIUS];
   double curr_center[2];
   double *curr_gazes=NULL;
   double *curr_poses=NULL;//[MAX_GRP_SIZE][2];
   unsigned char *curr_imgs=NULL;//[MAX_GRP_SIZE][HEIGHT][WIDTH];
   unsigned int curr_size;
   unsigned int *samplesInTree = NULL;


   int *test_nearest;
   double *test_gazes;
   double *test_poses;//[MAX_GRP_SIZE][2];
   unsigned char *test_imgs;//[MAX_GRP_SIZE][HEIGHT][WIDTH];


   
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
      file = new H5File(argv[1], H5F_ACC_RDWR);
   
      samplesInTree = (unsigned int *)calloc( NUM_OF_TREES , sizeof(int) );
      if (samplesInTree == NULL) {
         cout << "Error allocating NULL memory. Terminating\n";
	 return -1;
      }


      //for every group
      curr_size = 0;
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
	 // update sizes if size per group is bigger
	 if (dims[0] > curr_size)  {

	    //reallocate gaze mem
            curr_gazes = (double *)realloc( curr_gazes, (dims[0]*sizeof(double))<<1 );
	    if (curr_gazes == NULL)  {
	       cout << "error at realloc, curr_gazes" << endl;
	       return (-1);
	    }

	    //reallocate pose mem
	    curr_poses = (double *)realloc( curr_poses, (dims[0]*sizeof(double))<<1 );
	    if (curr_poses == NULL)  {
	       cout << "error at realloc, curr_gazes" << endl;
	       return (-1);
	    }

	    //reallocate img mem
	    curr_imgs = (unsigned char *)realloc( curr_imgs, dims[0]*WIDTH*HEIGHT*sizeof(unsigned char) );
	    if (curr_imgs == NULL)  {
	       cout << "error at realloc, curr_gazes" << endl;
	       return (-1);
	    }
          
	    curr_size = dims[0];
         }

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
         dataset.read(curr_imgs, PredType::C_S1, memspace, dataspace );


          	
         grpContribution = sqrt( dims[0]);//dims[0] is the numOfSamples in group1

	 treeGazes[i] = (double *)malloc( (grpContribution * sizeof(double)) << 1 );
	 if (treeGazes[i] == NULL)  {
	    cout << "Error allocating sqrt doubles. Exiting\n";
            return -1;  
	 }
	 treePoses[i] = (double *)malloc( (grpContribution * sizeof(double)) << 1 );
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
	          treeImgs[i][ samplesInTree[i]*WIDTH*HEIGHT + k*WIDTH+l ] = curr_imgs[randNum*WIDTH*HEIGHT + k*WIDTH + l];//curr_imgs[randNum][k][l];
               }
	    } 

	    //copy gaze
	    treeGazes[i][samplesInTree[i]<<1] = curr_gazes[randNum<<1];//[randNum][0];
	    treeGazes[i][(samplesInTree[i]<<1)+1 ] = curr_gazes[(randNum<<1)+1];//[randNum][1];

	    //copy pose
	    treePoses[i][samplesInTree[i]<<1] = curr_poses[randNum<<1];//[randNum][0];
	    treePoses[i][(samplesInTree[i]<<1)+1] = curr_poses[(randNum<<1)+1];//[randNum][1];


	    samplesInTree[i]++;
	 }

	 dataspace.close();
         dataset.close();
         memspace.close();
	 delete group;


	 /*
	  * R-nearest
	  */
	 for (int r = 0; r < RADIUS; r++)  {
	   
	    sprintf(grpName, "g%d", curr_nearest[r] ); 
            group_nearest = new Group(file->openGroup( grpName ) );
	   /*
            * gaze
            */
            DataSet dataset = group->openDataSet("gaze");  
            DataSpace dataspace = dataset.getSpace();//dataspace???
            rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
	    // update sizes if size per group is bigger
	    if (dims[0] > curr_size)  {
	
	       //reallocate gaze mem
               curr_gazes = (double *)realloc( curr_gazes, (dims[0]*sizeof(double))<<1 );
	       if (curr_gazes == NULL)  {
	          cout << "error at realloc, curr_gazes" << endl;
	          return (-1);
	       }

	       //reallocate pose mem
	       curr_poses = (double *)realloc( curr_poses, (dims[0]*sizeof(double)<<1) );
	       if (curr_poses == NULL)  {
	          cout << "error at realloc, curr_gazes" << endl;
	          return (-1);
	       }

	       //reallocate img mem
	       curr_imgs = (unsigned char *)realloc( curr_imgs, dims[0]*WIDTH*HEIGHT*sizeof(unsigned char) );
	       if (curr_imgs == NULL)  {
	          cout << "error at realloc, curr_gazes" << endl;
	          return (-1);
	       }
          
	       curr_size = dims[0];
            }
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

	       treeGazes[i] = (double *)realloc(treeGazes[i],  ( ((samplesInTree[i] + grpContribution) )*sizeof(double))<<1 );
	       if (treeGazes[i] == NULL)  {
	          cout << "Error allocating sqrt doubles(2). Exiting\n";
                  return -1;  
	       }
	       treeImgs[i] = (unsigned char *)realloc(treeImgs[i], WIDTH*HEIGHT*(samplesInTree[i]+grpContribution) * sizeof( unsigned char ) );
	       if (treeImgs[i] == NULL)  {
	          cout << "Error allocating sqrt doubles(2). Exiting\n";
                  return -1;  
	       }
	       treePoses[i] = (double *)realloc(treePoses[i],  (  ((samplesInTree[i] + grpContribution) )*sizeof(double)<<1) );
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
		     treeImgs[i][ samplesInTree[i]*WIDTH*HEIGHT + k*WIDTH+l ] = curr_imgs[randNum*WIDTH*HEIGHT+k*WIDTH+l];//[randNum][k][l];
                  }
	       } 

	       //copy gaze
	       treeGazes[i][(samplesInTree[i]<<1)   ] = curr_gazes[(randNum<<1)];//[randNum][0];
	       treeGazes[i][(samplesInTree[i]<<1)+1 ] = curr_gazes[(randNum<<1)+1];//[randNum][1];


	       //copy pose
	       treePoses[i][samplesInTree[i]<<1   ] = curr_poses[(randNum<<1)];//[randNum][0];
	       treePoses[i][(samplesInTree[i]<<1)+1 ] = curr_poses[(randNum<<1)+1];//[randNum][1];


	       samplesInTree[i]++;
	     }

	    dataspace.close();
            dataset.close();
            memspace.close();
	    delete group_nearest;

	  }//for r            
      }//for i


      free( curr_poses );
      free( curr_gazes );
      free( curr_imgs  );	

      // build forest
      trees = buildRegressionTree(samplesInTree, treeImgs, treeGazes, treePoses);


      /*
       * Turn off the auto-printing when failure occurs so that we can
       * handle the errors appropriately
       */
       Exception::dontPrint();
       file = new H5File(argv[2], H5F_ACC_RDWR);
      
       //inits
       dims[0]=0;
       dims[1]=0;
       dims[2]=0;
       dims[3]=0;

      
       DataSet dataset = file->openDataSet("nearestIDs");
       DataSpace dataspace = dataset.getSpace();
       rank = dataspace.getSimpleExtentDims( dims );
cout << "nearest dims are " << dims[0] <<", "<<dims[1]<<", "<<dims[2]<<", "<<dims[3] << endl;
       DataSpace memspace( rank, dims);
	

       int max_neighbours = dims[1]; 
       test_nearest = (int *)malloc( dims[0]*dims[1]*sizeof(int) );
       if (test_nearest == NULL) {
	  cout << "Error allocating memory" << endl;
	  return -1;
       }
       dataset.read(test_nearest, PredType::NATIVE_INT, memspace, dataspace );	
 

       //headpose         
       dataset = file->openDataSet("headpose");     
       dataspace = dataset.getSpace();//dataspace???
       rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
       cout << "headpose dims are " << dims[0] <<", "<<dims[1]<<", "<<dims[2]<<", "<<dims[3] << endl;
       memspace.setExtentSimple( rank, dims );
       test_poses = (double *)malloc( dims[0]*dims[1]*sizeof(double));//me -24 varaei memory error
       if (test_poses == NULL) {//n x 2
	  cout << "Error allocating memory" << endl;
	  return -1;
       }
       dataset.read(test_poses, PredType::NATIVE_DOUBLE, memspace, dataspace ); 
        


       //gaze         
       dataset = file->openDataSet("gaze");     
       dataspace = dataset.getSpace();//dataspace???
       rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
       cout << "gaze dims are " << dims[0] <<", "<<dims[1]<<", "<<dims[2]<<", "<<dims[3] << endl;
       memspace.setExtentSimple( rank, dims );
       test_gazes = (double *)malloc( dims[0]*dims[1]*sizeof(double) );
       if (test_gazes == NULL) {// n x 2
	  cout << "Error allocating memory" << endl;
	  return -1;
       }
       dataset.read(test_gazes, PredType::NATIVE_DOUBLE, memspace, dataspace );
      

 
      /*
       * data
       */
       dataset = file->openDataSet("data");     
       dataspace = dataset.getSpace();//dataspace???
       rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
       cout << "data dims are " << dims[0] <<", "<<dims[1]<<", "<<dims[2]<<", "<<dims[3] << endl;
       memspace.setExtentSimple( rank, dims );//24x1x9x15
       test_imgs = (unsigned char  *)malloc( dims[0]*dims[1]*dims[2]*dims[3]*sizeof(unsigned char));
       if (test_imgs == NULL) {// n x 9 x 15
	  cout << "Error allocating memory" << endl;
	  return -1;
       }
       dataset.read(test_imgs, PredType::C_S1, memspace, dataspace );

       double predict[2];
       treeT *temp_predict=NULL;
       double *errors = (double *)malloc( dims[0] * sizeof(double) );
       if (errors == NULL) {
          cout << "Error allocating memory" << endl; 
	  return -1;
       }


       // test phase
       for (j = 0; j < dims[0]; j++)  {
          predict[0] =  0;
          predict[1] =  0;
	  cout << endl << "***** no." << j << ". Test sample=(" << test_gazes[(j<<1)]*(180.0/M_PI) << ", " << test_gazes[(j<<1)+1]*(180.0/M_PI) << ") " << "******" << endl;
	  for (int k = 0; k < RADIUS+1; k++)  {     

             //each tree's prediction

             temp_predict = testSampleInTree(trees[ test_nearest[j*max_neighbours + k]-1 ], test_imgs, test_poses, j );
	     predict[0] = predict[0] + temp_predict->mean[0];
	     predict[1] = predict[1] + temp_predict->mean[1];
	     cout << "\t" << k << ": mean=(" << temp_predict->mean[0]*(180.0/M_PI) << ", " << temp_predict->mean[1]*(180.0/M_PI)  << "), stdev=" <<     sqrt(pow(temp_predict->stdev[0],2)+pow(temp_predict->stdev[1],2)) *(180.0/M_PI)  << ", tree=" << test_nearest[j*max_neighbours + k]-1 <<  ", RADIUS=" <<k  << ", error=" <<   sqrt( pow(temp_predict->mean[0]-test_gazes[(j<<1) ],2) + pow(temp_predict->mean[1]-test_gazes[(j<<1)+1],2) )*(180.0/M_PI) << ", n=" << temp_predict->numOfPtrs << endl;


             for (unsigned int h=0; h < temp_predict->numOfPtrs; h++) {
	       cout << "\t\t" <<h << " prediction=(" << treeGazes[test_nearest[j*max_neighbours + k]-1][2*temp_predict->ptrs[h]]*(180.0/M_PI)  << ","<< treeGazes[test_nearest[j*max_neighbours + k]-1][2*temp_predict->ptrs[h]+1]*(180.0/M_PI)  << ")" << endl;

	     }
	     
          }
	        
          // prediction = mean prediction of all trees
          predict[0] = predict[0]/(RADIUS+1);
          predict[1] = predict[1]/(RADIUS+1);
	  errors[j] = sqrt( pow(predict[0]-test_gazes[(j<<1) ],2) + pow(predict[1]-test_gazes[(j<<1)+1],2) );

	  cout << "Final prediction=(" << predict[0]*(180.0/M_PI) << ", " << predict[1]*(180.0/M_PI) << ") and error is:" << errors[j]*(180.0/M_PI) << endl;
       }


      
       //mean error
       double mean_error = 0;
       for (j = 0; j < dims[0]; j++)  {
          mean_error =  mean_error + errors[j]/dims[0];       
       }


       //stdev error
       double stdev_error = 0;
       for (j = 0; j < dims[0]; j++)  {
          stdev_error = stdev_error + pow(errors[j]-mean_error,2);
       }
       stdev_error = stdev_error/(dims[0]);
       stdev_error = sqrt( stdev_error ); 
       cout << "mean_error(deg) is: " << mean_error*(180.0/M_PI) << endl;
       cout << "stdev_error(deg) is: " << stdev_error*(180.0/M_PI) << endl;        

       //free( errors );

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


   free( test_nearest );
   free( test_poses );
   free( test_gazes );
   free( test_imgs );

   return 0;
}

treeT *testSampleInTree(treeT *curr, unsigned char *test_img, double *test_pose, int j)  {


   if (curr->right == NULL)  {//leaf reached
      return curr;
   } 
   else  { // right or left?
      if ( abs( test_img[ j*WIDTH*HEIGHT + curr->minPx1_vert*WIDTH+curr->minPx1_hor ] - test_img[ j*WIDTH*HEIGHT + curr->minPx2_vert*WIDTH+curr->minPx2_hor ]) >= curr->thres  )
         curr = testSampleInTree(curr->right, test_img, test_pose, j);
      else
	 curr = testSampleInTree(curr->left, test_img, test_pose, j);
   }


   return curr;
}

/*
 * falloc = forest allocation
 */
tree **falloc(unsigned int *fatherSize)   {

   
   treeT **trees = (treeT **)malloc( NUM_OF_TREES * sizeof(treeT *) );
   unsigned int j;

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
      trees[i]->ptrs = (unsigned int *)malloc( fatherSize[i] * sizeof(unsigned int) );
      if (trees[i]->ptrs == NULL)  {
         cout << "Error allocating tree memory at falloc(1). Exiting\n" << endl;
         return NULL;
      }

    
      trees[i]->numOfPtrs = fatherSize[i];
      for (j = 0; j < fatherSize[i]; j++)  {
         trees[i]->ptrs[j] = j;
      } 
      trees[i]->right = NULL;
      trees[i]->left = NULL;
 

   } 
   
   return trees;
}


int treeDepth(treeT *root, int depth)  {
   static int max_depth = -1;

   if (root == NULL)
      return max_depth;
   if (depth > max_depth) 
      max_depth = depth;

   treeDepth(root->right, depth + 1);
   treeDepth(root->left, depth + 1);

   return max_depth;
}


void toDotString(treeT *curr, int myID){
	

	if(curr->left != NULL){

                outputFile << "\t" << myID << " [label=\"Samples:" << curr->numOfPtrs <<  "\npx1:(" << curr->minPx1_vert << "," << curr->minPx1_hor << ")" << "\npx2:(" << curr->minPx2_vert << "," << curr->minPx2_hor << ")" << "\nthres:" << curr->thres << "\nmse:" << curr->mse << "\", shape=rectangle, color=black]\n";
		outputFile << "\t" << myID << " -> " << (myID<<1) + 1 << "\n";
		toDotString(curr->left, (myID<<1) + 1);
		
		outputFile << "\t" << myID << " -> " << (myID<<1) + 2 << "\n";
		toDotString(curr->right, (myID<<1) + 2);
	}else{ //leaf
            outputFile << "\t" << myID << " [label=\"Samples:"  << curr->numOfPtrs << "\nmeanGaze = \n=(" << curr->mean[0] <<","<<curr->mean[1] << ")"  << "\", shape=circle, color=green]\n";
        }
}

void drawTree(treeT *root){
	outputFile << "digraph Tree{\n";
	outputFile << "\tlabel=\"Tree\"\n";
	toDotString(root, 0);
	outputFile << "}\n";

	
	return;
}

unsigned int max_size(unsigned int *fatherSize)  {
   int i = 0;
   unsigned int max_size = fatherSize[0];
   for (i = 1; i < NUM_OF_TREES; i++)  {
      if (max_size < fatherSize[i])
         max_size = fatherSize[i]; 
   }

   return max_size;
}

treeT **buildRegressionTree(unsigned int *fatherSize,unsigned char **treeImgs,double **treeGazes,double**treePoses) {
      
   treeT **trees = NULL;
   char buffer[50]; 

   #ifdef PARALLEL 
      volatile treeT *currNode = NULL;
      volatile treeT *savedNode[MAX_RECURSION_DEPTH];
      volatile unsigned int  stackindex, state;
      volatile double minSquareError[NUM_OF_THREADS];
   #else
      treeT *currNode = NULL;
      treeT *savedNode[MAX_RECURSION_DEPTH];
      unsigned int  stackindex, state;
      double minSquareError;
   #endif
  

  /*
   * allocate **trees memory
   */
   trees = falloc(fatherSize);
   if (trees == NULL) {
      return NULL;
   }
 
  #ifdef PARALLEL
  #pragma omp parallel num_threads(NUM_OF_THREADS)
  #endif
  { 	
   unsigned int *l_r_fl_fr_ptrs = NULL;
   unsigned int i, j,l,r,ltreeSize=-1, rtreeSize=-1;
   unsigned short minPx1_vert, minPx1_hor, minPx2_vert, minPx2_hor, bestThres;
   unsigned short px1_hor, px1_vert, px2_hor, px2_vert, thres;
   unsigned int counter, counter2;
   double meanLeftGaze[2], meanRightGaze[2];
   double rtree_meanGaze[2], ltree_meanGaze[2]; 
   double squareError;
   #ifdef STDEV_CALC
   double stdev_node[2];
   #endif
 
   #ifdef PARALLEL
   unsigned int tid;
   double minError; unsigned int bestworker;
   #endif 
   /*
   * caching big arrays
   */
   unsigned char *cache_treeImgs; 
   

   cache_treeImgs = (unsigned char *)malloc( ( max_size(fatherSize) *sizeof(unsigned char))<<1 );
   if (cache_treeImgs == NULL) {
      cout << "error allocating memory for caching. Exiting\n"; 
      exit(-1);
   }    
   l_r_fl_fr_ptrs = (unsigned int *)malloc( (max_size(fatherSize) *sizeof(unsigned int))<<2 ); 
   if (l_r_fl_fr_ptrs == NULL) {
      cout << "error allocating memory for ptrs2. Exiting\n";
      exit(-1);
   }

   //some random inits to avoid compiler warnings
   rtree_meanGaze[0] = -10.0;
   ltree_meanGaze[0] = -10.0;
   ltree_meanGaze[1] = -10.0;

   #ifdef PARALLEL
   tid = omp_get_thread_num();
   #endif
   for (i = 0; i < NUM_OF_TREES; i++ )  {

      #ifdef PARALLEL
      #pragma omp single
      #endif     
      {
      stackindex = 0;
      state = 1; 
      currNode = trees[i];
      }

      #ifdef PARALLEL
      #pragma omp barrier
      #endif
      while (state != 2) {
	 

         if (currNode->numOfPtrs >= 7)  {
	    #ifdef PARALLEL
            minSquareError[tid] = 10000;//a huge value
	    #else
	    minSquareError = 10000;
	    #endif

   	    minPx1_vert =    10000;//again the same
	    minPx1_hor =     10000;//also here
	    minPx2_vert=     10000;//and here..
	    minPx2_hor =     10000;//and here 
	    bestThres  =     10000;//ah, and here
 
            
	    #ifdef PARALLEL
	    counter = tid;
	    #else
	    counter = 0;
	    #endif

	    while (counter < WIDTH*HEIGHT-1 )  {
	
               px1_vert = counter/WIDTH;   
	       px1_hor = counter%WIDTH;
               counter2 = counter+1;

	       while (counter2 < WIDTH*HEIGHT)  {

	          px2_vert = counter2/WIDTH;   
	          px2_hor = counter2%WIDTH;

	          if  ( sqrt( pow(px1_vert -px2_vert,2) + pow(px1_hor-px2_hor,2) ) < 6.5 )  {  
	
                     for (j = 0; j < currNode->numOfPtrs; j++)  {
		        cache_treeImgs[(j<<1)] = treeImgs[i][ currNode->ptrs[j]*WIDTH*HEIGHT + (px1_vert<<4)-px1_vert/*px1_vert*WIDTH*/+ px1_hor];  
	      	        cache_treeImgs[(j<<1) + 1] = treeImgs[i][currNode->ptrs[j]*WIDTH*HEIGHT + (px2_vert<<4)-px2_vert/*px2_vert*WIDTH*/ + px2_hor];
                     }

		     for (thres = 20; thres <= 40; thres++) {
			l = 0;
			r = 0;
			meanLeftGaze[0]  = 0;
			meanLeftGaze[1]  = 0;
			meanRightGaze[0] = 0;
			meanRightGaze[1] = 0;

			for (j = 0; j < currNode->numOfPtrs; j++)  {
			   if ( abs(cache_treeImgs[j<<1]-cache_treeImgs[(j<<1) +1])< thres )  {
				
			      //left child
			      l_r_fl_fr_ptrs[l] = currNode->ptrs[j];
			      l++;

			      meanLeftGaze[0] = meanLeftGaze[0] + treeGazes[i][currNode->ptrs[j]<<1];
			      meanLeftGaze[1] = meanLeftGaze[1] + treeGazes[i][(currNode->ptrs[j]<<1) + 1]; 
			   }
			   else {

			      //right child
			      l_r_fl_fr_ptrs[fatherSize[i]+r] = currNode->ptrs[j];
  			      r++;	   
                                
			      meanRightGaze[0] = meanRightGaze[0] + treeGazes[i][(currNode->ptrs[j]<<1)];
			      meanRightGaze[1] = meanRightGaze[1] + treeGazes[i][(currNode->ptrs[j]<<1) + 1];			      
			   }
		        }
			meanLeftGaze[0] = meanLeftGaze[0]  / l;
			meanLeftGaze[1] = meanLeftGaze[1]  / l;
			meanRightGaze[0] = meanRightGaze[0]/ r;
			meanRightGaze[1] = meanRightGaze[1]/ r;
			
			squareError = 0;
			for (j = 0; j < l; j++)  {
			   squareError = squareError + pow(meanLeftGaze[0]-treeGazes[i][ l_r_fl_fr_ptrs[j ]<<1   ], 2)  
			    		             + pow(meanLeftGaze[1]-treeGazes[i][ (l_r_fl_fr_ptrs[j ]<<1) +1], 2);

			}
			for (j = 0; j < r; j++)  {
			   squareError = squareError + pow(meanRightGaze[0]-treeGazes[i][ (l_r_fl_fr_ptrs[fatherSize[i] + j ]<<1) ], 2)  
						     + pow(meanRightGaze[1]-treeGazes[i][ (l_r_fl_fr_ptrs[fatherSize[i] + j]<<1) +1], 2);
		  	}

			#ifdef PARALLEL
			if (squareError < minSquareError[tid] )  {
			   minSquareError[tid] = squareError;
			#else
			if (squareError < minSquareError )  {
			   minSquareError = squareError;
			#endif

			   minPx1_vert =    px1_vert;// % something random here
			   minPx1_hor =     px1_hor;// % also here
			   minPx2_vert=     px2_vert;// % and here..
			   minPx2_hor =     px2_hor;// % and here
			   bestThres  =     thres;
		           ltreeSize = l;
			   rtreeSize = r;

			   for (j = 0; j < l; j++)  {
			      l_r_fl_fr_ptrs[(fatherSize[i]<<1) + j] =  l_r_fl_fr_ptrs[j];
			   }
			   for (j = 0; j < r; j++)  {
			      l_r_fl_fr_ptrs[/*3*fatherSize[i]*/(fatherSize[i]<<2)-fatherSize[i] + j] =  l_r_fl_fr_ptrs[fatherSize[i] + j];
			   }

			   rtree_meanGaze[0] = meanRightGaze[0];
			   rtree_meanGaze[1] = meanRightGaze[1];
			   ltree_meanGaze[0] = meanLeftGaze[0];
			   ltree_meanGaze[1] = meanLeftGaze[1];
			   
			} // min
		     }// thres
		  }//if sqrt <6.5     


		  counter2++;
               }// while inner counter
            
	       //counter++;
	       #ifdef PARALLEL
	       counter = counter + NUM_OF_THREADS;
	       #else
	       counter++;
	       #endif
            }// while outter counter
         }//>=3  

         else {
            ltreeSize = 0;
	    rtreeSize = 0;
	 }

        
	 #ifdef PARALLEL
         #pragma omp barrier
         #endif


	 #ifdef PARALLEL
	 minError = minSquareError[0];
	 bestworker = 0;	
	 for (j = 1; j < NUM_OF_THREADS; j++) {
	    if (minSquareError[j] < minError)  {
	       minError = minSquareError[j];
 	       bestworker = j;
	    }
	 }
	 #endif
	
         #ifdef PARALLEL
         if (tid == bestworker)  {
	 #endif
            if (ltreeSize > 0 && rtreeSize > 0 && (ltreeSize+rtreeSize>2) )  {
	       
	 
	       //complete the last info about the father 
               currNode->minPx1_hor = minPx1_hor; 
	       currNode->minPx2_hor = minPx2_hor;
	       currNode->minPx1_vert = minPx1_vert;
	       currNode->minPx2_vert = minPx2_vert;
               currNode->thres = bestThres;
               #ifdef PARALLEL
	       currNode->mse = minSquareError[tid];
	       #else
	       currNode->mse = minSquareError;
	       #endif

		
	       //create left child
	       currNode->left = (treeT *)malloc( sizeof(treeT) );
	       if (currNode->left==NULL)  { 
	          cout << "Error allocating mem7\n";
	          exit(-1);
               }

	       currNode->left->ptrs = (unsigned int *)malloc( ltreeSize * sizeof( unsigned int ) );
	       if (currNode->left->ptrs==NULL)  {
	          cout << "Error allocating mem8\n";
	          exit(-1);
	       }
	       currNode->left->numOfPtrs = ltreeSize;
	       currNode->left->mean[0] = ltree_meanGaze[0];
	       currNode->left->mean[1] = ltree_meanGaze[1];
	       currNode->left->right = NULL;
               currNode->left->left = NULL;
               for (j = 0; j < ltreeSize; j++) {
	          currNode->left->ptrs[j] = l_r_fl_fr_ptrs[(fatherSize[i]<<1) + j];
               }

	       //stdev
	       #ifdef STDEV_CALC
	       stdev_node[0] = 0; 
               stdev_node[1]=0;
               for (j = 0; j < ltreeSize; j++)  {
                  stdev_node[0] = stdev_node[0] + pow( treeGazes[i][ 2*currNode->left->ptrs[j] ]-   ltree_meanGaze[0],2);
		  stdev_node[1] = stdev_node[1] + pow( treeGazes[i][ 2*currNode->left->ptrs[j]+1] - ltree_meanGaze[1],2);
               }
	       currNode->left->stdev[0] = stdev_node[0]/(ltreeSize);
	       currNode->left->stdev[1] = stdev_node[1]/(ltreeSize);
	       #endif
	       
		
	       //create right child
	       currNode->right = (treeT *)malloc( sizeof(treeT) ); 
	       if (currNode->right==NULL)  { 
	          cout << "Error allocating mem9\n";
	          exit(-1);
               }

	       currNode->right->ptrs = (unsigned int *)malloc( rtreeSize*sizeof(unsigned int) );
               if (currNode->right->ptrs==NULL)  {
	          cout << "Error allocating mem10\n";
	          exit(-1);
	       }

	       currNode->right->numOfPtrs = rtreeSize;
	       currNode->right->mean[0] = rtree_meanGaze[0];
	       currNode->right->mean[1] = rtree_meanGaze[1];
	       currNode->right->right = NULL;
	       currNode->right->left = NULL;
	       for (j = 0; j < rtreeSize; j++) {
	          currNode->right->ptrs[j] = l_r_fl_fr_ptrs[/*3*fatherSize[i]*/(fatherSize[i]<<2)-fatherSize[i] + j];
               }


	       #ifdef STDEV_CALC
	       stdev_node[0] = 0;
	       stdev_node[1] = 0;
               for (j = 0; j < rtreeSize; j++)  {
                  stdev_node[0] = stdev_node[0] + pow( treeGazes[i][ 2*currNode->right->ptrs[j] ] - rtree_meanGaze[0],2);
		  stdev_node[1] = stdev_node[1] + pow( treeGazes[i][ 2*currNode->right->ptrs[j]+1]- rtree_meanGaze[1],2);
               }
	       currNode->right->stdev[0] = stdev_node[0]/(rtreeSize);
	       currNode->right->stdev[1] = stdev_node[1]/(rtreeSize);
	       #endif

	       //save left brother in stack
               savedNode[stackindex] = currNode->left;
	       stackindex++;

	  
	       //currNode = right son
	       currNode = currNode->right;
 	    
            }
            else {
               if (stackindex == 0)  {

	          state = 2;
               }
               else {
	          stackindex--;
	          currNode = savedNode[stackindex];              
	       }
            }
         
	 #ifdef PARALLEL  
         }// end of bestworker update staff
         #pragma omp barrier
         #endif

      }//while state!=2

      #ifdef PARALLEL
      #pragma omp barrier
      #endif


      #ifdef PARALLEL
      if (tid == 0 )
         cout << "thread " << tid << ", i = " << i << endl; 
      #else 
         cout << "i = " << i << endl; 
      #endif 



      
      if (tid==0)  {
         try{
            sprintf(buffer, "../trees/%d_mytree.dot", i);  	
            outputFile.open( buffer );//"mytree.dot");
	    drawTree(trees[i]);
	    outputFile.close();		
         }catch(...){
	    std::cerr << "problem. Terminating\n";
	    exit(1);
         }
      }

   
   }// for i

   free( cache_treeImgs );      
   free( l_r_fl_fr_ptrs ); 
   } // end of "omp parallel"

   

   return trees;
}





/*
void drawTree(treeT * root, int indent=0)
{
 treeT **list;
 int nodesInLine;
 int i, j, k; 
 
 
cout << "Current Depth:\t\t\t\t\t******************************* TREE  *****************************************" << "\n" << endl;


 int depth = treeDepth(root, 0); 
 levelOrder = (treeT **)malloc( pow(2,depth+1) * sizeof( treeT *) );
 if (levelOrder == NULL) {
    cout << "Error allocating memory" << endl;
    return ;
 }

 howManyTabs = (int *)malloc( pow(2,depth+1) * sizeof( sizeof(int) ) );
 if (howManyTabs == NULL)  {
    cout << "Error allocating memory" << endl;
    return ;
 }

 levelOrder[0] = root;
 howManyTabs[0] = root->numOfPtrs;
 //howManyTabs = 10;

 nodesInLine = 1;
 for (i = 0; i <= depth; i++)  { 

    //t samples    
    cout << "depth:" << i; 
    for (k = 0; k < howManyTabs[0]; k++) { // d=0/tabs=9, d=1/tabs=8
          cout << "\t";
    } 
    for (j = 0; j < nodesInLine; j++)  {
       if (levelOrder[j]) {
          cout << "Samples:" << levelOrder[j]->numOfPtrs;
       }
       else
          cout << "            "; 

       for (k = 0; k <= depth - i; k++) {
          cout << "\t\t";
       }

    }
    cout  << endl;


    //print pixel 1(if node) or mean Gaze(if leaf)
    for (k = 0; k < howManyTabs[0]; k++) { // d=0/tabs=9, d=1/tabs=8
          cout << "\t";
    } 
    for (j = 0; j < nodesInLine; j++)  {


       if (levelOrder[j] == NULL)
          cout << "            "; 
       else if (levelOrder[j]->right == NULL)
          cout << "m:(" << levelOrder[j]->mean[0];// << "," << list[j]->mean[1] << ")"; 
       else 
          cout << "px1:("<< levelOrder[j]->minPx1_vert << "," << levelOrder[j]->minPx1_hor << ")";
      
          
       for (k = 0; k <= depth - i; k++) {
          cout << "\t\t";
       }
    }
    cout << endl;

   //print pixel 2(if node)
    for (k = 0; k < howManyTabs[0]; k++) { // d=0/tabs=9, d=1/tabs=8
          cout << "\t";
    } 
    for (j = 0; j < nodesInLine; j++)  {

       if (levelOrder[j] == NULL)  
	  cout << "            ";
       else if (levelOrder[j]->right == NULL)
          cout << "   " << levelOrder[j]->mean[1] << ")"; 
       else
	  cout << "px2:("<< levelOrder[j]->minPx2_vert << "," << levelOrder[j]->minPx2_hor << ")";

       for (k = 0; k <= depth - i; k++) {
          cout << "\t\t";
       }
    }
    cout << endl;

  
   //print thres(if node)
    for (k = 0; k < howManyTabs[0]; k++) { // d=0/tabs=9, d=1/tabs=8
          cout << "\t";
    } 
    for (j = 0; j < nodesInLine; j++)  {

       if (levelOrder[j] == NULL)
	  cout << "            "; 
       else if (levelOrder[j]->right == NULL)
	  cout << "            "; 
       else 
          cout << "thres:"<< levelOrder[j]->thres;
       
          
       for (k = 0; k <= depth - i; k++) {
          cout << "\t\t";
       }
    }
    cout << endl;
    cout << endl;

    for (j = i; j >= 0; j--)  {
      if (levelOrder[j] == NULL)  { 
          levelOrder[2*j+1] = NULL;
          levelOrder[2*j] = NULL; 
       }
       else {   
          levelOrder[2*j+1] = levelOrder[j]->right;
          levelOrder[2*j] = levelOrder[j]->left; 
	  howManyTabs[2*j+1] = 
       }
    }

    for (j = i; j >= 0; j--)  {
      if (levelOrder[j] == 0)  { 
          levelOrder[2*j+1] = 0;
          levelOrder[2*j] = 0; 
       }
       else {   
          levelOrder[2*j+1] = levelOrder[j]->right;
          levelOrder[2*j] = levelOrder[j]->left; 
       }
    }
    //update values
    howManyTabs = howManyTabs - 2;
    nodesInLine = nodesInLine << 1;//pow(i+1,2);
 }   

free(levelOrder);
 exit(-1);
  
}
*/

