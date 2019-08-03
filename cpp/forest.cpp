/* Eidiko thema: Axelos Christos, Aem 1814, etos 5o, eksamino 10o
 * Titlos Eidikou Thematos: Epeksergasia dedomenwn video proswpou apo pollaples pozes 
 */

#define EXPORT_TO_FILE


#define PARALLEL
#ifdef PARALLEL
   #define NUM_OF_THREADS 4
#endif
/* - If you don't want to parallelize the algorithm, just put in comments the below define 
 * - Parallelization is done using OpenMP
 * - You can define number of threads below
 */



//#define PRINT_TREES_PREDICTIONS
/* 
 * - If you comment that out, the terminal will make the output the full predictions
 *   of all trees and not just the final mean error and standar deviation
 */


//#define PRINT_LEAF_SAMPLES
/*
 * - If you comment that out, the terminal will print also the data of all leaf node
 * that is predicted. Usefull if you want to make a regression function in the leaves
 *
 * 
 */



#define STDEV_CALC
/*
 * - useful if you need to include the standar deviation for each node for some 
 * optimizations of the algorithm(eg. "confidence" values in the leaves)
 */



#define LEAF_OPTIMIZATION
#ifdef LEAF_OPTIMIZATION
   #define MIN_SAMPLES_PER_LEAF 3
#endif
/*
 * - If you want to apply leaf_optimization(using minimum number of training samples per leaf,
 * just put the comments of
 * - You can define below the minimum number of training samples per leaf
 */



#define NUM_OF_TREES 93//204//238
/*
 * - total number of trees that get build built
 * - the number of clusters must be equal with the number below
 */



/*
 * - just some defines..
 */
#define MAX_SAMPLES_PER_TREE 1000
#define MAX_RECURSION_DEPTH 15
#define MAX_GRP_SIZE 500
#define LEFT 1
#define RIGHT 2



/*
 * - Normalized image dimensions 
 */
#define HEIGHT 9
#define WIDTH 15



/*
 * - This parameter sets the number of neighbours(R) used by the algorithm
 */
#define RADIUS 30



/*
 * Libraties
 */
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
#include <iostream>
#include <fstream>
#include <sstream>
using namespace H5;
using namespace std;

const H5std_string FILE_NAME( "myfile.h5" );

/*
 *  - every tree node or leaf is described by the following structure:
 *     a) mean: mean gaze prediction of this node
 *     b) stdev: standard deviation of this node
 *     c) mse: mean square error( calculated in buildRegressionForest function )
 *     d) left: pointer to left subtree
 *     e) right: pointer to right subtree
 *     f) ptrs: pointer(dynamic array) that shows the list of training samples in current subtree.
 *     g) numPtrs: number of training samples in current subtree. Needed because "ptrs" is dynamic array
 *     h) thres: threshold in each tree node( calculated in buildRegressionForest function )
 *     i) minPx1_hor: width of pixel 1 in each tree node ( calculated in buildRegressionForest function )
 *     j) minPx2_hor: width of pixel 2 in each tree node ( calculated in buildRegressionForest function )
 *     k) minPx1_vert: height of pixel 1 in each tree node ( calculated in buildRegressionForest function )
 *     l) minPx2_vert: height of pixel 2 in each tree node ( calculated in buildRegressionForest function )
 */

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
treeT **buildRegressionForest(unsigned int *rootSize,unsigned char **treeImgs,double **treeGazes,double**treePoses);
/*
 * - function buildRegressionForest

 * - here is implemented the main algorithm. All the decision trees are trained based on the random forest algorithm.
   More info about the algorithm's implementation can be found here https://github.com/trakaros/MPIIGaze


 * - inputs are:

      a) int *rootSize: It is an array with size equal to the number of trees. It 
      contains the number of training samples in the root node of each tree

      b) unsigned char **treeImgs: This variable is a pointer to the image data of all trees that are 
      inputs to the algorithm. The reason why is '**' instead of '*' is because the this variable is a 
      pointer to an array of pointers. Each element in the array of pointer is a '*' pointer and
      points out the memory where the image data for tree i are saved. It works as described bellow: 

      Example 1: Get 3rd training sample Image from the 2nd tree:
        treeImgs[1]=2nd tree(counting from zero) = the pointer treeImgs[1] contains all data about
        the images of training samples that are used in tree number 2
 
        treeImgs[1][ 2*WIDTH*HEIGHT] = pointer to 1st pixel of 3rd image of 2nd tree. In order to get
        the data of the 3rd training sample, we shift "2*WIDTH*HEIGHT" bytes in order to get the adress 
        of that pixel. So if we get all the data between treeImgs[1][2*WIDTH*HEIGHT] and 
        treeImgs[1][2*WIDTH*HEIGHT - 1] we get the full image of that sample

      Example 2: Get pixel (6,8) from 3rd training sample Image from the 2nd tree
        In Example 1, we found that the region of 3rd training sample Image of the 2nd tree is between
        treeImgs[1][2*WIDTH*HEIGHT] and treeImgs[1][2*WIDTH*HEIGHT-1]. So, pixel (6,8) can be found as bellow:
        treeImgs[1][2*WIDTH*HEIGHT + (6-1)*WIDTH + (8-1)]

      Generally: 
	treeImgs[numOfTree-1][ (sampleNum-1)*WIDTH*HEIGHT + (row_num-1)*WIDTH+ (col_num-1) ]


     c) double **treeGazes: This variable is a pointer to the gaze data(2d) of all trees that are inputs 
      to the algorithm. The reason why is '**' instead of '*' is because the this variable is a 
      pointer to an array of pointers. Each element in the array of pointer is a '*' pointer and
      points out the memory where the gaze data(2d) for tree i are saved. It works as described bellow:
    
      Example 1: Get 3rd training sample Gaze Data from the 2nd tree:
        treeGazes[1]=2nd tree(counting from zero) = the pointer treeGazes[1] contains all data about
        the Gazes of training samples that are used in tree number 2

        treeGazes[1][(numOfSample-1)*2] = treeGazes[1][(3-1)*2] = treeGazes[1][4], is the pointer to that
        sample. The reason why we calculate by 2 is because each sample has a (theta,phi) angle(rads), so
        i save it as: [sample1Theta, sample1Phi, sample2Theta, sample2Phi, ...] and every new sample must
        be in a position multiple of 2
  
     Example 2: Get "theta" and "phi" angles of 3rd training sample Gaze Data from the 2nd tree:
        - We already calculated the pointer to the 3rd training sample Gaze Data of the 2nd tree and it is
        treeGazes[1][4]. So theta = treeGazes[1][4] and phi = treeGazes[1][4+1]


    c) double **treePoses: This variable works in the exact same way as treesGazes, however it describes the
    2d pose angle(rad) instead of the 2d gaze angle(rad)


  *  - outputs are:

    a) treeT **trees: It contains an array of pointers. Each of these pointers are the root pointers of all 
    decision trees created. This is the trained model and it's used for the algorithm evaluation


   - Some extra info: I use 3 types of defines in that function. These defines have to do with parallelization
   and data printing. I explain them in the define session above

 */
treeT *testSampleInTree(treeT *currNode, unsigned char *test_img, double *test_pose, int numOfSample); 
/*
 * - function testSampleInTree

 * - In this function we extract the 2d-gaze data from each leaf of the R-neighbour-trees. It's a recursive
 function where we check all children until we reach a NULL node(leaf). Then we take that leaf value

 * - inputs are:

      a) treeT *currNode: the current children node of the tree.  (that variable recursively changes untill NULL is reached)
      b) unsigned char *test_img: the current image of the test sample that is evaluated(dynamically allocated)
      c) double *test_pose: the current  pose of the test sample that is evaluated(dynamically allocated)
      d) int numOfSample: number of test sample(it is needed as index for the dynamically allocated test_pose, test_img vectors)

*/

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

//https://stackoverflow.com/questions/20005784/saving-a-binary-tree-to-a-file
void saveTree(tree *t, ofstream &out) {

  if (!t)
    out << "# ";
  else {
    //out << p->data << " ";
    out << t->mean[0] << " " << t->mean[1] << " "
        << t->stdev[0] << " " << t->stdev[1] << " "
        << t->mse << " "
        << t->thres << " "
        << t->minPx1_hor << " " << t->minPx2_hor << " "
        << t->minPx1_vert << " " << t->minPx2_vert << " ";

    saveTree(t->left, out);
    saveTree(t->right, out);
  }
  return;
}

treeT *loadTree(treeT *t,std::stringstream& lineStream,int i) {
  //int token;
  //bool isNumber;
  //std::string token;
  
  //while(lineStream >> token)
  //{
  //   std::cout << "Token(" << token << ")\n";
  //}
  //std::cout << "New Line Detected\n";
  std::string temp;
  //if (i==0) {
  //  while (lineStream >> temp)
  //    cout << "LINE:" << temp << endl;
  //}


  //if (i == 0) {
  if (lineStream >> temp) {
    if (std::isdigit(temp[0]) || temp[0] == '-' )  {
      t = (treeT *)malloc( sizeof(treeT) );
      t->right = NULL;
      t->left = NULL;
      t->mean[0] = atof(temp.c_str());

      //for (int j=0;j<9;j++) {
      //  lineStream >> temp;
      //  cout << "temp:" << temp << endl; 
      //}
      lineStream >> t->mean[1];
      lineStream >> t->stdev[0];
      lineStream >> t->stdev[1];
      lineStream >> t->mse;
      lineStream >> t->thres;
      lineStream >> t->minPx1_hor;
      lineStream >> t->minPx2_hor;
      lineStream >> t->minPx1_vert;
      lineStream >> t->minPx2_vert;
     
      //cout << "************ adding node: *******" << endl;
      //cout << "mean:("<< t->mean[0] << "," <<t->mean[1]<<")" << endl; 
      //cout << "stdev:("<< t->stdev[0] << "," <<t->stdev[1]<<")" << endl; 
      //cout << "mse:" << t->mse;
      //cout << "thres:"<<t->thres<<endl;
      //cout << "right child" << endl;
      t->left  = loadTree(t->left, lineStream,i);
      //cout << "left child" << endl;
      t->right = loadTree(t->right, lineStream,i);

    }
    else {
      //cout << "leaf reached because temp=" << temp  << endl;
      return NULL;
    }
  }
  else {
    //cout << "new line detected" << endl;
    return NULL;
  }
  //}
  return t;


  //if (!readNextToken(token, fin, isNumber)) 
  //  return;
  //if (isNumber) {
  //  t = new BinaryTree(token);  
  //  loadTree(t->left, fin);
  //  loadTree(t->right, fin);
  //}
}


treeT **importForestFromTxt(ifstream& infile) {
  
  treeT **trees = NULL;
  trees = (treeT **)malloc( NUM_OF_TREES * sizeof(treeT *) );
  //for (unsigned i = 0; i < NUM_OF_TREES; i++)  {
  //   trees[i] = (treeT *)malloc( sizeof(treeT) );
  //   trees[i]->right = NULL;
  //   trees[i]->left = NULL;
  //} 
  for (int i = 0; i < NUM_OF_TREES; i++)  {
      std::string line;
      std::getline(infile, line);
      std::stringstream lineStream(line);
      std::string temp;
      trees[i] = loadTree(trees[i],lineStream,i);
  } 

  return trees;
} 


void exportForestToTxt(tree **trees,int ntrees,ofstream &out) {
  //double mean[2];
  //double stdev[2];
  //double mse;
  //struct tree *left;
  //struct tree *right;
  //unsigned int *ptrs;
  //unsigned int numOfPtrs;
  //unsigned short thres;
  //unsigned short minPx1_hor;
  //unsigned short minPx2_hor;
  //unsigned short minPx1_vert;
  //unsigned short minPx2_vert;
for (int i = 0; i < ntrees; i++)  {
  saveTree(trees[i],out);
  out << endl;
}

return;
}




/*************** MAIN ********************************************/
int main(int argc, char *argv[])  {

 
  /*
   * - For the training and test data, i use the HDF5 functionality, where I use one .h5 file for training
   * an one .h5 for testing
  
   * - You can  view what view what these files contain, using the command "hdfview <file-name>.h5
   */

  /*
   * read hdf5 data
   */
   int curr_nearest[RADIUS];
  /*
   * - curr_nearest is a temporary variable and contains the R-nearest clusters from the current Cluster
   */
   double curr_center[2];//(theta,phi) cluster center
   double *curr_gazes=NULL;
   double *curr_poses=NULL;
   unsigned char *curr_imgs=NULL;
   unsigned int curr_size;
   unsigned int *samplesInTree = NULL;//vector that contains number of samples used in each tree(dynamically allocated based on num of trees)


   int *test_nearest;
   double *test_gazes;//Total test gazes read from TEST.h5 file, if it was a static array, you can consider it as: test_gazes[MAX_GRP_SIZE][2];
   double *test_poses;//Total test poses read from TEST.h5 file,if it was a static array, you can consider it as: test_poses[MAX_GRP_SIZE][2];
   unsigned char *test_imgs;//Total test image data read from TEST.h5 file, if it was a static array, you can consider it as: test_imgs[MAX_GRP_SIZE][HEIGHT][WIDTH];


   
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

/* 
 * MEMORY ALLOCATION
 */
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




     /*
      * - The following for loop builds the training samples for each tree
      *
      * - The data are gained from the <TRAIN>.h5 file, where if you open that
      * file using the "hdfview" command you will see several groups
      *
      * - Each group contains the training samples(gaze,pose,img) that are near that group
      *
      * - The NUM_OF_TREES must be equal to the number of groups.
      *
      * - Finally, to build a tree, several clusters(the R-nearest) must contribute
      */
      //for every group
      curr_size = 0;
      for (i = 0; i < NUM_OF_TREES; i++ )  {
 	// read data of group i
        
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
         dataset.read(curr_imgs, PredType::NATIVE_UCHAR/*GIA STRING:C_S1*/, memspace, dataspace );


          	
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
	  * main Group's contribution. Based on Breiman's 2001 paper, each cluster of the
          * R nearest contribure the square root of their samples to construct a tree. These
          * square root samples are chosen randomly
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
	  * now that we finished with the central group we continue with the
          * R-nearest groups/clusters
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
            dataset.read(curr_imgs, PredType::NATIVE_UCHAR/*GIA STRING:C_S1*/, memspace, dataspace );

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
      
      //#ifdef EXPORT_TO_FILE
      //ofstream myfile;
      // for (int q=0; q<dims[0]*dims[1];q++)
      //  myfile << test_nearest[q] << " ";
      //#endif

	    for (j= 0; j < grpContribution; j++)  {
	      /*
	       * we repeat here the random selection for the R-nearest groups
	       */ 
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
	
/************************************************************************************************************
 ******** c h e c k p o i n t    h e r e ********************************************************/


     /*
      * - Here we start the building of the tree nodes. This function takes a lot of time
      * - After that function, we continue with the algorithm evaluation
      */ 
     
     #ifdef EXPORT_TO_FILE
     trees = buildRegressionForest(samplesInTree, treeImgs, treeGazes, treePoses);
     ofstream myfile;
     myfile.open ("example.txt");//"tree_info.txt");
     exportForestToTxt(trees,NUM_OF_TREES,myfile);
     myfile.close();
     myfile.open("nearests.txt");     
     #endif


     #ifdef IMPORT_FROM_FILE
     ifstream myfile;
     myfile.open("example.txt");
     trees = importForestFromTxt(myfile);
     myfile.close(); 
     outputFile.open( "trees/correct.dot");//"mytree.dot");
     draw(trees[0])
     outputFile.close();
     #endif
     cout << "ola kala:" << trees[0]->thres << endl;


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
       memspace.setExtentSimple( rank, dims );//24x1x9x15
       test_imgs = (unsigned char  *)malloc( dims[0]*dims[1]*dims[2]*dims[3]*sizeof(unsigned char));
       if (test_imgs == NULL) {// n x 9 x 15
	  cout << "Error allocating memory" << endl;
	  return -1;
       }
       dataset.read(test_imgs,PredType::NATIVE_UCHAR/*GIA STRING:C_S1*/, memspace, dataspace );
       

       dataspace.close();
       dataset.close();
       memspace.close();
       file->close(); 	
       delete file;
     
       double predict[2];
       treeT *temp_predict=NULL;
       double *errors = (double *)malloc( 2* dims[0] * sizeof(double) );
       if (errors == NULL) {
          cout << "Error allocating memory" << endl; 
	  return -1;
       }
      #ifdef EXPORT_TO_FILE 
      myfile.close();
      #endif 

    
      /*
       * - here starts the evaluation part
       *
       * - We calculate the final mean error and final standar deviation of mean error, as well as other statistical
       *   info about the trees' predictions
       */
     
       // test phase
       for (j = 0; j < dims[0]; j++)  {
          predict[0] =  0;
          predict[1] =  0;
          #ifdef PRINT_TREES_PREDICTIONS
	  cout << endl << "***** no." << j << ". Test sample=(" << test_gazes[(j<<1)]*(180.0/M_PI) << ", " << test_gazes[(j<<1)+1]*(180.0/M_PI) << ") " << "******" << endl;
	  #endif
      
	  for (int k = 0; k < RADIUS+1; k++)  {     

             //each tree's prediction

       temp_predict = testSampleInTree(trees[ test_nearest[j*max_neighbours + k]-1 ], test_imgs, test_poses, j );
	     predict[0] = predict[0] + temp_predict->mean[0];
	     predict[1] = predict[1] + temp_predict->mean[1];
	   
             
	     #ifdef PRINT_TREES_PREDICTIONS
	     cout << "\t" << k << ": mean=(" << temp_predict->mean[0]*(180.0/M_PI) << ", " << temp_predict->mean[1]*(180.0/M_PI)  << "), stdev=" <<     sqrt(pow(temp_predict->stdev[0],2)+pow(temp_predict->stdev[1],2)) *(180.0/M_PI)  << ", tree=" << test_nearest[j*max_neighbours + k]-1 <<  ", RADIUS=" <<k  << ", error=" <<   sqrt( pow(temp_predict->mean[0]-test_gazes[(j<<1) ],2) + pow(temp_predict->mean[1]-test_gazes[(j<<1)+1],2) )*(180.0/M_PI) << ", n=" << temp_predict->numOfPtrs << endl;
	    #endif
	      
	     #ifdef PRINT_LEAF_SAMPLES
             for (unsigned int h=0; h < temp_predict->numOfPtrs; h++) {
	       cout << "\t\t" <<h << " prediction=(" << treeGazes[test_nearest[j*max_neighbours + k]-1][2*temp_predict->ptrs[h]]*(180.0/M_PI)  << ","<< treeGazes[test_nearest[j*max_neighbours + k]-1][2*temp_predict->ptrs[h]+1]*(180.0/M_PI)  << ")" << endl;

	     }
	     #endif
          }
	        
         /*
          * predict = mean gaze prediction of all trees(theta, phi)
	  */ 
          predict[0] = predict[0]/(RADIUS+1);
          predict[1] = predict[1]/(RADIUS+1);
	  errors[2*j] = predict[0]-test_gazes[(j<<1)];
          if (errors[2*j] < 0)
             errors[2*j] = -errors[2*j];

	  errors[2*j+1] = predict[1]-test_gazes[(j<<1)+1];
	  if (errors[2*j+1] < 0)
	     errors[2*j+1] = -errors[2*j+1];


	 #ifdef PRINT_TREES_PREDICTIONS
	  cout << "Final prediction=(" << predict[0]*(180.0/M_PI) << ", " << predict[1]*(180.0/M_PI) << ") and error is:(" << errors[2*j]*(180.0/M_PI) << ","<< errors[2*j+1]*(180.0/M_PI) << ")" << endl;
         #endif
       }


      
       //mean error
       double mean_error[2] = {0,0};
       for (j = 0; j < dims[0]; j++)  {

          mean_error[0] =  mean_error[0] + errors[2*j]/dims[0];
	  mean_error[1] =  mean_error[1] + errors[2*j+1]/dims[0]; 
          
       }


       //stdev error
       double stdev_error[2] = {0,0};
       for (j = 0; j < dims[0]; j++)  {
          stdev_error[0] = stdev_error[0] + pow(errors[2*j]-mean_error[0],2);
	  stdev_error[1] = stdev_error[1] + pow(errors[2*j+1]-mean_error[1],2);
       }

       stdev_error[0] = stdev_error[0]/(dims[0]);
       stdev_error[1] = stdev_error[1]/(dims[0]);
       stdev_error[0] = sqrt( stdev_error[0] );
       stdev_error[1] = sqrt( stdev_error[1] );  

       cout << "mean_error(deg) is: (" << mean_error[0]*(180.0/M_PI) << ","<< mean_error[1]*(180.0/M_PI) << ") or better: " << (mean_error[0]+mean_error[1])*(180.0/M_PI) << " degrees" << endl;
       cout << "stdev_error(deg) is: (" << stdev_error[0]*(180.0/M_PI) << "," << stdev_error[1]*(180.0/M_PI) <<  ") or better: " << (stdev_error[0]+stdev_error[1])*(180.0/M_PI) << " degrees" <<  endl;        

       free( errors );
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


   //free( test_nearest );
   free( test_poses );
   free( test_gazes );
   free( test_imgs );

   return 0;
}


/*
 * - function testSampleInTree
 * 
 * - Here we find the mean gaze value of the leaf node of each tree. This value is the prediction
 *   of current tree in the evaluation stage
 */
treeT *testSampleInTree(treeT *curr, unsigned char *test_img, double *test_pose, int numOfSample)  {


   if (curr->right == NULL)  {//leaf reached
      return curr;
   } 
   else  { // right or left?
      if ( abs( test_img[ numOfSample*WIDTH*HEIGHT + curr->minPx1_vert*WIDTH+curr->minPx1_hor ] - test_img[ numOfSample*WIDTH*HEIGHT + curr->minPx2_vert*WIDTH+curr->minPx2_hor ]) >= curr->thres  )
         curr = testSampleInTree(curr->right, test_img, test_pose, numOfSample);
      else
	 curr = testSampleInTree(curr->left, test_img, test_pose, numOfSample);
   }


   return curr;
}

/*
 * function falloc(forest allocation)
 *
 * - Here we initialize all the memory that we need to construct our forest
 */
tree **falloc(unsigned int *rootSize)   {

   
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
      trees[i]->ptrs = (unsigned int *)malloc( rootSize[i] * sizeof(unsigned int) );
      if (trees[i]->ptrs == NULL)  {
         cout << "Error allocating tree memory at falloc(1). Exiting\n" << endl;
         return NULL;
      }

    
      trees[i]->numOfPtrs = rootSize[i];
      for (j = 0; j < rootSize[i]; j++)  {
         trees[i]->ptrs[j] = j;
      } 
      trees[i]->right = NULL;
      trees[i]->left = NULL;
 

   } 
   
   return trees;
}


/*
 *  - function treeDepth
 */
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





/*
 * - function max_size
 *
 * - it calculates the max number of training samples that a tree can have, in order
 *   to allocate the proper memory
 */
unsigned int max_size(unsigned int *rootSize)  {
   int i = 0;
   unsigned int max_size = rootSize[0];
   for (i = 1; i < NUM_OF_TREES; i++)  {
      if (max_size < rootSize[i])
         max_size = rootSize[i]; 
   }

   return max_size;
}


/*
 * - function buildRegressionForest
 *
 * - i explain the the details in the prototype definition of this function above
 */
treeT **buildRegressionForest(unsigned int *rootSize,unsigned char **treeImgs,double **treeGazes,double**treePoses) {
  

      
   treeT **trees = NULL;
   char buffer[50]; 

   #ifdef PARALLEL 
      volatile treeT *currNode = NULL;
      volatile treeT *savedNode[MAX_RECURSION_DEPTH];//variable used for stack emulation( explained above )
      volatile unsigned int  stackindex, state; //variable used for stack emulation( explained above )
      volatile double minSquareError[NUM_OF_THREADS];
   #else
      treeT *currNode = NULL;
      treeT *savedNode[MAX_RECURSION_DEPTH]; //variable used for stack emulation( explained above )
      unsigned int  stackindex, state; //variable used for stack emulation( explained above )
      double minSquareError;
   #endif
  

  /*
   * allocate **trees memory
   */
   trees = falloc(rootSize);
   if (trees == NULL) {
      return NULL;
   }
 
  #ifdef PARALLEL
  #pragma omp parallel num_threads(NUM_OF_THREADS)
  #endif
  {
   //define randomization   
   std::random_device rd; // obtain a random number from hardware
   std::mt19937 eng(rd()); // seed the generator
 
  
  /*
   * - Variable definitions:
   *
   *  1)minPx1_vert, minPx1_hor, minPx2_vert, minPx2_hor:
   *  - These are values/features that we need to calculate when we create each node
   *  -  We need to calculate/learn these pixel 1 and pixel 2 coordinates via training.
   *
   *  2) ltreeSize, rtreeSize:
   *  - These variables save the number of training samples that are in the left and right
   *    subtree when we create a splitting node
   *
   *  3) meanLeftGaze, meanRightGaze:
   *  - Temporary variables that save the mean gaze value of the right and left subtree
   *  
   *  4) rtree_meanGaze, ltree_meanGaze:
   *  - Variables that get the final mean gaze value of the right and left subtree. When
   *    we find a new minimum square error we assign them as rtree_meanGaze = meanRightGaze
   *    and ltree_meanGaze = meanLeftGaze
   * 
   *  5) squareError, minSquareError:
   *  - Temporary variable used to calculate error. We save the min(squareError) in minSquareError
   *
   *  6) counter:
   *  - This variable counts the number of searches until it reaches value square root of all possible
   *  feature compinations. Used to implement the random search method
   *
   *  7) l_r_fl_fr_ptrs:
   *  - I'm sorry for this awful variable name, but i tried to give a short and as well explained variable
   *    name as I could. The reason why I gave that awful variable name is because I wanted to "cache" all
   *    the important data of 4 variables in one, because it made great improvement in Performance
   *
   *  - It comes from: "left_pointers, right_pointers, final_left_pointers and final_right_pointers" and it is
   *    a 2d variable with dimensions: 4 x max_root_size(max_root_size is the maximum number of samples that  
   *    root of the trees can have
   *
   *  - The left_pointers and right_pointers save TEMPORARILY the indexes of training samples. These indexes then
   *    can be used then to "load" values from variables treeImgs, treeGazes, treePoses.
   *
   *  - The final_left and final_right pointers save the final value of these indexes described above. When we 
   *    calculate the final_left and final_right indexes, we assign the left to the final_left and right to the
   *    final right positions  
   *
   *  - For example, the following code assigns left to final_left and right to final_right indexes
   *    for (i = 0; i < ltree_size; i++)  { // assign left to final_left
   *       l_r_fl_fr_ptrs[2*max_root_size + i] = l_r_fl_fr_ptrs[0*max_root_size + i];
   *    }	  
   *    for (i = 0; i < rtree_size; i++) { // assign right to final_right
   *       l_r_fl_fr_ptrs[3*max_root_size + i] = l_r_fl_fr_ptrs[1*max_root_size + i];
   *    }   
   */	

   unsigned int *l_r_fl_fr_ptrs = NULL;
   unsigned int i, j,l,r,ltreeSize=-1, rtreeSize=-1;
   unsigned short minPx1_vert, minPx1_hor, minPx2_vert, minPx2_hor, bestThres;
   unsigned short px1_hor, px1_vert, px2_hor, px2_vert, thres;
   unsigned int counter, randNum;
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
   

   cache_treeImgs = (unsigned char *)malloc( ( max_size(rootSize) *sizeof(unsigned char))<<1 );
   if (cache_treeImgs == NULL) {
      cout << "error allocating memory for caching. Exiting\n"; 
      exit(-1);
   }    
   l_r_fl_fr_ptrs = (unsigned int *)malloc( (max_size(rootSize) *sizeof(unsigned int))<<2 ); 
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

      
     /*
      * - state variable can get the following values:
      *    a) state = 1: tree is in the building phase
      *    b) state = 2: tree has been built. Go to next tree
      */
     
      #ifdef PARALLEL
      #pragma omp barrier
      #endif
      while (state != 2) {

         /*
          - In every loop of this while, I emulate the following recursion:

          - Lets say that a tree has only the root node and we have done our 1st split in this node.
	  - Then, there are created 2 child nodes which have the root as father
          - What we do is put the left children in the "stack" and make the right as a root
          - Now repeat that for the new root, until the new root becomes NULL
          - Then "pop" from the stack the last "left" node and make him the new root
          - The algorithm repeats until there are not any left children in the stack

          - That's how every tree is built. However i dont make that recursively. Instead, i emulate it
          by using the variables: stackindex, currNode and savedNode
    
          */


    
	#ifdef LEAF_OPTIMIZATION
        if (currNode->numOfPtrs >= MIN_SAMPLES_PER_LEAF)  {
	#endif

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


	   /*
            * - The number of loop iterations should normaly be (WIDTH*HEIGHT)^2 if we had made grid search of all
            *   features, because the features pixe1 and pixel2 can both get values between [0 -> WIDTH*HEIGHT-1]. 
            *
            * - However, as i have described in the report, i use the square root of that and i make (random search)
  	    *
      	    * - So, the number of iterations should be sqrt( (WIDTH*HEIGHT)^2) = WIDTH*HEIGHT
            */
	    while (counter <  WIDTH*HEIGHT )  {
                          

	      /*
               * - as Breiman says in his paper(2001), feature selection should be random
	       * - that's why i randomly select pixel1 and pixel2	
               */
                  std::uniform_int_distribution<> distr(0, WIDTH*HEIGHT - 1); // range
	          randNum = distr(eng);
                  px1_vert = randNum/WIDTH;   
	          px1_hor = randNum%WIDTH;
	      do { 
	          randNum =  distr(eng); 
	          px2_vert = randNum/WIDTH;   
	          px2_hor =  randNum%WIDTH;

	      } while ( (sqrt( pow(px1_vert -px2_vert,2) + pow(px1_hor-px2_hor,2) ) > 6.5 ) || (randNum == px1_vert*WIDTH+px1_hor ));
	      /*
		- pixel1 and pixel2 should not be more far than 6.5 pixel distance
		- I follow the instructions of Sugano's paper
	      */
	

		 /*
		  * - I use right/left shifts in order to implement some multiplications/divisions faster
 		  *
		  * - Now in this loop we classify the training samples that have reached the current node into the left subtree or right,
 		  *   based on the px1, px2 that we randomly selected
		  * 
		  * - We randomly select also the threshold between 10 and 100  
		  */
                  for (j = 0; j < currNode->numOfPtrs; j++)  {


		    /*
		     * - here i just want to make the cache memory more effective
  		     * - so, i extract the really usable data into a special variable "cache_treeImgs"
		     * - i need to extract only two pixel-ascii values per sample(2 Bytes each sample)
		     */
		     cache_treeImgs[(j<<1)] = treeImgs[i][ currNode->ptrs[j]*WIDTH*HEIGHT + (px1_vert<<4)-px1_vert/*px1_vert*WIDTH*/+ px1_hor];  
	      	        cache_treeImgs[(j<<1) + 1] = treeImgs[i][currNode->ptrs[j]*WIDTH*HEIGHT + (px2_vert<<4)-px2_vert/*px2_vert*WIDTH*/ + px2_hor];
                     }


		     //q is just a counter
		     for (int q = 0; q < sqrt(50); q++)  {
		        std::uniform_int_distribution<> distr(10, 100); // range
	       		thres =  distr(eng); 
 
			l = 0;
			r = 0;
			meanLeftGaze[0]  = 0;
			meanLeftGaze[1]  = 0;
			meanRightGaze[0] = 0;
			meanRightGaze[1] = 0;


		       /*
			* - split all samples in the current node into the left and right subtree
			*/
			for (j = 0; j < currNode->numOfPtrs; j++)  {

			  /*
			   * - Here is the split part
			   *
 			   * - If the ascii distance between the pixel1 and pixel2(their values are cached in cache_treeImgs)
			   *   is smaller than the thres, then the sample goes in left subtree, else in the right 
			   */
			   if ( abs(cache_treeImgs[j<<1]-cache_treeImgs[(j<<1) +1])< thres )  {
				
			      //left child
			      l_r_fl_fr_ptrs[l] = currNode->ptrs[j];
			      l++;

			      meanLeftGaze[0] = meanLeftGaze[0] + treeGazes[i][currNode->ptrs[j]<<1];
			      meanLeftGaze[1] = meanLeftGaze[1] + treeGazes[i][(currNode->ptrs[j]<<1) + 1]; 
			   }
			   else {

			      //right child
			      l_r_fl_fr_ptrs[rootSize[i]+r] = currNode->ptrs[j];
  			      r++;	   
                                
			      meanRightGaze[0] = meanRightGaze[0] + treeGazes[i][(currNode->ptrs[j]<<1)];
			      meanRightGaze[1] = meanRightGaze[1] + treeGazes[i][(currNode->ptrs[j]<<1) + 1];			      
			   }
		        }
			meanLeftGaze[0] = meanLeftGaze[0]  / l;
			meanLeftGaze[1] = meanLeftGaze[1]  / l;
			meanRightGaze[0] = meanRightGaze[0]/ r;
			meanRightGaze[1] = meanRightGaze[1]/ r;
			

			/*
			 * - here we calculate the squareError that caused the specific split
			 */
			squareError = 0;
			for (j = 0; j < l; j++)  {
			   squareError = squareError + pow(meanLeftGaze[0]-treeGazes[i][ l_r_fl_fr_ptrs[j ]<<1   ], 2)  
			    		             + pow(meanLeftGaze[1]-treeGazes[i][ (l_r_fl_fr_ptrs[j ]<<1) +1], 2);

			}
			for (j = 0; j < r; j++)  {
			   squareError = squareError + pow(meanRightGaze[0]-treeGazes[i][ (l_r_fl_fr_ptrs[rootSize[i] + j ]<<1) ], 2)  
						     + pow(meanRightGaze[1]-treeGazes[i][ (l_r_fl_fr_ptrs[rootSize[i] + j]<<1) +1], 2);
		  	}


		       /*
			* -if that split caused the minimum square error, then we save 
			* these parameters that minimize the error
			*/
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
			      l_r_fl_fr_ptrs[(rootSize[i]<<1) + j] =  l_r_fl_fr_ptrs[j];
			   }
			   for (j = 0; j < r; j++)  {
			 
			      /*
			       * - here instead of making a multiplication by three, a make a left shift by 2 and make a substract
			       */	
			      l_r_fl_fr_ptrs[/*3*rootSize[i]*/(rootSize[i]<<2)-rootSize[i] + j] =  l_r_fl_fr_ptrs[rootSize[i] + j];
			   }

			   rtree_meanGaze[0] = meanRightGaze[0];
			   rtree_meanGaze[1] = meanRightGaze[1];
			   ltree_meanGaze[0] = meanLeftGaze[0];
			   ltree_meanGaze[1] = meanLeftGaze[1];
			   
			} // min
		     }// thres
		//  }//if sqrt <6.5     


		//  counter2++;
               //}// while inner counter
            
	       //counter++;
	       #ifdef PARALLEL
	       counter = counter + NUM_OF_THREADS;
	       #else
	       counter++;
	       #endif
            }// while outter counter
       // }//>=3  


         #ifdef LEAF_OPTIMIZATION
	 }
         else {
            ltreeSize = 0;
	         rtreeSize = 0;
         }
	 #endif
        

	 #ifdef PARALLEL
         #pragma omp barrier
         #endif


	/*
	 * - sychronization staff, if #PARALLEL is defined
	 */
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

	    /*
	     * - if both left and right subtrees have at least one sample, then create those 2 children
	     * - else if left or right have not any samples, don't create any left or right children and
	     *   the current node will be a terminal node 
	     */
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
	          currNode->left->ptrs[j] = l_r_fl_fr_ptrs[(rootSize[i]<<1) + j];
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
	          currNode->right->ptrs[j] = l_r_fl_fr_ptrs[/*3*rootSize[i]*/(rootSize[i]<<2)-rootSize[i] + j];
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
		  /*
		   * - there are no more left children waiting in the stack, so the tree is done. Go to the next tree!
		   */
	          state = 2;
               }
               else {
		 /*
		  * - repeat the procedure for the left child that is saved in the  stack
		  */
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
         cout << "tree: " << i << endl; 
      #else 
         cout << "current tree:  " << i << endl; 
      #endif 

      #ifdef PARALLEL
      if (tid==0)  {
      #endif
         try{
            sprintf(buffer, "trees/%d_mytree.dot", i);  	
            outputFile.open( buffer );//"mytree.dot");
	           drawTree(trees[i]);
	           outputFile.close();
		
         }catch(...){
	    std::cerr << "problem. Terminating\n";
	    exit(1);
         }
      #ifdef PARALLEL
      }
      #endif

   
   }// for i

   free( cache_treeImgs );      
   free( l_r_fl_fr_ptrs ); 
   } // end of "omp parallel"

   

 


   return trees;
}





