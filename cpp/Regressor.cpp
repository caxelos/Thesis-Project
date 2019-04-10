#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include "H5Cpp.h"

using namespace H5;
using namespace std;

#define NUM_OF_TREES 238
#define RADIUS 30
#define HEIGHT 9
#define WIDTH 15
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "Regressor.h"



void Regressor::load_model(void) {
	ifstream myfile;
		myfile.open("example.txt");
		importForestFromTxt(myfile);
		importNearestTrees();
		myfile.close(); 
}


void Regressor::predict(double *headpose, unsigned char *eyeImg, double *predict) {
	predict[0] =  0;
	predict[1] =  0;
	treeT *temp_predict;

	//calculate center(=c) or N closest centers
	int c=0;
	for (int k = 0; k < RADIUS+1; k++)  {     
		temp_predict = testSampleInTree(this->trees[this->nearests[c*60+k-1]], eyeImg);
		predict[0] = predict[0] + temp_predict->mean[0];
		predict[1] = predict[1] + temp_predict->mean[1];
	}
	predict[0] = predict[0]/(RADIUS+1);
	predict[1] = predict[1]/(RADIUS+1);
}       

void Regressor::close() {
	for (int i = 0; i < NUM_OF_TREES; i++)  {
			free(this->trees[i]);
		}
		free(this->trees);
		free(this->nearests);
		free(this->centers);
}


        //functions:
treeT *Regressor::loadTree(treeT *t,std::stringstream& lineStream) {
	std::string temp;
	if (lineStream >> temp) {
		if (std::isdigit(temp[0]) || temp[0] == '-' )  {
			t = (treeT *)malloc( sizeof(treeT) );
			t->right = NULL;
			t->left = NULL;
			t->mean[0] = atof(temp.c_str());

			lineStream >> t->mean[1];
			lineStream >> t->stdev[0];
			lineStream >> t->stdev[1];
			lineStream >> t->mse;
			lineStream >> t->thres;
			lineStream >> t->minPx1_hor;
			lineStream >> t->minPx2_hor;
			lineStream >> t->minPx1_vert;
			lineStream >> t->minPx2_vert;
			t->left  = loadTree(t->left, lineStream);
			t->right = loadTree(t->right, lineStream);
		}
		else { 
			return NULL;//leaf reached
		}
	}
	else {
		return NULL;//new line detected
	}
	return t;
}

void Regressor::importForestFromTxt(ifstream& infile) {
    this->trees = (treeT **)malloc( NUM_OF_TREES * sizeof(treeT *) );
    for (int i = 0; i < NUM_OF_TREES; i++)  {
      std::string line;
      std::getline(infile, line);
      std::stringstream lineStream(line);
      std::string temp;
      this->trees[i] = loadTree(this->trees[i],lineStream);
    } 
}
void Regressor::importNearestTrees(void) {
    H5File *file = new H5File("TRAIN_samples_10000_dist0.05.h5", H5F_ACC_RDWR);
    int rank; 
    char grpName[10];
    Group *group = NULL;
    hsize_t dims[4]; /* memory space dimensions */
	
	this->nearests = (int *)malloc(NUM_OF_TREES*60* sizeof(int));
	this->centers = (double *)malloc(NUM_OF_TREES*2*sizeof(double));
	for (int i = 0; i < NUM_OF_TREES; i++ )  {

		//this->nearests[i] = (int *)malloc(60*sizeof(int));
		//this->centers[i] = (double *)malloc(2*sizeof(double));

		sprintf(grpName, "g%d", i+1); 
		group = new Group(file->openGroup( grpName ) );
		DataSet dataset = group->openDataSet("nearestIDs");     
        DataSpace dataspace = dataset.getSpace();//dataspace???
        
        rank = dataspace.getSimpleExtentDims( dims );// get rank =    numOfDims
        DataSpace memspace( rank, dims );     
        dataset.read(this->nearests+i*60, PredType::NATIVE_INT, memspace, dataspace );  

        dataset = group->openDataSet("center");     
        dataspace = dataset.getSpace();//dataspace???
        rank = dataspace.getSimpleExtentDims( dims );// get rank = numOfDims
        memspace.setExtentSimple( rank, dims );
        dataset.read(this->centers+i*2, PredType::NATIVE_DOUBLE, memspace, dataspace ); 
	}
}

treeT *Regressor::testSampleInTree(treeT *curr,unsigned char *eyeImg)  {
	if (curr->right == NULL)  {//leaf reached
		return curr;
	} 
	else  { // right or left?
		if ( abs(eyeImg[curr->minPx1_vert*WIDTH+curr->minPx1_hor]-eyeImg[curr->minPx2_vert*WIDTH+curr->minPx2_hor ]) >= curr->thres  )
			curr = testSampleInTree(curr->right, eyeImg);
		else
			curr = testSampleInTree(curr->left, eyeImg);
	}
	return curr;
} 


int main() {
	Regressor forest;
	double prediction[2];
	double headpose[2];
	headpose[0]=0;headpose[1]=0;
	//unsigned char *eyeImg;
	//eyeImg=(unsigned char *)malloc(WIDTH*HEIGHT);
	cv::Mat eyeImg = cv::Mat::zeros(36,60,cv::DataType<unsigned char>::type);
	forest.load_model();
	forest.predict(headpose,eyeImg.data,prediction);
	cout << "predicted:(" << prediction[0]<<"," << prediction[1]<<")" << endl;
	forest.close();

	return 0;
}


