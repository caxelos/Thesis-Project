#include <fstream>
#include <sstream>
#include <stdlib.h>

#define NUM_OF_TREES 238
#define RADIUS 30
#define HEIGHT 9
#define WIDTH 15


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

class Regressor {
	public:
        //functions
		void load_model(char *filename) {
			ifstream myfile;
     		myfile.open(filename);
     		this.trees = importForestFromTxt(myfile);
     		this.nearest = importNearestTrees(myfile);
     		myfile.close(); 
		}


		void predict(double *headpose, unsigned char *eyeImg, double *predict) {
			predict[0] =  0;
			predict[1] =  0;
			for (int k = 0; k < RADIUS+1; k++)  {     
				temp_predict = testSampleInTree(this.trees[this.nearest[j*max_neighbours + k]-1 ], eyeImg);
				predict[0] = predict[0] + temp_predict->mean[0];
				predict[1] = predict[1] + temp_predict->mean[1];
			}
			predict[0] = predict[0]/(RADIUS+1);
			predict[1] = predict[1]/(RADIUS+1);
		}       

		void close() {
			for (i = 0; i < NUM_OF_TREES; i++)  {
      			free(this.trees[i]);
      			free(this.nearest[i])
   			}
   			free(this.trees);    
		}

    protected:
    	treeT **trees=NULL;
    	int **nearest=NULL;

        //functions:
		treeT *loadTree(treeT *t,std::stringstream& lineStream) {
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

        void importForestFromTxt(ifstream& infile) {
            this.trees = (treeT **)malloc( NUM_OF_TREES * sizeof(treeT *) );
            for (int i = 0; i < NUM_OF_TREES; i++)  {
              std::string line;
              std::getline(infile, line);
              std::stringstream lineStream(line);
              std::string temp;
              this.trees[i] = loadTree(this.trees[i],lineStream);
            } 
        }
        treeT *testSampleInTree(treeT *curr,unsigned char *eyeImg)  {
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
};

int main() {
	Regressor forest;
	double prediction[2];
	double headpose[2];
	headpose[0]=0;headpose[1]=0;
	unsigned char *eyeImg;
	eyeImg=calloc(WIDTH*HEIGHT);

	forest.load_model("example.txt");
	forest.predict(headpose,eyeImg,prediction);
	forest.close()

	return 0;
}


