struct tree {
   double mean[2];
   double stdev[2];
   double mse;
   struct tree *left;
   struct tree *right;
   //unsigned int *ptrs;
   //unsigned int numOfPtrs;
   unsigned short thres;
   unsigned short minPx1_hor;
   unsigned short minPx2_hor;
   unsigned short minPx1_vert;
   unsigned short minPx2_vert;
};
typedef struct tree treeT;

class Regressor {  
	protected:   
		treeT **trees;
    	int *nearests;
    	double *centers; 
    	treeT *loadTree(treeT *t,std::stringstream& lineStream);
    	void importForestFromTxt(ifstream& infile);
    	void importNearestTrees(void);
    	treeT *testSampleInTree(treeT *curr,unsigned char *eyeImg);
	public:   
		void load_model(void);
		void predict(double *headpose, unsigned char *eyeImg, double *predict);
		void close(); 
};