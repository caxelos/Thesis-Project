#include <torch/script.h> // One-stop header.
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "Convolutional.h"

void Convolutional::load_model(void) {
	torch::jit::script::Module module = torch::jit::load("../model.pt");
}


void Convolutional::predict(cv::Mat imgCV,float *poseF,float *gaze) {
	std::vector<torch::jit::IValue> inputs;
    torch::Tensor pose= torch::rand({1,2});
    torch::Tensor img= torch::from_blob(imgCV.data, {1, 1, imgCV.rows,imgCV.cols});//torch::rand({1,1,60, 36})
    int a = img[0][0].item<int>();
    pose[0][0]=poseF[0];pose[0][1]=poseF[1];
    inputs.push_back(img);
	//inputs.push_back(torch::zeros({1,1,60, 36}));
	inputs.push_back(pose);
	//prosoxi stin arxitektoniki tou diktuou.Isws einai diaforetiki!
	torch::Tensor output = module.forward(inputs).toTensor();
	gaze[0] = output[0].item<float>();
	gaze[1] = output[1].item<float>();

	cv::Size s = imgCV.size();
	//int rows = s.height;
	//int cols = s.width;
	//cout << "image dims:(rows=" << rows<<",cols="<<cols<<")"<<endl;
	//img = output.data;
				
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';


	//cout << "real img[0][0] is:" << output.data[0] << endl;
	//cout << "tensor img[0][0] is:" << img[0][0][0][0].item<unsigned char>()-' ' << endl;
}



