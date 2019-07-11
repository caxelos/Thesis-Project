#include <torch/script.h> // One-stop header.
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "Convolutional.h"
using namespace std;

void Convolutional::load_model(void) {
	//torch::jit::script::Module module = torch::jit::load("../model.pt");
	this->module = torch::jit::load("../model.pt");

}

//tsekare edw:https://github.com/pytorch/pytorch/issues/12506
//https://gist.github.com/zeryx/526dbc05479e166ca7d512a670e6b82d
//https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/6
//permute:https://stackoverflow.com/questions/51143206/difference-between-tensor-permute-and-tensor-view-in-pytorch
//efficient push_back:https://en.cppreference.com/w/cpp/utility/move

void Convolutional::predict(cv::Mat imgCV,float *poseF,float *gaze) {
	this->module = torch::jit::load("../model.pt");

	std::vector<torch::jit::IValue> inputs;
    torch::Tensor pose= torch::rand({1,2});

    //auto img = torch::from_blob(imgCV.data, {1, 1, 60, 36});
    //cout << "size:" << img.sizes() << endl;//size:[1, 36, 60, 1]
    //img = img.permute({0, 3, 1, 2});

    //prosoxh! Apo torch::kU8, egine: torch::kFloat32, logw tou Normalization me to 1/255
    torch::Tensor img = (torch::from_blob(imgCV.data, {1, 1,imgCV.rows,imgCV.cols},torch::kFloat32));//torch::Tensor img= torch::from_blob(imgCV.ptr<float>(),{imgCV.rows,imgCV.cols});//
    std::cout << img.slice(/*dim=*/1, /*start=*/0, /*end=*/1) << '\n';//3.2470e-09
    
    //std::vector<int64_t> sizes = {1, 1, imgCV.rows, imgCV.cols};
    //at::TensorOptions options(at::ScalarType::Byte);
	//at::Tensor img = torch::from_blob(imgCV.data, at::IntList(sizes), options);

    pose[0][0]=poseF[0];pose[0][1]=poseF[1];//int a = img[0][0].item<int>();
    //inputs.push_back(torch::zeros({1,1,60, 36})) ;//inputs.emplace_back(img);//inputs.push_back(img);
	inputs.push_back(img);
	inputs.push_back(pose);
	torch::Tensor output = module.forward(inputs).toTensor();
	gaze[0] = output[0][0].item<float>();
	gaze[1] = output[0][1].item<float>();
	//cout << "gaze_n:("<<gaze[0]* 180.0/M_PI<<","<<gaze[1]* 180.0/M_PI <<")"<<endl;
	cv::Size s = imgCV.size();
	//int rows = s.height;
	//int cols = s.width;
	//cout << "image dims:(rows=" << rows<<",cols="<<cols<<")"<<endl;
	//img = output.data;
				
	//std::cout << img.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';//3.2470e-09


	//cout << "real img[0][0] is:" << output.data[0] << endl;
	//cout << "tensor img[0][0] is:" << img[0][0][0][0].item<unsigned char>()-' ' << endl;
}



