
// sudo cmake -DCMAKE_PREFIX_PATH=../libtorch .. && make
//isws thelei "sudo su"
#include <iostream>
#include <memory>
#include "libtorch/include/torch/script.h" // One-stop header.


//https://github.com/iamhankai/cpp-pytorch/blob/master/example-app.cpp
int main(int argc, const char* argv[]) {
	//if (argc != 2) {
	//	std::cerr << "usage: example-app <path-to-exported-script-module>\n";
	//	return -1;
	//}

	// Deserialize the ScriptModule from a file using torch::jit::load().
	// Error:The bellow command is for nightly build only
	//std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("model.pt");
	torch::jit::script::Module module = torch::jit::load("../model.pt");
	//assert(module != nullptr);

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	//std::vector<unsigned char> img = {1,2,3};
	//std::vector<float> pose = {3.0, 2.0};

	inputs.push_back(torch::ones({1,1,60, 36}));//(1,chanel,width,height)
	inputs.push_back(torch::ones({1,2}));
	//inputs.push_back(torch::tensor(img));
	//inputs.push_back(torch::tensor(pose));
	
	//prosoxi stin arxitektoniki tou diktuou.Isws einai diaforetiki!
	torch::Tensor output = module.forward(inputs).toTensor();


	// Execute the model and turn its output into a tensor.
	//auto output = module.forward(inputs).toTensor();

	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';



  std::cout << "ok\n";
}