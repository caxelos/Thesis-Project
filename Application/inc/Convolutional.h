#include <torch/script.h> // One-stop header.
class Convolutional {
	protected:
		torch::jit::script::Module module;

	public:
		void load_model(char *modelpath);
		void predict(cv::Mat img,float *pose,float *gaze);

};