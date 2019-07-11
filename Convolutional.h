#include <torch/script.h> // One-stop header.
class Convolutional {
	protected:
		torch::jit::script::Module module;

	public:
		void load_model(void);
		void predict(cv::Mat img,float *pose,float *gaze);

};