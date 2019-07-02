
// sudo cmake -DCMAKE_PREFIX_PATH=../libtorch .. && make
//isws thelei "sudo su"
#include <iostream>
#include <memory>
#include "libtorch/include/torch/script.h" // One-stop header.

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  // Error:The bellow command is for nightly build only
  //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("model.pt");
  torch::jit::script::Module module = torch::jit::load("model.pt");
  
  //assert(module != nullptr);
  std::cout << "ok\n";
}