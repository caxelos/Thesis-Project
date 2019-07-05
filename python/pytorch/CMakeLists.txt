############################
# RUN WITH THIS COMMAND AS ROOT:
# sudo cmake -DCMAKE_PREFIX_PATH=/home/user/MPIIGaze/python/pytorch/libtorch .. && make
####################


cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_cmake)

#set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(ok ok.cpp)
target_link_libraries(ok "${TORCH_LIBRARIES}")
set_property(TARGET ok PROPERTY CXX_STANDARD 11)

#prosoxi!!!.If I have a "libc10.so" warning in "usr/lib":
#Cannot generate a safe runtime search path for target ok because files in
#  some directories may conflict with libraries in implicit directories,run the command:
#cp ../libtorch/lib/libc10.so /usr/lib
