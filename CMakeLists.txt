cmake_minimum_required(VERSION 3.0.0)
project(ImageAnalysis1 VERSION 0.1.0)

add_library(backprop bpnn/backprop.cpp)
target_include_directories(backprop PUBLIC bpnn)

add_subdirectory(IALib)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(IA
        #cviko7.cpp
        #E8.cpp
        Exercise11.cpp)

target_link_libraries( IA
        ${OpenCV_LIBS}
        backprop
        IA_LIB
        OpenMP::OpenMP_CXX
        )
