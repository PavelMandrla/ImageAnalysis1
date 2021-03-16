cmake_minimum_required(VERSION 3.0.0)
project(dzo VERSION 0.1.0)

add_library(backprop bpnn/backprop.cpp)
target_include_directories(backprop PUBLIC bpnn)

add_subdirectory(IALib)

find_package( OpenCV REQUIRED )

add_executable(dzo
        Exercise5.cpp
        #main.cpp
        )

target_link_libraries( dzo
        ${OpenCV_LIBS}
        backprop
        IA_LIB
        )
