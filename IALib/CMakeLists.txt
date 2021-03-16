find_package(OpenCV REQUIRED)
add_library(IA_LIB STATIC IALib.h IALib.cpp)
target_link_libraries(IA_LIB ${OpenCV_LIBS})