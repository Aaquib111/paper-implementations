cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(transformer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(transformer transformer.cpp)
target_link_libraries(transformer "${TORCH_LIBRARIES}")
set_property(TARGET transformer PROPERTY CXX_STANDARD 17)

add_executable(attention attention.cpp)
target_link_libraries(attention "${TORCH_LIBRARIES}")
set_property(TARGET attention PROPERTY CXX_STANDARD 17)