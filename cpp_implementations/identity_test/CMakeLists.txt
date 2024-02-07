cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(identity-test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(identity-test identity-test.cpp)
target_link_libraries(identity-test "${TORCH_LIBRARIES}")
set_property(TARGET identity-test PROPERTY CXX_STANDARD 17)