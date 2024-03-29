cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(transformer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    transformer driver.cpp components/transformer_block.cpp 
    components/embedding.cpp components/unembedding.cpp
    components/transformer.cpp components/attention.cpp 
    components/layer_norm.cpp components/multi_layer_perceptron.cpp
)
target_link_libraries(transformer "${TORCH_LIBRARIES}")
set_property(TARGET transformer PROPERTY CXX_STANDARD 17)