#include "multi_layer_perceptron.hpp"

MLPImpl::MLPImpl(int64_t d_model, int64_t d_mlp)
    : fc1(register_module("fc1", torch::nn::Linear(d_model, d_mlp))),
        fc2(register_module("fc2", torch::nn::Linear(d_mlp, d_model))) {}

torch::Tensor MLPImpl::gelu(torch::Tensor x) {
    // Implementation of GeLU used by GPT2, Pythia, etc
    return 0.5 * x * (1.0 + torch::tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x.pow(3))));
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    return fc2(gelu(fc1(x)));
}