#include "layer_norm.hpp"

LayerNormImpl::LayerNormImpl(int64_t d_model, double eps) 
    : eps(eps) {
    weight = register_parameter("weight", torch::ones({d_model}));
    bias = register_parameter("bias", torch::zeros({d_model}));
}

torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
    auto mean = x.mean(-1, true);
    auto std = (x - mean).pow(2).mean(-1, true).sqrt();
    return weight * (x - mean) / (std + eps) + bias;
}