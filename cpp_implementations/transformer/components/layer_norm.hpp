#include <torch/torch.h>

#ifndef LAYERNORM_H
#define LAYERNORM_H

struct LayerNormImpl : public torch::nn::Module {
    LayerNormImpl(int64_t d_model, double eps = 1e-5);
    torch::Tensor forward(torch::Tensor x);

    torch::Tensor weight, bias;
    double eps;
};

TORCH_MODULE(LayerNorm);

#endif