#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <torch/torch.h>

struct LayerNormImpl : public torch::nn::Module {
  LayerNormImpl(int64_t d_model, double eps);
  torch::Tensor forward(torch::Tensor x);

  torch::Tensor weight, bias;
  double eps;
};

TORCH_MODULE(LayerNorm);

#endif