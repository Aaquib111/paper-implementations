#include <torch/torch.h>

#ifndef MLP_H
#define MLP_H

struct MLPImpl : public torch::nn::Module {
    MLPImpl(int64_t d_model, int64_t d_mlp);
    torch::Tensor gelu(torch::Tensor x);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1, fc2;
};

TORCH_MODULE(MLP);

#endif