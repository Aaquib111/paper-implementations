#include <torch/torch.h>

struct LayerNormImpl : torch::nn::Module {
    LayerNormImpl(int64_t d_model, double eps = 1e-5) 
        : eps(eps) {
        weight = register_parameter("weight", torch::ones({d_model}));
        bias = register_parameter("bias", torch::zeros({d_model}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto mean = x.mean(-1, true);
        auto std = (x - mean).pow(2).mean(-1, true).sqrt();
        return weight * (x - mean) / (std + eps) + bias;
    }

    torch::Tensor weight, bias;
    double eps;
};
