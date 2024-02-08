#include <torch/torch.h>

struct UnembeddingImpl : torch::nn::Module {
    UnembeddingImpl(int64_t d_model, int64_t vocab_size)
        : fc(register_module("fc", torch::nn::Linear(d_model, vocab_size))) {}

    torch::Tensor forward(torch::Tensor x) {
        return fc(x);
    }

    torch::nn::Linear fc;
};
