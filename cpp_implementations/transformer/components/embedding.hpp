#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <torch/torch.h>

struct EmbeddingImpl : public torch::nn::Module {
    EmbeddingImpl(int64_t vocab_size, int64_t embedding_dim);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Embedding embedding;
};

TORCH_MODULE(Embedding);

#endif