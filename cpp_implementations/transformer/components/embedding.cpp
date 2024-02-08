#include <torch/torch.h>

struct EmbeddingImpl : torch::nn::Module {
    EmbeddingImpl(int64_t vocab_size, int64_t embedding_dim)
        : embedding(torch::nn::Embedding(vocab_size, embedding_dim)) {
        register_module("embedding", embedding);
    }

    torch::Tensor forward(torch::Tensor x) {
        return embedding(x);
    }

    torch::nn::Embedding embedding;
};
