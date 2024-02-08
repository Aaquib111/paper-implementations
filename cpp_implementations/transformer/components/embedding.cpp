#include "embedding.hpp"

EmbeddingImpl::EmbeddingImpl(int64_t vocab_size, int64_t embedding_dim)
    : embedding(torch::nn::Embedding(vocab_size, embedding_dim)) {
  register_module("embedding", embedding);
}

torch::Tensor EmbeddingImpl::forward(torch::Tensor x) { return embedding(x); }