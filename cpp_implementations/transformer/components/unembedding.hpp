#include <torch/torch.h>

#ifndef UNEMBEDDING_H
#define UNEMBEDDING_H

struct UnembeddingImpl : public torch::nn::Module {
  UnembeddingImpl(int64_t d_model, int64_t vocab_size);
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc;
};

TORCH_MODULE(Unembedding);

#endif