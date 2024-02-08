#include <torch/torch.h>

#include "attention.hpp"
#include "layer_norm.hpp"
#include "multi_layer_perceptron.hpp"

#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

struct TransformerBlockImpl : public torch::nn::Module {
  TransformerBlockImpl(int64_t d_model, int64_t d_mlp, int64_t n_heads,
                       int64_t d_head, int64_t rotary_dim, int64_t rotary_base,
                       int64_t n_ctx);
  torch::Tensor forward(torch::Tensor x);

  Attention self_attn;
  LayerNorm norm1, norm2;
  MLP mlp;
};

TORCH_MODULE(TransformerBlock);

#endif