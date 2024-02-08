#include <torch/torch.h>

#include "embedding.hpp"
#include "layer_norm.hpp"
#include "multi_layer_perceptron.hpp"
#include "transformer_block.hpp"
#include "unembedding.hpp"

#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

struct TransformerImpl : public torch::nn::Module {
  TransformerImpl(int64_t vocab_size, int64_t d_model, int64_t d_mlp,
                  int64_t n_heads, int64_t d_head, int64_t rotary_dim,
                  int64_t rotary_base, int64_t n_ctx, int64_t num_blocks);

  torch::Tensor forward(torch::Tensor x);

  Embedding embed;
  std::vector<TransformerBlock> blocks;
  LayerNorm final_norm;
  Unembedding unembed;
};

TORCH_MODULE(Transformer);
#endif