#include <torch/torch.h>

#ifndef ATTENTION_H
#define ATTENTION_H

struct AttentionImpl : public torch::nn::Module {
  AttentionImpl(int64_t d_model, int64_t n_heads, int64_t d_head,
                int64_t rotary_dim, int64_t rotary_base, int64_t n_ctx);
  std::pair<torch::Tensor, torch::Tensor> calculate_sin_cos_rotary(
      int64_t rotary_dim, int64_t n_ctx, int64_t rotary_base);
  torch::Tensor rotate_every_two(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor x);
  torch::Tensor apply_causal_mask(torch::Tensor x);
  torch::Tensor forward(torch::Tensor x);

  int64_t d_model, n_heads, d_head, rotary_dim, n_ctx, rotary_base;
  torch::Tensor IGNORE, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k,
      bias_v, bias_o, sin, cos;
};

TORCH_MODULE(Attention);
#endif