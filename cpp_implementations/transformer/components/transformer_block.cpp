#include "transformer_block.hpp"

TransformerBlockImpl::TransformerBlockImpl(int64_t d_model, int64_t d_mlp,
                                           int64_t n_heads, int64_t d_head,
                                           int64_t rotary_dim, int64_t rotary_base, 
                                           int64_t n_ctx, bool use_cache)
    : self_attn(
          Attention(d_model, n_heads, d_head, rotary_dim, rotary_base, n_ctx, use_cache)),
      norm1(LayerNorm(d_model, 1e-5)),
      norm2(LayerNorm(d_model, 1e-5)),
      mlp(MLP(d_model, d_mlp)) {
  register_module("self_attn", self_attn);
  register_module("norm1", norm1);
  register_module("norm2", norm2);
  register_module("mlp", mlp);
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
  auto attn_output = self_attn->forward(norm1->forward(x));
  x = x + attn_output;
  auto mlp_output = mlp->forward(norm2->forward(x));
  x = x + mlp_output;
  return x;
}