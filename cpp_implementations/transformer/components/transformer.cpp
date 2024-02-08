#include "transformer.hpp"

TransformerImpl::TransformerImpl(int64_t vocab_size, int64_t d_model,
                                 int64_t d_mlp, int64_t n_heads, int64_t d_head,
                                 int64_t rotary_dim, int64_t rotary_base,
                                 int64_t n_ctx, int64_t num_blocks)
    : embed(Embedding(vocab_size, d_model)),
      final_norm(LayerNorm(d_model, 1e-5)),
      unembed(Unembedding(d_model, vocab_size)) {
  // Initialize transformer blocks
  for (int i = 0; i < num_blocks; ++i) {
    auto block = TransformerBlock(d_model, d_mlp, n_heads, d_head, rotary_dim,
                                  rotary_base, n_ctx, true);
    blocks.push_back(block);
    register_module("block" + std::to_string(i), block);
  }
  register_module("embed", embed);
  register_module("final_norm", final_norm);
  register_module("unembed", unembed);
}

torch::Tensor TransformerImpl::forward(torch::Tensor x) {
  x = embed->forward(x);
  for (auto& block : blocks) {
    x = block->forward(x);
  }
  x = final_norm->forward(x);
  x = unembed->forward(x);
  return x;
}