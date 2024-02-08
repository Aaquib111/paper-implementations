#include "attention.hpp"

// Attention with rotary positional encoding and KV caching
AttentionImpl::AttentionImpl(int64_t d_model, int64_t n_heads, int64_t d_head,
                             int64_t rotary_dim, int64_t rotary_base,
                             int64_t n_ctx)
    : d_model(d_model),
      n_heads(n_heads),
      d_head(d_head),
      rotary_dim(rotary_dim),
      n_ctx(n_ctx),
      rotary_base(rotary_base),
      IGNORE(torch::tensor(-std::numeric_limits<float>::infinity())
                 .to(torch::kFloat32)) {
  // Register weight tensors
  weight_q = register_parameter("weight_q",
                                torch::nn::init::xavier_uniform_(
                                    torch::empty({n_heads, d_model, d_head})));
  weight_k = register_parameter("weight_k",
                                torch::nn::init::xavier_uniform_(
                                    torch::empty({n_heads, d_model, d_head})));
  weight_v = register_parameter("weight_v",
                                torch::nn::init::xavier_uniform_(
                                    torch::empty({n_heads, d_model, d_head})));
  weight_o = register_parameter("weight_o",
                                torch::nn::init::xavier_uniform_(
                                    torch::empty({n_heads, d_head, d_model})));

  // Register bias tensors
  bias_q = register_parameter(
      "bias_q", torch::nn::init::zeros_(torch::empty({n_heads, d_head})));
  bias_k = register_parameter(
      "bias_k", torch::nn::init::zeros_(torch::empty({n_heads, d_head})));
  bias_v = register_parameter(
      "bias_v", torch::nn::init::zeros_(torch::empty({n_heads, d_head})));
  bias_o = register_parameter("bias_o",
                              torch::nn::init::zeros_(torch::empty({d_model})));

  // Register ignore buffers
  register_buffer("IGNORE", IGNORE);

  // Rotary positional encoding
  auto pair = calculate_sin_cos_rotary(rotary_dim, n_ctx, rotary_base);
  sin = pair.first;
  cos = pair.second;

  register_buffer("rotary_sin", sin);
  register_buffer("rotary_cos", cos);
}

std::pair<torch::Tensor, torch::Tensor> AttentionImpl::calculate_sin_cos_rotary(
    int64_t rotary_dim, int64_t n_ctx, int64_t rotary_base) {
  auto pos = torch::arange(n_ctx).to(torch::kFloat32);
  auto dim = torch::arange(rotary_dim / 2).to(torch::kFloat32);

  auto freq = torch::pow(rotary_base, dim / rotary_dim / 2.0);
  freq = torch::cat({freq, freq}, 0);

  auto angles = pos.unsqueeze(-1) * freq.unsqueeze(0);

  return {torch::sin(angles).to(torch::kFloat32),
          torch::cos(angles).to(torch::kFloat32)};
}

torch::Tensor AttentionImpl::rotate_every_two(torch::Tensor x) {
  // Use GPT-NeoX style rotation (folding full length in half)
  auto rot_x = x.clone();
  auto n = x.size(-1) / 2;
  rot_x.slice(-1, 0, n) = -x.slice(-1, n, rotary_dim);
  rot_x.slice(-1, n, rotary_dim) = x.slice(-1, 0, n);

  return rot_x;
}

torch::Tensor AttentionImpl::apply_rotary(torch::Tensor x) {
  auto x_pos = x.size(1);
  auto x_rot = x.slice(-1, 0, rotary_dim);
  auto x_pass = x.slice(-1, rotary_dim, d_head);
  auto x_flip = rotate_every_two(x_rot);

  // 1, x_pos, 1, rotary_dim
  auto curr_rotary_cos = cos.slice(0, 0, x_pos).unsqueeze(0).unsqueeze(-2);

  // 1, x_pos, 1, rotary_dim
  auto curr_rotary_sin = sin.slice(0, 0, x_pos).unsqueeze(0).unsqueeze(-2);

  // 1, x_pos, n_heads, d_head
  auto x_rotated = x_rot * curr_rotary_cos + x_flip * curr_rotary_sin;

  return torch::cat({x_rotated, x_pass}, -1);
}

torch::Tensor AttentionImpl::apply_causal_mask(torch::Tensor x) {
  auto mask =
      torch::triu(torch::ones({x.size(-2), x.size(-1)}), 1).to(torch::kBool);
  return x.masked_fill(mask, IGNORE);
}

torch::Tensor AttentionImpl::forward(torch::Tensor x) {
  // Query vector, shape (batch, seq_pos, n_heads, d_head)
  torch::Tensor q =
      torch::matmul(x.unsqueeze(1), weight_q).permute({0, 2, 1, 3}) + bias_q;

  // Key vector, shape (batch, seq_pos, n_heads, d_head)
  torch::Tensor k =
      torch::matmul(x.unsqueeze(1), weight_k).permute({0, 2, 1, 3}) + bias_k;

  // Apply RoPE to q and k
  q = apply_rotary(q);
  k = apply_rotary(k);

  // Calculate attention scores = qk^T, of shape (batch, n_heads, seq_pos,
  // seq_pos)
  torch::Tensor attn_scores =
      torch::matmul(q, k.permute({0, 1, 3, 2})) / std::sqrt(d_head);

  // Apply causal mask to attention scores
  attn_scores = apply_causal_mask(attn_scores);

  // Apply softmax to attention scores to get attention pattern
  torch::Tensor attn_pattern = attn_scores.softmax(-1);
  attn_pattern = torch::where(attn_pattern.isnan(),
                              torch::zeros_like(attn_pattern), attn_pattern);

  // Value vector, shape (batch, seq_pos, n_heads, d_head)
  torch::Tensor v =
      torch::matmul(x.unsqueeze(1), weight_v).permute({0, 2, 1, 3}) + bias_v;

  // Attention output before applying weight_o, shape (batch, query_pos,
  // n_heads, d_head)
  torch::Tensor z = torch::matmul(attn_pattern, v);

  // Pass through weight_o to get final attention output
  torch::Tensor attn_out = torch::matmul(z.reshape({z.size(0), z.size(1), -1}),
                                         weight_o.reshape({-1, d_model})) +
                           (bias_o / n_heads);

  return attn_out;
}