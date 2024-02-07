#include <torch/torch.h>
#include <cmath>
#include <iostream>

// Embedding Module

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

// LayerNorm

struct LayerNormImpl : torch::nn::Module {
    LayerNormImpl(int64_t d_model, double eps = 1e-5) 
        : eps(eps) {
        weight = register_parameter("weight", torch::ones({d_model}));
        bias = register_parameter("bias", torch::zeros({d_model}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto mean = x.mean(-1, true);
        auto std = (x - mean).pow(2).mean(-1, true).sqrt();
        return weight * (x - mean) / (std + eps) + bias;
    }

    torch::Tensor weight, bias;
    double eps;
};

// MLP

struct MLPImpl : torch::nn::Module {
    MLPImpl(int64_t d_model, int64_t d_mlp)
        : fc1(register_module("fc1", torch::nn::Linear(d_model, d_mlp))),
          fc2(register_module("fc2", torch::nn::Linear(d_mlp, d_model))) {}
    
    torch::Tensor gelu(torch::Tensor x) {
        // Implementation of GeLU used by GPT2, Pythia, etc
        return 0.5 * x * (1.0 + torch::tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x.pow(3))));
    }

    torch::Tensor forward(torch::Tensor x) {
        return fc2(gelu(fc1(x)));
    }

    torch::nn::Linear fc1, fc2;
};


// Attention with rotary positional encoding and KV caching

struct AttentionImpl : torch::nn::Module {
    AttentionImpl(int64_t d_model, int64_t n_heads, int64_t d_head, int64_t rotary_dim, int64_t n_ctx, int64_t rotary_base)
        : d_model(d_model), n_heads(n_heads), d_head(d_head), rotary_dim(rotary_dim), n_ctx(n_ctx), rotary_base(rotary_base),
          IGNORE(torch::tensor(-std::numeric_limits<float>::infinity()).to(torch::kFloat32)) {
            weight_q = register_parameter("weight_q", torch::nn::init::xavier_uniform_(torch::empty({n_heads, d_model, d_head})));
            weight_k = register_parameter("weight_k", torch::nn::init::xavier_uniform_(torch::empty({n_heads, d_model, d_head})));
            weight_v = register_parameter("weight_v", torch::nn::init::xavier_uniform_(torch::empty({n_heads, d_model, d_head})));
            weight_o = register_parameter("weight_o", torch::nn::init::xavier_uniform_(torch::empty({n_heads, d_head, d_model})));

            bias_q = register_parameter("bias_q", torch::nn::init::zeros_(torch::empty({n_heads, d_head})));
            bias_k = register_parameter("bias_k", torch::nn::init::zeros_(torch::empty({n_heads, d_head})));
            bias_v = register_parameter("bias_v", torch::nn::init::zeros_(torch::empty({n_heads, d_head})));
            bias_o = register_parameter("bias_o", torch::nn::init::zeros_(torch::empty({d_model})));

            // Register ignore buffers
            register_buffer("IGNORE", IGNORE);

            // Rotary positional encoding
            auto pair = calculate_sin_cos_rotary(rotary_dim, n_ctx, rotary_base);
            sin = pair.first;
            cos = pair.second;

            register_buffer("rotary_sin", sin);
            register_buffer("rotary_cos", cos);
        }

    std::pair<torch::Tensor, torch::Tensor> calculate_sin_cos_rotary(int64_t rotary_dim, int64_t n_ctx, int64_t rotary_base) {
        auto pos = torch::arange(n_ctx).to(torch::kFloat32);
        auto dim = torch::arange(rotary_dim / 2).to(torch::kFloat32);

        auto freq = torch::pow(rotary_base, dim / rotary_dim / 2.0);
        freq = torch::cat({freq, freq}, 0);

        auto angles = pos.unsqueeze(-1) * freq.unsqueeze(0);

        return {torch::sin(angles).to(torch::kFloat32), torch::cos(angles).to(torch::kFloat32)};
    }

    torch::Tensor rotate_every_two(torch::Tensor x) {
        auto rot_x = x.clone();
        auto n = x.size(-1);
        rot_x.slice(-1, 0, n) = -x.slice(-1, n, rotary_dim);
        rot_x.slice(-1, n, rotary_dim) = x.slice(-1, 0, n);

        return rot_x;
    }

    torch::Tensor apply_rotary(torch::Tensor x) {
        auto x_pos = x.size(1);
        auto x_rot = x.slice(-1, 0, rotary_dim);
        auto x_pass = x.slice(-1, rotary_dim, d_head);
        auto x_flip = rotate_every_two(x_rot);

        auto curr_rotary_cos = cos.slice(0, 0, x_pos).unsqueeze(0).unsqueeze(-2); // 1, x_pos, 1, rotary_dim
        auto curr_rotary_sin = sin.slice(0, 0, x_pos).unsqueeze(0).unsqueeze(-2); // 1, x_pos, 1, rotary_dim
        auto x_rotated = x_rot * curr_rotary_cos + x_flip * curr_rotary_sin; // 1, x_pos, n_heads, d_head

        return torch::cat({x_rotated, x_pass}, -1);
    }

    torch::Tensor apply_causal_mask(torch::Tensor x) {
        auto mask = torch::triu(torch::ones({x.size(-2), x.size(-1)}), 1).to(torch::kBool);
        return x.masked_fill(mask, IGNORE);
    }

    torch::Tensor forward(torch::Tensor x) {
        
    }

    torch::Tensor weight_q, weight_k, weight_v, weight_o;
    torch::Tensor bias_q, bias_k, bias_v, bias_o, IGNORE;
    torch::Tensor sin, cos;
    int64_t d_model, n_heads, d_head, rotary_dim, n_ctx, rotary_base;
};

// Unembed

struct UnembedImpl : torch::nn::Module {
    UnembedImpl(int64_t d_model, int64_t vocab_size)
        : fc(register_module("fc", torch::nn::Linear(d_model, vocab_size))) {}

    torch::Tensor forward(torch::Tensor x) {
        return fc(x);
    }

    torch::nn::Linear fc;
};