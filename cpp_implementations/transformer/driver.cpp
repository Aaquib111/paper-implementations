#include <torch/torch.h>
#include "components/transformer.hpp"
#include <iostream>

int main() {
    int64_t vocab_size = 10000;
    int64_t d_model = 512;
    int64_t n_heads = 8;
    int64_t d_mlp = 2048;
    int64_t d_head = 64;
    int64_t rotary_dim = 64;
    int64_t n_ctx = 512;
    int64_t rotary_base = 10000;
    int64_t num_blocks = 4;

    Transformer transformer(
        vocab_size, 
        d_model, 
        d_mlp, 
        n_heads, 
        d_head, 
        rotary_dim, 
        rotary_base, 
        n_ctx, 
        num_blocks
    );
    torch::Tensor input_ids = torch::randint(0, vocab_size, {32, 10});
    
    auto output = transformer->forward(input_ids);
    std::cout << output.sizes() << std::endl;

    return 0;
}