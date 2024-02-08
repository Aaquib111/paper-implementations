#include <torch/torch.h>
#include <iostream>

int main(){
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;

    // Define tensors
    torch::Tensor pattern = torch::randn({10, 12, 5, 6});
    torch::Tensor v = torch::randn({10, 6, 5, 32});

    // // Reshape or permute resid_stream for matrix multiplication: (Batch * Seq_len * n_heads, d_model)
    // auto reshaped_resid_stream = resid_stream.view({-1, 128});
    // std::cout << reshaped_resid_stream.sizes() << std::endl;

    // // Perform matrix multiplication
    // // As there's no direct einsum in C++ API, we manually align dimensions and perform bmm or matmul as necessary.
    // // For our case, since we reshaped resid_stream, we can directly use matmul if we reshape key_projection accordingly.
    // auto reshaped_key_projection = key_projection.reshape({12*128, 64});
    // std::cout << reshaped_key_projection.sizes() << std::endl;
    // torch::Tensor key_vector = torch::matmul(reshaped_resid_stream, reshaped_key_projection);

    // // Reshape the result back to (batch, seq_pos, n_heads, d_head)
    // torch::Tensor result = key_vector.view({10, 3, 12, 64});

    torch::Tensor result = torch::matmul(pattern, v.permute({0, 3, 2, 1}));
    std::cout << result.sizes() << std::endl;
}