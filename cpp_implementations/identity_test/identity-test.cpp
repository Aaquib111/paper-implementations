#include <torch/torch.h>
#include <iostream>

int main(){
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;

    // Random tensor of shape (2, 5, 3)
    torch::Tensor tensor2 = torch::randn({2, 5, 3});
    // Random tensor of shape (10, 7, 5)
    torch::Tensor tensor3 = torch::randn({10, 7, 5});

    // Matmul into result of shape (10, 7, 2, 3)
    torch::Tensor result = torch::mat(tensor3.unsqueeze(0), tensor2);
    std::cout << result.sizes() << std::endl;
}