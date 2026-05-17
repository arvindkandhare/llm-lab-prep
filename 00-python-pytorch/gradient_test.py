import torch

# Create a 1x2 tensor.
# requires_grad=True tells PyTorch to track operations on this value.
x = torch.tensor([[3.0, 4.0],[5,6]], requires_grad=True)

# Build a simple equation:
# y = x^2 + 2x + 1  (applied element-wise)
y = x * x + 2 * x + 1

# sum() reduces to a scalar so backward() can run
y.sum().backward()

print("x:", x)
print("y:", y)
print("gradient:", x.grad)
