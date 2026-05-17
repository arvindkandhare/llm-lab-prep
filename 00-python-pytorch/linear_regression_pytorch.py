import torch

# We will create fake training data from a simple line:
# y = 2x + 1
true_w = 1.0
true_b = 2.0

# x is our input data.
# Each row has one input value.
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

# y is the correct answer for each x value.
y = true_w * x + true_b

# These are the values the model will learn.
# We start with random guesses.
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# The learning rate controls how big each update step is.
learning_rate = 0.01

# Train for a fixed number of steps.
for step in range(1000):
    # Make a prediction using the current guesses for w and b.
    prediction = w * x + b

    # Mean squared error:
    # 1. Find the difference between prediction and correct answer.
    # 2. Square the differences so negative errors count as positive.
    # 3. Take the average.
    loss = ((prediction - y) ** 2).mean()

    # Calculate gradients for w and b.
    loss.backward()

    # Update w and b without tracking this update in autograd.
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Clear old gradients before the next training step.
        w.grad.zero_()
        b.grad.zero_()

    # Print progress every 100 steps.
    if step % 100 == 0:
        print(f"step: {step}, loss: {loss.item():.6f}")

print()
print("True equation:    y = 2x + 1")
print(f"Learned equation: y = {w.item():.4f}x + {b.item():.4f}")
