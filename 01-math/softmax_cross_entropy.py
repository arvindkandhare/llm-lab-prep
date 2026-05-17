import torch
import torch.nn.functional as F


# Imagine a model is choosing the next token from 4 possible tokens.
tokens = ["cat", "dog", "pizza", "tree"]

# The target token is always "cat" in this exercise.
# Index 0 corresponds to "cat" in the tokens list.
target_index = torch.tensor(0)
target_token = tokens[target_index.item()]

# Three different prediction quality cases.
# - good_prediction: model gives "cat" the highest score
# - bad_prediction: model gives other tokens higher scores
# - uncertain_prediction: model gives all tokens equal scores
cases = {
    "good_prediction": torch.tensor([2.0, 1.0, 0.1, -1.0]),
    "bad_prediction": torch.tensor([0.1, 3.0, 2.0, 1.0]),
    "uncertain_prediction": torch.tensor([1.0, 1.0, 1.0, 1.0]),
}


def print_case_result(case_name: str, logits: torch.Tensor) -> None:
    # Softmax converts raw logits into probabilities that sum to 1.
    probabilities = F.softmax(logits, dim=0)

    # cross_entropy expects a batch shape:
    # logits: [batch_size, number_of_classes]
    # targets: [batch_size]
    # unsqueeze(0) adds a batch dimension of size 1.
    loss = F.cross_entropy(logits.unsqueeze(0), target_index.unsqueeze(0))

    # 1. Case name
    print(f"\nCase: {case_name}")

    # 2. Logits
    print("Logits:")
    for token, score in zip(tokens, logits):
        print(f"  {token}: {score.item():.2f}")

    # 3. Softmax probabilities
    print("Softmax probabilities:")
    for token, probability in zip(tokens, probabilities):
        print(f"  {token}: {probability.item():.4f}")

    # 4. Target token probability
    target_probability = probabilities[target_index].item()
    print(f"Target token: {target_token}")
    print(f"Target token probability: {target_probability:.4f}")

    # 5. Cross-entropy loss
    # Lower loss means the model assigned higher probability to the target token.
    print(f"Cross-entropy loss: {loss.item():.4f}")

# Run all three cases and print the requested outputs.
for name, case_logits in cases.items():
    print_case_result(name, case_logits)
