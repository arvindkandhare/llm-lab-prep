"""
Embedding demo for beginners.

Embeddings are the bridge between discrete token IDs and continuous vectors.
- Token ID (integer): 3
- Embedding vector (real numbers): [-0.15, 0.82, 0.34, -0.61]

Models work with these continuous vectors for computation and learning.
"""

import torch
import torch.nn as nn


# Configuration
vocab_size = 9           # Number of unique tokens in our vocabulary
embedding_dim = 4        # Dimension of each embedding vector

# Token IDs from our example sentence: "The dog is happy."
sentence_token_ids = [2, 3, 4, 5, 8]


# Step 1: Create an embedding layer.
# This layer has a learnable parameter: a vocab_size x embedding_dim matrix.
# Each row corresponds to the embedding vector for that token ID.
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

print("=" * 60)
print("Embedding Layer Setup")
print("=" * 60)
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Total embedding matrix shape: {embedding_layer.weight.shape}")
print()


# Step 2: Convert sentence token IDs into a tensor.
# Tensors are the data structure PyTorch works with.
token_ids_tensor = torch.tensor(sentence_token_ids)

print("=" * 60)
print("Token IDs")
print("=" * 60)
print(f"Token IDs: {sentence_token_ids}")
print(f"Token IDs as tensor: {token_ids_tensor}")
print(f"Tensor shape: {token_ids_tensor.shape}")
print()


# Step 3: Pass token IDs through the embedding layer.
# This performs a "lookup" operation:
# - Token ID 2 -> retrieves row 2 of the embedding matrix
# - Token ID 3 -> retrieves row 3 of the embedding matrix
# - And so on...
embeddings = embedding_layer(token_ids_tensor)

print("=" * 60)
print("Embedding Output")
print("=" * 60)
print(f"Output shape: {embeddings.shape}")
print(f"(sequence_length={len(sentence_token_ids)}, embedding_dim={embedding_dim})")
print()


# Step 4-6: Print embedding vectors for each token.
print("=" * 60)
print("Embedding Vector for Each Token")
print("=" * 60)

for i, token_id in enumerate(sentence_token_ids):
    embedding_vector = embeddings[i]
    print(f"Token ID {token_id}: {embedding_vector.tolist()}")

print()


# Bonus: Show the full embedding matrix.
print("=" * 60)
print("Full Embedding Matrix (all token embeddings)")
print("=" * 60)
print("Each row is the embedding vector for that token ID:")
print(embedding_layer.weight)
print()

# Show what it looks like:
print("Row 0 (token <UNK>):  ", embedding_layer.weight[0].tolist())
print("Row 2 (token 'the'):  ", embedding_layer.weight[2].tolist())
print("Row 3 (token 'dog'):  ", embedding_layer.weight[3].tolist())
