import torch
import torch.nn as nn

# A token embedding tells the model which token it is.
# A positional embedding tells the model where that token appears in the sequence.
# Transformers usually add these two vectors together.

vocab_size = 9
embedding_dim = 4
max_sequence_length = 10
sentence_token_ids = [2, 3, 4, 5, 8]
position_ids = [0, 1, 2, 3, 4]

# Step 1: Create an embedding table for tokens.
# This stores one learnable vector for each token in the vocabulary.
token_embedding = nn.Embedding(vocab_size, embedding_dim)

# Step 2: Create an embedding table for positions.
# This stores one learnable vector for each position index.
positional_embedding = nn.Embedding(max_sequence_length, embedding_dim)

# Step 3: Convert token IDs into a PyTorch tensor.
token_ids_tensor = torch.tensor(sentence_token_ids)

# Step 4: Convert position IDs into a PyTorch tensor.
position_ids_tensor = torch.tensor(position_ids)

# Step 5: Look up token embeddings.
# Shape: [sequence_length, embedding_dim]
token_vectors = token_embedding(token_ids_tensor)

# Step 6: Look up positional embeddings.
# Shape: [sequence_length, embedding_dim]
position_vectors = positional_embedding(position_ids_tensor)

# Step 7: Add token embeddings and positional embeddings.
# This gives the model both token identity and token position.
combined_vectors = token_vectors + position_vectors

# Step 8: Print token IDs.
print("Token IDs:", sentence_token_ids)

# Step 9: Print position IDs.
print("Position IDs:", position_ids)
print()

# Step 10: Print token embedding shape.
print("Token embedding shape:", token_vectors.shape)

# Step 11: Print positional embedding shape.
print("Positional embedding shape:", position_vectors.shape)

# Step 12: Print combined embedding shape.
print("Combined embedding shape:", combined_vectors.shape)
print()

# Step 13: Print one example from each embedding type.
example_index = 0
print(f"Example token ID: {sentence_token_ids[example_index]}")
print(f"Example position ID: {position_ids[example_index]}")
print("Token embedding:", token_vectors[example_index])
print("Position embedding:", position_vectors[example_index])
print("Combined embedding:", combined_vectors[example_index])
print()

# Print all combined vectors so it is easy to see one row per token position.
print("Combined embeddings for the full sentence:")
for index, vector in enumerate(combined_vectors):
    print(f"Position {index}, token {sentence_token_ids[index]}: {vector}")
