"""
A simple tokenizer demo for beginners.

Tokenization is the first step in NLP:
- Convert text into numbers (token IDs) that models can understand.
- The vocabulary is the list of all known words.
"""

# Step 1: Create a small vocabulary dictionary.
# This maps each word to a unique integer ID.
vocabulary = {
    "<UNK>": 0,      # Unknown token for words not in vocabulary
    "<PAD>": 1,      # Padding token (used when batching sequences of different lengths)
    "the": 2,
    "dog": 3,
    "is": 4,
    "happy": 5,
    "cat": 6,
    "and": 7,
    ".": 8,
}

# Create a reverse mapping: ID -> word (useful for decoding)
id_to_word = {id: word for word, id in vocabulary.items()}


def tokenize(sentence: str) -> list:
    """
    Convert a sentence into token IDs.
    
    Args:
        sentence: The input text to tokenize.
    
    Returns:
        A list of token IDs.
    """
    # Convert to lowercase and split by spaces.
    words = sentence.lower().split()
    
    # Map each word to its ID, or use <UNK> if not found.
    token_ids = []
    for word in words:
        # Remove punctuation for this simple demo (except '.')
        # In real tokenizers, punctuation handling is more sophisticated.
        if word.endswith("."):
            # Handle words with trailing periods.
            word_without_period = word[:-1]
            token_id = vocabulary.get(word_without_period, vocabulary["<UNK>"])
            token_ids.append(token_id)
            # Add the period as a separate token.
            token_ids.append(vocabulary.get(".", vocabulary["<UNK>"]))
        else:
            token_id = vocabulary.get(word, vocabulary["<UNK>"])
            token_ids.append(token_id)
    
    return token_ids


def detokenize(token_ids: list) -> str:
    """
    Convert token IDs back into text.
    
    Args:
        token_ids: A list of token IDs.
    
    Returns:
        The reconstructed text.
    """
    # Map each ID back to a word.
    words = [id_to_word.get(id, "<UNK>") for id in token_ids]
    
    # Join words with spaces (ignoring padding tokens).
    text = " ".join([w for w in words if w != "<PAD>"])
    
    return text


# Example 1: A sentence with all known words.
print("=" * 60)
print("Example 1: All known words")
print("=" * 60)

sentence1 = "The dog is happy."
print(f"Original sentence: {sentence1}")

# Tokenize the sentence.
token_ids1 = tokenize(sentence1)
print(f"Token IDs: {token_ids1}")

# Print the mapping for each token.
print("Token mapping:")
for word, token_id in zip(sentence1.lower().split(), token_ids1):
    if word.endswith("."):
        word_without_period = word[:-1]
        print(f"  {word_without_period} -> {vocabulary.get(word_without_period, vocabulary['<UNK>'])}")
        print(f"  . -> {vocabulary.get('.', vocabulary['<UNK>'])}")
    else:
        print(f"  {word} -> {vocabulary.get(word, vocabulary['<UNK>'])}")

# Detokenize back to text.
reconstructed1 = detokenize(token_ids1)
print(f"Reconstructed: {reconstructed1}")


# Example 2: A sentence with an unknown word.
print("\n" + "=" * 60)
print("Example 2: Unknown word (dragon)")
print("=" * 60)

sentence2 = "The dragon is happy."
print(f"Original sentence: {sentence2}")

# Tokenize the sentence.
token_ids2 = tokenize(sentence2)
print(f"Token IDs: {token_ids2}")

# Print the mapping for each token.
print("Token mapping:")
for word in sentence2.lower().split():
    if word.endswith("."):
        word_without_period = word[:-1]
        token_id = vocabulary.get(word_without_period, vocabulary["<UNK>"])
        print(f"  {word_without_period} -> {token_id} ('{id_to_word.get(token_id, '<UNK>')}')")
        print(f"  . -> {vocabulary.get('.', vocabulary['<UNK>'])}")
    else:
        token_id = vocabulary.get(word, vocabulary["<UNK>"])
        print(f"  {word} -> {token_id} ('{id_to_word.get(token_id, '<UNK>')}')")

# Detokenize back to text.
reconstructed2 = detokenize(token_ids2)
print(f"Reconstructed: {reconstructed2}")
print("Note: 'dragon' was converted to <UNK> because it's not in the vocabulary.")
