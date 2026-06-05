import torch
import gc # Import garbage collector

def clear_cuda_cache():
    """Clears the CUDA memory cache and runs garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA memory cache cleared.")
    else:
        print("CUDA is not available.")
    gc.collect() # Explicitly run Python's garbage collector
    print("Python garbage collector ran.")

def to_first_token_ids(model, strings):
    """
    Small function that returns the first token id of strings
    Input:
    - strings: List[str]
    Output:
    - ids: List[int]

    Example:
    ["spam", "not spam"] -> ["spam", "not "] (first tokens) -> [3241, 124]
    """
    ids = set()
    for s in strings:
        toks = model.tokenizer.encode(s, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return sorted(ids)

def cosine_similarity(t1, t2):
  assert t1.shape == t2.shape
  return torch.dot(t1, t2).item()/(t1.norm()*t2.norm()).item()

