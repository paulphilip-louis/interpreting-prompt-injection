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
