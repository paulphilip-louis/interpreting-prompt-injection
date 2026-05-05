import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")