import torch
import numpy as np

# emulate sigmoid logic
x = np.random.rand(10).astype(np.float32)
y = 1 / (1 + np.exp(-x))
print("Numpy:", y)
