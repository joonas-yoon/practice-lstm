import torch
import torch.nn.functional as F
import numpy as np

from model import LSTMClassifier


SEQUENCE_LEN = 1000
BATCH_SIZE = 64
HIDDEN_DIM = 128
OUTPUT_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_tensor = torch.from_numpy(np.random.randn(BATCH_SIZE, SEQUENCE_LEN, 6))
input_tensor = input_tensor.float().to(device)

model = LSTMClassifier(hidden_size=HIDDEN_DIM,
                       output_size=OUTPUT_CLASSES,
                       device=device)

output_logit = model(input_tensor)

print('logit', output_logit)
print('logit', output_logit.shape)

probability = F.softmax(output_logit, dim=1)
print('probability', probability, probability.shape)

