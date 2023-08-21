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


LR = 0.0001

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.train()
optimizer.zero_grad()

logits = model(input_tensor)
y_preds = torch.argmax(logits.squeeze(1), dim=-1)
# Technically, this labels should come from dataset
y_labels = torch.randint(0, OUTPUT_CLASSES, (BATCH_SIZE,), device=device)

y_preds = y_preds.float().clone().detach().requires_grad_(True)
y_labels = y_labels.float()

loss = criterion(y_preds, y_labels)

loss.backward()
optimizer.step()

print('loss', loss.item())
print(f"acc (in this batch) {(y_preds == y_labels).sum().item()}/{len(y_preds)}")

