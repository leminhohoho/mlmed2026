from ast import Module
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 4
epochs = 100
val_iter = 10
log_iter = 10
lr = 1e-4

train = np.loadtxt("./mitbih_train.csv", delimiter=",")
test = np.loadtxt("./mitbih_test.csv", delimiter=",")

X_train = train[:, :-1]
y_train = train[:, -1]

X_test = test[:, :-1]
y_test = test[:, -1]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


class Model(nn.Module):
    def __init__(self, d_model, labels):
        super().__init__()

        self.l1 = nn.Linear(d_model, d_model * 2)
        self.l2 = nn.Linear(d_model * 2, d_model * 2)
        self.l3 = nn.Linear(d_model * 2, labels)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        logits = self.l3(x)

        return logits


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    ECGDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    ECGDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

model = Model(187, 5)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []


def avg(arr):
    return sum(arr) / len(arr)


for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if batch_idx % log_iter == 0:
            print(
                f"Batch {batch_idx} - Train loss: {train_losses[-1]} - Average train loss: {avg(train_losses)}"
            )

    print(
        f"Epoch {epoch} - Train loss: {train_losses[-1]} - Average train loss: {avg(train_losses)}"
    )

    if epoch % val_iter and epoch != 0:
        model.eval()
        with torch.no_grad():
            x, y = next(iter(val_loader))

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            val_losses.append(loss.item())

            print(
                f"Val {epoch // val_iter} - Val loss: {val_losses[-1]} - Average val loss: {avg(val_losses)}"
            )
