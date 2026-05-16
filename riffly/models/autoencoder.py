import torch

from riffly.models.general import ModelInterface


# Define a new class for the decoder
class Decoder(ModelInterface):
    def __init__(self, columns, rows) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, columns * rows),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class AE(ModelInterface):
    def __init__(self, columns, rows) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(columns * rows, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
            torch.nn.Sigmoid(),
        )
        self.decoder = Decoder(columns, rows)

    def forward(self, x):
        x = x.reshape(-1, self.columns * self.rows)
        encoded = self.encoder(x)
        return self.decoder(encoded)
