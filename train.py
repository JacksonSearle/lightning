import os
import time
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

start = time.time()

# small model
# encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
# decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# big model
encoder = nn.Sequential(
    nn.Linear(28 * 28, 128),  # Increase to 128 neurons
    nn.ReLU(),
    nn.Linear(128, 2048),      # Add another layer with 256 neurons
    nn.ReLU(),
    nn.Linear(2048, 128),      # Add another layer stepping down
    nn.ReLU(),
    nn.Linear(128, 3)
)

decoder = nn.Sequential(
    nn.Linear(3, 128),       # Increase to 128 neurons
    nn.ReLU(),
    nn.Linear(128, 2048),     # Add another layer with 256 neurons
    nn.ReLU(),
    nn.Linear(2048, 128),     # Step down through another 128 neuron layer
    nn.ReLU(),
    nn.Linear(128, 28 * 28)  # Output layer size remains the same
)


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, num_workers=16)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=10, accelerator='gpu', devices='auto', strategy="ddp")
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

end = time.time()
print("Total time:", end - start)
