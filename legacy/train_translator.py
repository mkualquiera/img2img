# This trains the model that converts FrozenCLIPImageEmbedder latent
# representations to FrozenCLIPTextEmbedder latent representations.

import os
import random
import time

import joblib
import torch
from mixer import MLPMixerBlock
from tqdm import tqdm

import wandb

# set random seeds
random.seed(42)


class TranslatorDataset:
    def __init__(self, files: list[str], batch_size: int):
        self.files = files
        self.batch_size = batch_size

    @staticmethod
    def from_path(path: str, batch_size: int):
        # Get all joblib files recursively using walk
        joblibs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".joblib"):
                    joblibs.append(os.path.join(root, file))

        # shuffle the files
        random.shuffle(joblibs)

        return TranslatorDataset(joblibs, batch_size)

    def split(self, percent: float) -> tuple["TranslatorDataset", "TranslatorDataset"]:
        split_index = int(len(self.files) * percent)
        return (
            TranslatorDataset(self.files[:split_index], self.batch_size),
            TranslatorDataset(self.files[split_index:], self.batch_size),
        )

    # iterator
    def __iter__(self):
        return TranslatorDatasetIterator(self)


class TranslatorDatasetIterator:
    def __init__(self, dataset: TranslatorDataset):
        self.dataset = dataset
        self.index = 0
        self.remainder: None | dict[str, torch.Tensor] = None

    def __iter__(self):
        return self

    def __next__(self):
        while (
            self.remainder is None
            or len(self.remainder["image_embeddings"]) < self.dataset.batch_size
        ):
            curr_file = self.dataset.files[self.index]

            tstart = time.time()

            loaded = joblib.load(curr_file)

            tend = time.time()

            if (
                loaded["image_embeddings"].shape[0]
                != loaded["text_embeddings"].shape[0]
            ):
                print("WARNING: Mismatched shapes in", curr_file)
                self.index += 1
                continue

            print(f"Loaded {curr_file} in {tend - tstart} seconds")

            if self.remainder is not None:
                # Concatenate loaded to remainder
                self.remainder["image_embeddings"] = torch.cat(
                    [
                        self.remainder["image_embeddings"],
                        loaded["image_embeddings"].to("cpu"),
                    ],
                    dim=0,
                )
                self.remainder["text_embeddings"] = torch.cat(
                    [
                        self.remainder["text_embeddings"],
                        loaded["text_embeddings"].to("cpu"),
                    ],
                    dim=0,
                )
            else:
                self.remainder = loaded
                self.remainder["image_embeddings"] = self.remainder[
                    "image_embeddings"
                ].to("cpu")
                self.remainder["text_embeddings"] = self.remainder[
                    "text_embeddings"
                ].to("cpu")

            self.index += 1

            if self.index >= len(self.dataset.files):
                self.index = 0

        # Get the batch
        batch = {
            "image_embeddings": self.remainder["image_embeddings"][
                : self.dataset.batch_size
            ],
            "text_embeddings": self.remainder["text_embeddings"][
                : self.dataset.batch_size
            ],
        }

        # Remove the batch from remainder
        self.remainder["image_embeddings"] = self.remainder["image_embeddings"][
            self.dataset.batch_size :
        ]
        self.remainder["text_embeddings"] = self.remainder["text_embeddings"][
            self.dataset.batch_size :
        ]

        return batch


# First model attempt
# Basically just two linear layers with relu in between
class TranslatorModel(torch.nn.Module):
    def __init__(
        self, image_embedder_dims: tuple[int, int], text_embedder_dims: tuple[int, int]
    ):
        super().__init__()
        self.first_stage = torch.nn.Linear(
            image_embedder_dims[1], text_embedder_dims[1]
        )
        self.second_stage = torch.nn.Linear(
            image_embedder_dims[0], text_embedder_dims[0]
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        # image_embeddings is (batch_size, image_embedder_dims[0], image_embedder_dims[1])
        x = self.first_stage(image_embeddings)
        # x is (batch_size, image_embedder_dims[0], text_embedder_dims[1])
        # relu
        x = torch.nn.functional.relu(x)
        # transpose x to (batch_size, text_embedder_dims[1], image_embedder_dims[0])
        x = x.transpose(1, 2)
        # x is (batch_size, text_embedder_dims[1], image_embedder_dims[0])
        x = self.second_stage(x)
        # x is (batch_size, text_embedder_dims[1], text_embedder_dims[0])
        # transpose x to (batch_size, text_embedder_dims[0], text_embedder_dims[1])
        x = x.transpose(1, 2)
        # x is (batch_size, text_embedder_dims[0], text_embedder_dims[1])
        return x


# Second model attempt
# It uses a number of MLPMixer blocks, which are basically just a bunch of linear layers
# with relu in between, and then a layer norm
class MixerTranslatorModel(torch.nn.Module):
    def __init__(
        self,
        image_embedder_dims: tuple[int, int],
        text_embedder_dims: tuple[int, int],
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.first_stage = torch.nn.Sequential(
            *[
                MLPMixerBlock(image_embedder_dims, hidden_size, hidden_size)
                for _ in range(num_layers)
            ]
        )
        self.second_stage = TranslatorModel(image_embedder_dims, text_embedder_dims)

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        # image_embeddings is (batch_size, image_embedder_dims[0], image_embedder_dims[1])
        x = self.first_stage(image_embeddings)
        # x is (batch_size, image_embedder_dims[0], image_embedder_dims[1])
        x = self.second_stage(x)
        # x is (batch_size, text_embedder_dims[0], text_embedder_dims[1])
        return x


def train(
    input_path: str,
    batch_size: int = 250,
    lr: float = 0.002,
    iters: int = 500 * 4 * 9,  # 6000 * 3,
    num_layers: int = 2,
    hidden_size: int = 128,
    gradient_accumulation: int = 1,
    resume_from: None | str = None,
):
    # Load the dataset
    dataset = TranslatorDataset.from_path(input_path, batch_size)
    train_dataset, val_dataset = dataset.split(0.9)
    train_dataset, val_dataset = iter(train_dataset), iter(val_dataset)

    # Load the model
    model = MixerTranslatorModel((257, 1024), (77, 768), hidden_size, num_layers)

    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from))

    model = model.to("cuda")

    # Load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, verbose=True, threshold=0.005
    )

    # Load the loss function
    mse_loss_fn = torch.nn.MSELoss()

    # Load the other loss function
    cos_loss_fn = torch.nn.CosineSimilarity(dim=2)

    wandb.init(
        project="mixer-translator",
        config={
            "input_path": input_path,
            "batch_size": batch_size,
            "model_class": str(model.__class__),
            "optimizer_class": str(optimizer.__class__),
            "lr": lr,
            "iters": iters,
            "layers": num_layers,
            "hidden_size": hidden_size,
            "gradient_accumulation": gradient_accumulation,
        },
    )

    wandb.define_metric("train_loss", step_metric="batch")
    wandb.define_metric("val_loss", step_metric="batch")

    # Train
    for iteration in tqdm(range(iters + 1)):
        # Get the batch
        batch = next(train_dataset)

        # Move to cuda
        batch["image_embeddings"] = batch["image_embeddings"].to("cuda")
        batch["text_embeddings"] = batch["text_embeddings"].to("cuda")

        # Get the predictions
        predictions = model(batch["image_embeddings"])

        # Calculate the loss
        mse_loss = mse_loss_fn(predictions, batch["text_embeddings"])
        cos_loss = 1 - cos_loss_fn(predictions, batch["text_embeddings"]).mean()

        loss = mse_loss + cos_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        if iteration % gradient_accumulation == 0:
            optimizer.step()

        # Log with iteration
        wandb.log(
            {
                "train_loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "cos_loss": cos_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "batch": iteration,
            }
        )

        if iteration % 10 == 0:
            # Validate
            with torch.no_grad():
                val_batch = next(val_dataset)
                val_batch["image_embeddings"] = val_batch["image_embeddings"].to("cuda")
                val_batch["text_embeddings"] = val_batch["text_embeddings"].to("cuda")
                val_predictions = model(val_batch["image_embeddings"])
                val_mse_loss = mse_loss_fn(
                    val_predictions, val_batch["text_embeddings"]
                )
                val_cos_loss = (
                    1
                    - cos_loss_fn(val_predictions, val_batch["text_embeddings"]).mean()
                )
                val_loss = val_mse_loss + val_cos_loss
                wandb.log(
                    {
                        "val_loss": val_loss.item(),
                        "val_mse_loss": val_mse_loss.item(),
                        "val_cos_loss": val_cos_loss.item(),
                        "batch": iteration,
                    }
                )

        scheduler.step(loss)

        if iteration % 250 == 0:
            # Save the model
            torch.save(model.state_dict(), f"translator_model_{iteration}.pt")

            # Upload the model
            wandb.save(f"translator_model_{iteration}.pt", policy="now")

            for example in ["cat", "dog", "math", "noir", "dragon"]:
                # Load the embeddings

                image_embeddings = torch.load(f"{example}_embedding.pt").to("cuda")

                # Get the predictions
                predictions = model(image_embeddings)

                # Save the predictions
                torch.save(predictions, f"{example}_{iteration}.pt")

                # Upload the predictions
                wandb.save(f"{example}_{iteration}.pt", policy="now")


def optimize_lr(
    input_path: str,
    batch_size: int = 256,
    iters: int = 7,  # 6000 * 3,
    num_layers: int = 2,
    hidden_size: int = 128,
    gradient_accumulation: int = 1,
    resume_from: None | str = None,
    starting_lr: float = 10**-10,
    max_lr: float = 10,
):
    # Load the loss function
    mse_loss_fn = torch.nn.MSELoss()

    # Load the other loss function
    cos_loss_fn = torch.nn.CosineSimilarity(dim=2)

    wandb.init(project="mixer-translator-optimization")

    current_lr = starting_lr

    while current_lr < max_lr:
        random.seed(0)

        # Load the dataset
        dataset = TranslatorDataset.from_path(input_path, batch_size)
        train_dataset, val_dataset = dataset.split(0.9)
        train_dataset, val_dataset = iter(train_dataset), iter(val_dataset)

        if current_lr != starting_lr:
            del model

            del optimizer

        # Load the model
        model = MixerTranslatorModel((257, 1024), (77, 768), hidden_size, num_layers)

        if resume_from is not None:
            model.load_state_dict(torch.load(resume_from))

        model = model.to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

        # Train

        for iteration in tqdm(range(iters + 1)):
            batch = next(train_dataset)

            # Move to cuda
            batch["image_embeddings"] = batch["image_embeddings"].to("cuda")
            batch["text_embeddings"] = batch["text_embeddings"].to("cuda")

            # Get the predictions
            predictions = model(batch["image_embeddings"])

            # Calculate the loss
            mse_loss = mse_loss_fn(predictions, batch["text_embeddings"])
            cos_loss = 1 - cos_loss_fn(predictions, batch["text_embeddings"]).mean()

            loss = mse_loss + cos_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            if iteration % gradient_accumulation == 0:
                optimizer.step()

        # Validate

        with torch.no_grad():
            val_batch = next(val_dataset)
            val_batch["image_embeddings"] = val_batch["image_embeddings"].to("cuda")
            val_batch["text_embeddings"] = val_batch["text_embeddings"].to("cuda")
            val_predictions = model(val_batch["image_embeddings"])
            val_mse_loss = mse_loss_fn(val_predictions, val_batch["text_embeddings"])
            val_cos_loss = (
                1 - cos_loss_fn(val_predictions, val_batch["text_embeddings"]).mean()
            )
            val_loss = val_mse_loss + val_cos_loss

        wandb.log(
            {
                "val_loss": val_loss.item(),
                "val_mse_loss": val_mse_loss.item(),
                "val_cos_loss": val_cos_loss.item(),
                "lr": current_lr,
            }
        )

        current_lr *= 2

        print(f"Current LR: {current_lr}")


if __name__ == "__main__":
    # train("scripts/translator/processed/")  # , resume_from="base_outliers.pt")

    # optimize_lr("scripts/translator/processed/", resume_from="translator_model_500.pt")

    train("scripts/translator/processed/")
