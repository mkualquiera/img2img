"""Main runnable module for training the reprojection models.
"""

import argparse
import os

import torch
from loguru import logger
from tqdm import tqdm

import wandb
from img2img.config import load_model
from img2img.data.loading import ReprojectionDataset


def train(
    run_name: str,
    data_path: str,
    model_config_path: str,
    seed: int = 42,
    data_split: float = 0.8,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 32,
    training_steps: int = 100000,
    load_checkpoint: str | None = None,
    validation_frequency: int = 100,
    checkpoint_frequency: int = 1000,
    wandb_project: str = "img2img_reprojector_train",
    benchmarks_path: str | None = "assets/benchmarks",
    img_extension: str = ".jpg",
    upload_checkpoints_to_wandb: bool = False,
):
    # Ensure the runs directory exists.
    os.makedirs("runs", exist_ok=True)

    # Ensure that the directory for this run does not exist.
    run_path = os.path.join("runs", run_name)
    if os.path.exists(run_path):
        raise ValueError(f"Run directory already exists: {run_path}")

    # Create the directory for this run.
    logger.debug(f"Creating run directory: {run_path}")
    os.makedirs(run_path)

    # Load the model config.
    loaded_model_objects, loaded_config = load_model(model_config_path)

    model = loaded_model_objects.model.to("cuda")
    optimizer = loaded_model_objects.optimizer
    scheduler = loaded_model_objects.scheduler
    loss_fn = loaded_model_objects.loss_fn

    if load_checkpoint is not None:
        logger.info(f"Loading checkpoint: {load_checkpoint}")
        model.load_state_dict(torch.load(load_checkpoint))

    # Initialize the dataset.
    dataset = ReprojectionDataset.from_path(
        data_path, batch_size, img_extension=img_extension
    )
    training_dataset, validation_dataset = dataset.split(data_split)
    training_dataset_iterator = iter(training_dataset)
    validation_dataset_iterator = iter(validation_dataset)

    # Create the wandb run.
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "load_checkpoint": load_checkpoint,
            "loaded_config": loaded_config,
            "data_path": data_path,
            "data_split": data_split,
            "batch_size": batch_size,
            "training_steps": training_steps,
            "seed": seed,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "validation_frequency": validation_frequency,
            "checkpoint_frequency": checkpoint_frequency,
        },
    )

    wandb.define_metric("training_loss", step_metric="step")
    wandb.define_metric("validation_loss", step_metric="step")
    wandb.watch(model)

    # Train the model.
    for step in tqdm(range(training_steps)):
        # Get the next batch of data.
        training_embedded_images, training_embedded_texts = next(
            training_dataset_iterator
        )

        training_embedded_images = training_embedded_images.to("cuda")
        training_embedded_texts = training_embedded_texts.to("cuda")

        # Run the model.
        predicted_text_embeddings = model(training_embedded_images)

        # Calculate the loss.
        loss = loss_fn(predicted_text_embeddings, training_embedded_texts)

        # Backpropagate the loss.
        loss.backward()

        if step % gradient_accumulation_steps == 0:
            # Update the model.
            optimizer.step(None)
            scheduler.step()
            optimizer.zero_grad()

        # Log the loss.
        wandb.log(
            {
                "training_loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "step": step,
            },
            step=step,
        )

        # Validate the model.
        if step % validation_frequency == 0:
            with torch.no_grad():
                # Get the next batch of data.
                validation_embedded_images, validation_embedded_texts = next(
                    validation_dataset_iterator
                )

                validation_embedded_images = validation_embedded_images.to("cuda")
                validation_embedded_texts = validation_embedded_texts.to("cuda")

                # Run the model.
                predicted_text_embeddings = model(validation_embedded_images)

                # Calculate the loss.
                loss = loss_fn(predicted_text_embeddings, validation_embedded_texts)

                # Log the loss.
                wandb.log(
                    {
                        "validation_loss": loss.item(),
                        "step": step,
                    },
                    step=step,
                )

        # Save the model and run benchmarks.
        if step % checkpoint_frequency == 0:
            torch.save(model.state_dict(), os.path.join(run_path, f"model_{step}.pt"))

            if upload_checkpoints_to_wandb:
                wandb.save(os.path.join(run_path, f"model_{step}.pt"))

            if benchmarks_path is not None:
                # Get all .pt files in the benchmarks directory.
                benchmark_files = [
                    os.path.join(benchmarks_path, f)
                    for f in os.listdir(benchmarks_path)
                    if f.endswith(".pt")
                ]

                for benchmark_file in benchmark_files:
                    # Load the image embedding.
                    image_embedding = torch.load(benchmark_file).to("cuda")

                    logger.debug("Benchmarking on " + benchmark_file)

                    # Run the model.
                    predicted_text_embedding = model(image_embedding)

                    # Save the predicted text embedding.
                    basename = os.path.basename(benchmark_file)
                    result_path = os.path.join(
                        run_path, f"benchmark_{step}_{basename}.pt"
                    )

                    torch.save(predicted_text_embedding, result_path)

                    if upload_checkpoints_to_wandb:
                        wandb.save(result_path)

    # Save the final model.
    torch.save(model.state_dict(), os.path.join(run_path, "model.pt"))

    if upload_checkpoints_to_wandb:
        wandb.save(os.path.join(run_path, "model.pt"))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help=(
            "The name of the run. This will be used as the name of"
            " the directory in which the model is saved."
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the data directory.",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="The path to the model config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for reproducibility.",
    )
    parser.add_argument(
        "--data_split",
        type=float,
        default=0.8,
        help="The fraction of data to use for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use during training.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=100000,
        help="The number of training steps to perform.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="The path to a checkpoint to load.",
    )
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=100,
        help="The frequency with which to validate the model.",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=1000,
        help="The frequency with which to save checkpoints.",
    )
    parser.add_argument(
        "--benchmarks_path",
        type=str,
        default="assets/benchmarks",
        help="The path to the benchmarks directory.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="img2img_reprojector_train",
        help="The wandb project to use.",
    )
    parser.add_argument(
        "--img_extension",
        type=str,
        default=".jpg",
        help="The extension of the image files.",
    )
    parser.add_argument(
        "--upload_checkpoints_to_wandb",
        type=bool,
        default=False,
        help="Whether to upload checkpoints to wandb.",
    )
    args = parser.parse_args()

    train(**vars(args))
