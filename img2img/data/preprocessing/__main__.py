"""Module for data preprocessing. Generally speaking, it involves taking
images and text and encoding them using the encoder models.
"""

import argparse
import os
import shutil

import numpy as np
import torch
import tqdm
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster, Lock
from loguru import logger
from PIL import Image
from transformers import CLIPProcessor

from img2img.models.embedders import FrozenCLIPEmbedder, FrozenCLIPImageEmbedder


def get_file_list(base_path: str, extension: str) -> list[str]:
    """Get a list of all files with a given extension in a directory.

    Parameters
    ----------
    base_path : str
        The base path to search.
    extension : str
        The extension to search for.

    Returns
    -------
    list[str]
        A list of all files with the given extension.
    """
    return [
        os.path.join(root, name)
        for root, _, files in os.walk(base_path)
        for name in files
        if name.endswith(extension)
    ]


def load_texts_from_paths(text_paths: list[str]) -> list[str]:
    """Load a list of texts from a list of paths.

    Parameters
    ----------
    text_paths : list[str]
        The paths to load from.

    Returns
    -------
    list[str]
        The loaded texts.
    """

    handles = [open(path, "r") for path in text_paths]
    texts = [handle.read() for handle in handles]
    for handle in handles:
        handle.close()

    return texts


def load_images_from_paths(image_paths: list[str]) -> list[Image.Image]:
    """Load a list of images from a list of paths.

    Parameters
    ----------
    image_paths : list[str]
        The paths to load from.

    Returns
    -------
    list[Image.Image]
        The loaded images.
    """

    images = [Image.open(path) for path in image_paths]
    return images


def tokenize_strings(embedder: FrozenCLIPEmbedder, strings: list[str]) -> torch.Tensor:
    """Tokenize a list of strings.

    Parameters
    ----------
    embedder : FrozenCLIPEmbedder
        The embedder to use.
    strings : list[str]
        The strings to tokenize.

    Returns
    -------
    list[str]
        The tokenized strings.
    """
    return embedder.tokenize(strings).cpu()


def process_images(processor: CLIPProcessor, images: list[Image.Image]) -> torch.Tensor:
    """Process a list of images.

    Parameters
    ----------
    processor : CLIPProcessor
        The processor to use.
    images : list[Image.Image]
        The images to process.

    Returns
    -------
    torch.Tensor
        The processed images.
    """
    return torch.tensor(np.array(processor(images=images)["pixel_values"]))


def embed_text(
    embedder: FrozenCLIPEmbedder, tokens: torch.Tensor, gpu_lock: Lock
) -> torch.Tensor:
    """Embed a list of tokenized strings.

    Parameters
    ----------
    embedder : FrozenCLIPEmbedder
        The embedder to use.
    tokens : torch.Tensor
        The tokenized strings.

    Returns
    -------
    torch.Tensor
        The embedded strings.
    """
    result = embedder.to("cuda")(tokens.to("cuda")).cpu()
    return result


def embed_images(
    embedder: FrozenCLIPImageEmbedder, images: torch.Tensor, gpu_lock: Lock
) -> torch.Tensor:
    """Embed a list of images.

    Parameters
    ----------
    embedder : FrozenCLIPImageEmbedder
        The embedder to use.
    images : torch.Tensor
        The images to embed.

    Returns
    -------
    torch.Tensor
        The embedded images.
    """
    with gpu_lock:
        result = embedder.to("cuda")(images.to("cuda")).cpu()
    return result


def load_text_embedder() -> FrozenCLIPEmbedder:
    """Load the CLIP text embedder.

    Returns
    -------
    FrozenCLIPEmbedder
        The CLIP embedder.
    """
    return FrozenCLIPEmbedder(device="cpu").cpu()


def load_image_embedder() -> FrozenCLIPImageEmbedder:
    """Load the CLIP image embedder.

    Returns
    -------
    FrozenCLIPImageEmbedder
        The CLIP embedder.
    """
    return FrozenCLIPImageEmbedder(device="cpu").cpu()


def save_embeddings(
    old_paths: list[str], embeddings: torch.Tensor, output_path: str, disk_lock: Lock
) -> list[str]:
    """Save a list of embeddings to a path.

    Parameters
    ----------
    old_paths : list[str]
        The paths to the original texts.
    embeddings : torch.Tensor
        The embeddings to save.
    output_path : str
        The path to save to.

    Returns
    -------
    list[str]
        The paths to the saved embeddings.
    """

    old_filenames = [os.path.basename(old_path) for old_path in old_paths]

    new_paths = [
        os.path.join(output_path, filename) + ".tensor" for filename in old_filenames
    ]

    for embedding, new_path in zip(embeddings, new_paths):
        torch.save(embedding, new_path)

    return new_paths


def embed_files_from_path(
    embedder: FrozenCLIPEmbedder | FrozenCLIPImageEmbedder,
    input_path: str,
    output_path: str,
    batch_size: int = 4,
    task_group_size: int = 16,
    n_workers: int = 2,
    n_thread_per_worker: int = 4,
    force: bool = False,
) -> None:
    """Embed a list of tokenized strings.

    Parameters
    ----------
    embedder : FrozenCLIPEmbedder | FrozenCLIPImageEmbedder
        The embedder to use.
    input_path : str
        The path to the input text.
    output_path : str
        The path to the output text.
    batch_size : int, optional
        The batch size to use, by default 512
    task_group_size : int, optional
        The number of batches to group together into a single dask compute call,
        by default 100
    force : bool, optional
        Whether to overwrite the output path if it already exists, by default False
    """

    is_image = isinstance(embedder, FrozenCLIPImageEmbedder)

    if not force:
        if os.path.exists(output_path):
            if input("Output path already exists. Overwrite? [y/N] ") == "y":
                shutil.rmtree(output_path)
            else:
                logger.error("Output path already exists. Exiting.")

    # Create the output path
    os.makedirs(output_path, exist_ok=True)

    filenames = get_file_list(input_path, ".txt")
    logger.info(f"Found {len(filenames)} files to embed.")

    # Split into batches. Last batch may be smaller than batch_size
    batches = [
        filenames[i : i + batch_size] for i in range(0, len(filenames), batch_size)
    ]
    logger.info(f"Split into {len(batches)} batches.")

    # Split into task groups. Last task group may be smaller than task_group_size
    task_groups = [
        batches[i : i + task_group_size]
        for i in range(0, len(batches), task_group_size)
    ]
    logger.info(f"Split into {len(task_groups)} task groups.")

    # Create a dask cluster
    with LocalCluster(
        processes=True, n_workers=n_workers, threads_per_worker=n_thread_per_worker
    ) as cluster:
        with Client(cluster) as client:
            # Show dashboard
            logger.info(f"Dashboard running at {cluster.dashboard_link}")  # type: ignore
            # Scatter the embedder to the cluster
            embedder_scattered = client.scatter(embedder)
            gpu_lock = Lock(name="gpu_lock")
            disk_lock = Lock(name="disk_lock")
            for task_group in tqdm.tqdm(task_groups, desc="Embedding files"):
                # Create a dask task for each batch
                tasks = []
                for batch in task_group:
                    loaded_files = delayed(
                        load_images_from_paths if is_image else load_texts_from_paths
                    )(batch)
                    processed = delayed(
                        process_images if is_image else tokenize_strings
                    )(embedder_scattered, loaded_files)
                    embeddings = delayed(embed_text)(
                        embedder_scattered, processed, gpu_lock
                    )
                    new_paths = delayed(save_embeddings)(
                        batch, embeddings, output_path, disk_lock
                    )
                    tasks.append(new_paths)

                # Compute the tasks
                new_paths = client.compute(tasks)
                # Wait for the tasks to finish
                client.gather(new_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        help="The path to the input files.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path to the output files.",
        required=True,
    )
    parser.add_argument(
        "--is_image",
        action="store_true",
        help="Whether the input is image files.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size to use.",
        default=4,
    )

    parser.add_argument(
        "--task_group_size",
        type=int,
        help="The number of batches to group together into a single dask compute call.",
        default=16,
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        help="The number of workers to use.",
        default=2,
    )

    parser.add_argument(
        "--n_thread_per_worker",
        type=int,
        help="The number of threads per worker to use.",
        default=4,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to overwrite the output path if it already exists.",
    )

    args = parser.parse_args()

    embed_files_from_path(
        load_image_embedder() if args.is_image else load_text_embedder(),
        args.input_path,
        args.output_path,
        args.batch_size,
        args.task_group_size,
        args.n_workers,
        args.n_thread_per_worker,
        args.force,
    )
