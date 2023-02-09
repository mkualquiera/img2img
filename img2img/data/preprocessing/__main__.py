"""Module for data preprocessing. Generally speaking, it involves taking
images and text and encoding them using the encoder models.
"""

import os
import shutil

import torch
import tqdm
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster, Lock
from loguru import logger

from img2img.models.embedders import FrozenCLIPEmbedder


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


def load_texts_from_paths(text_paths: list[str], disk_lock: Lock) -> list[str]:
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
    with gpu_lock:
        result = embedder.to("cuda")(tokens.to("cuda")).cpu()
    return result


def load_embedder() -> FrozenCLIPEmbedder:
    """Load the CLIP embedder.

    Returns
    -------
    FrozenCLIPEmbedder
        The CLIP embedder.
    """
    return FrozenCLIPEmbedder(device="cpu").cpu()


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

    with disk_lock:
        for embedding, new_path in zip(embeddings, new_paths):
            torch.save(embedding, new_path)

    return new_paths


def embed_texts_from_path(
    embedder: FrozenCLIPEmbedder,
    input_path: str,
    output_path: str,
    batch_size: int = 4,
    task_group_size: int = 16,
    force: bool = False,
) -> None:
    """Embed a list of tokenized strings.

    Parameters
    ----------
    embedder : FrozenCLIPEmbedder
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

    # bs=4, tgs=2, nworkers=4, tworker=2 -> 40:00
    # bs=4, tgs=8, nworkers=2, tworker=2 -> 18:00
    # bs=4, tgs=16, nworkers=2, tworker=2 -> 30:00
    # bs=4, tgs=8, nworkers=2, tworker=4 -> 19:00

    # Create a dask cluster
    with LocalCluster(processes=True, n_workers=2, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            # Show dashboard
            logger.info(f"Dashboard running at {cluster.dashboard_link}")  # type: ignore
            # Scatter the embedder to the cluster
            embedder_scattered = client.scatter(embedder)
            gpu_lock = Lock(name="gpu_lock")
            disk_lock = Lock(name="disk_lock")
            for task_group in tqdm.tqdm(task_groups, desc="Embedding text"):
                # Create a dask task for each batch
                tasks = []
                for batch in task_group:
                    texts = delayed(load_texts_from_paths)(batch, disk_lock)
                    tokens = delayed(tokenize_strings)(embedder_scattered, texts)
                    embeddings = delayed(embed_text)(
                        embedder_scattered, tokens, gpu_lock
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
    embed_texts_from_path(
        load_embedder(),
        "data_files/downloaded/laion6p0",
        "data_files/embedded/laion6p0",
    )
