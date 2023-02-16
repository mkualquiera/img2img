"""Module for downloading data. For example, point it to a parquet file from
https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus
"""

import argparse
import os
import shutil

from img2dataset import download
from loguru import logger


def download_parquet(
    parquet_path: str,
    output_path: str,
    n_proc: int = 8,
    n_threads: int = 256,
    url_col: str = "URL",
    caption_col: str = "TEXT",
):
    """Archetypical function for downloading data listed in a parquet file.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file.
    output_path : str
        Path to output directory.
    n_proc : int, optional
        Number of processes to use, by default 8
    n_threads : int, optional
        Number of threads to use, by default 256
    """

    if os.path.exists(output_path):
        if input(f"Output path {output_path} exists. Delete? [y/n]").lower() == "y":
            shutil.rmtree(output_path)
        else:
            logger.error("Output path exists. Exiting.")
            return

    logger.info("Starting download...")
    download(
        processes_count=n_proc,
        thread_count=n_threads,
        url_list=parquet_path,
        url_col=url_col,
        caption_col=caption_col,
        distributor="multiprocessing",
        resize_mode="center_crop",
        input_format="parquet",
        output_format="files",
        enable_wandb=True,
        wandb_project="img2img_download",
        output_folder=output_path,
        image_size=224,  # Default for ViT
    )
    logger.info("Download complete!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Make sure to error and show usage if parquet_path is not provided or
    # output_path is not provided
    parser.add_argument(
        "--parquet_path",
        type=str,
        help="Path to parquet file.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output directory.",
        required=True,
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=8,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=256,
        help="Number of threads to use.",
    )
    parser.add_argument(
        "--url_col",
        type=str,
        default="URL",
        help="Name of column containing URLs.",
    )
    parser.add_argument(
        "--caption_col",
        type=str,
        default="TEXT",
        help="Name of column containing captions.",
    )

    args = parser.parse_args()

    download_parquet(
        args.parquet_path,
        args.output_path,
        args.n_proc,
        args.n_threads,
        args.url_col,
        args.caption_col,
    )
