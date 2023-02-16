"""Module for loading data."""

import random
from dataclasses import dataclass

import torch
from transformers import CLIPProcessor

from img2img.data.preprocessing import (
    embed_images,
    embed_text,
    get_file_list,
    load_image_embedder,
    load_images_from_paths,
    load_text_embedder,
    load_texts_from_paths,
    process_images,
    tokenize_strings,
)
from img2img.models.embedders import FrozenCLIPEmbedder, FrozenCLIPImageEmbedder


@dataclass
class ReprojectionDataset:
    """Dataset for reprojecting image embeddings to text embeddings. It has a
    list of file pairs and a batch size. It can be split into two datasets.
    """

    files: list[tuple[str, str]]
    batch_size: int
    text_embedder: FrozenCLIPEmbedder
    image_embedder: FrozenCLIPImageEmbedder
    image_processor: CLIPProcessor

    @staticmethod
    def from_path(
        path: str, batch_size: int, img_extension: str = ".jpg"
    ) -> "ReprojectionDataset":
        """Create a dataset from a path.

        Parameters
        ----------
        path : str
            The path to the dataset.
        batch_size : int
            The batch size.
        img_extension : str, optional
            The image extension, by default ".jpg"

        Returns
        -------
        ReprojectionDataset
            The dataset.
        """

        image_files = get_file_list(path, img_extension)
        # Get the text files
        text_files = [file.replace(img_extension, ".txt") for file in image_files]
        pairs = list(zip(image_files, text_files))

        # Shuffle the pairs
        random.shuffle(pairs)

        # Load the embedders
        text_embedder = load_text_embedder()
        image_embedder = load_image_embedder()

        return ReprojectionDataset(
            pairs,
            batch_size,
            text_embedder,
            image_embedder,
            image_embedder.get_processor(),
        )

    def split(
        self, percent: float
    ) -> tuple["ReprojectionDataset", "ReprojectionDataset"]:
        """Split the dataset into two datasets.

        Parameters
        ----------
        percent : float
            The percentage of the first dataset.

        Returns
        -------
        tuple[ReprojectionDataset, ReprojectionDataset]
            The two datasets.
        """

        split_index = int(len(self.files) * percent)
        return (
            ReprojectionDataset(
                self.files[:split_index],
                self.batch_size,
                self.text_embedder,
                self.image_embedder,
                self.image_processor,
            ),
            ReprojectionDataset(
                self.files[split_index:],
                self.batch_size,
                self.text_embedder,
                self.image_embedder,
                self.image_processor,
            ),
        )

    def __iter__(self):
        return TranslatorDatasetIterator(self)


class TranslatorDatasetIterator:
    """Iterator for the TranslatorDataset."""

    def __init__(self, dataset: ReprojectionDataset):
        self.dataset = dataset
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of data. (text, image)

        Returns
        -------
        dict[torch.Tensor, torch.Tensor]
            The batch of data.
        """

        if self.batch_index >= len(self.dataset.files):
            self.batch_index = 0

        batch_end = self.batch_index + self.dataset.batch_size

        batch_files = self.dataset.files[self.batch_index : batch_end]
        batch_images, batch_texts = zip(*batch_files)

        loaded_images, loaded_texts = (
            load_images_from_paths(list(batch_images)),
            load_texts_from_paths(list(batch_texts)),
        )

        processed_images, tokenized_texts = (
            process_images(self.dataset.image_processor, loaded_images),
            tokenize_strings(self.dataset.text_embedder, loaded_texts),
        )

        embedded_images, embedded_texts = (
            embed_images(self.dataset.image_embedder, processed_images),
            embed_text(self.dataset.text_embedder, tokenized_texts),
        )

        self.batch_index = batch_end

        return embedded_images, embedded_texts
