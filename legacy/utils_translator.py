# This has utils for the model that converts FrozenCLIPImageEmbedder latent
# representations to FrozenCLIPTextEmbedder latent representations.

import random
import time
from typing import cast
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenCLIPImageEmbedder
import torch
import numpy as np
from datasets.load import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import CLIPProcessor
from PIL import Image
import requests
import joblib
from img2dataset import download
import os
import tqdm
import wandb


def download_all():
    output_dir = os.path.abspath("scripts/translator/data")
    download(
        processes_count=8,
        url_list="scripts/translator/raw/lexica.json",
        output_folder=output_dir,
        output_format="files",
        input_format="json",
        url_col="url",
        caption_col="prompt",
        resize_mode="center_crop",
        image_size=224,
    )


def get_text(path: str):
    with open(path) as f:
        return f.read().strip()


def preprocess(
    input_path: str,
    output_path: str,
    batch_size: int = 128,
    image_extension: str = ".jpg",
):
    """Preprocess the dataset to get the image and text embeddings."""

    # Get all jpg files recursively using walk
    image_paths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(image_extension):
                # ensure the file does have an acompaining text file
                text_path = os.path.join(root, file.replace(image_extension, ".txt"))
                if os.path.exists(text_path):
                    image_paths.append(os.path.join(root, file))

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image_embedder = FrozenCLIPImageEmbedder()
    text_embedder = FrozenCLIPEmbedder()

    wandb.init(
        project="mixer-translator-preprocessing",
        config={
            "batch_size": batch_size,
            "image_extension": image_extension,
            "num_images": len(image_paths),
        },
    )

    for batch_start in tqdm.tqdm(range(0, len(image_paths), batch_size)):
        batch_end = min(batch_start + batch_size, len(image_paths))
        # print(f"Processing batch {batch_start}-{batch_end}")
        batch = image_paths[batch_start:batch_end]
        # Load the images
        # print("Loading images")

        def try_open_image(path):
            try:
                return Image.open(path)
            except:
                batch.remove(path)
                return None

        ts = time.time()

        images = [try_open_image(image) for image in batch]

        images = [image for image in images if image is not None]

        image_load_time = time.time() - ts

        # print("Done loading images")

        # Preprocess the images

        ts = time.time()

        pixel_values = torch.tensor(np.array(processor(images=images)["pixel_values"]))

        image_preprocess_time = time.time() - ts

        # Get the image embeddings

        ts = time.time()

        image_embeddings = image_embedder(pixel_values)

        image_embeddings_time = time.time() - ts

        # print("Image embeddings", image_embeddings.shape)

        ts = time.time()

        texts = [get_text(image.replace(image_extension, ".txt")) for image in batch]

        print(texts[-1])
        # abspath
        print("file://" + os.path.abspath(batch[-1]))

        text_preprocess_time = time.time() - ts

        # Get the text embeddings

        ts = time.time()

        text_embeddings = text_embedder(texts)

        text_embeddings_time = time.time() - ts

        # print("Text embeddings", text_embeddings.shape)

        # Save the embeddings

        ts = time.time()

        joblib.dump(
            {
                "image_embeddings": image_embeddings.to("cpu"),
                "text_embeddings": text_embeddings.to("cpu"),
            },
            output_path + f"/{batch_start}_{batch_end}.joblib",
        )

        save_time = time.time() - ts

        wandb.log(
            {
                "batch_size": len(batch),
                "image_load_time": image_load_time,
                "image_preprocess_time": image_preprocess_time,
                "image_embeddings_time": image_embeddings_time,
                "text_preprocess_time": text_preprocess_time,
                "text_embeddings_time": text_embeddings_time,
                "save_time": save_time,
                "percent_done": batch_end / len(image_paths),
            }
        )

        # print("Saved batch", batch_start, batch_end, time.time() - ts)


from train_translator import MixerTranslatorModel


def sample(image_path: str, output_path: str, output_embedding_path: str):
    image = Image.open(image_path)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    pixel_values = torch.tensor(np.array(processor(images=[image])["pixel_values"])).to(
        "cuda"
    )

    image_embedder = FrozenCLIPImageEmbedder()

    image_embeddings = image_embedder(pixel_values)

    # Create empty model
    model = MixerTranslatorModel((257, 1024), (77, 768), 128, 2).to("cuda")

    # Load the model
    model.load_state_dict(torch.load("translator_model_500.pt"))

    torch.save(image_embeddings, output_embedding_path)

    text_embeddings = model(image_embeddings).to("cpu")

    # Save the embeddings using torch.save
    torch.save(text_embeddings, output_path)


if __name__ == "__main__":
    # preprocess(
    #    "scripts/translator/data",
    #    "scripts/translator/processed",
    #    image_extension=".webp",
    # )

    sample("dragon.png", "dragon.pt", "dragon_embedding.pt")

    # download_all()
