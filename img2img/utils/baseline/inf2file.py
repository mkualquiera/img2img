import os
import torch
from diffusers import (
    StableDiffusionImageVariationPipeline,
    AutoencoderKL,
    StableDiffusionPipeline,
)
from torchvision import transforms
from PIL import Image as PILImage
from PIL.Image import Image
import numpy as np
from img2img.data.preprocessing import (
    load_image_embedder,
    process_images,
    embed_images,
    load_images_from_paths,
)
from img2img.config import load_model
from loguru import logger
from tqdm import tqdm
from img2img.utils.baseline import sample_baseline


def do_inference(
    data_path: str,
    seed: int = 42,
    generated_per_input: int = 4,
    batch_size: int = 1,
    offset: int = 0,
    img_extension: str = ".jpg",
    total_inputs: int = 240,
    output_path: str = "inference",
):
    if total_inputs % batch_size != 0:
        raise ValueError(
            (
                f"Total inputs ({total_inputs}) must be a multiple of "
                f"batch size ({batch_size})."
            )
        )
    # Ensure output path exists.
    os.makedirs(output_path, exist_ok=True)

    # Recursively get all files with extension.
    all_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(img_extension):
                all_files.append(os.path.join(root, file))

    all_files = sorted(all_files)
    all_files = all_files[offset : offset + total_inputs]

    # Create pipeline
    logger.info("Creating diffusion pipeline...")
    pipeline = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    other_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )  # type: ignore
    pipeline.vae = other_pipeline.vae  # type: ignore
    del other_pipeline
    pipeline = pipeline.to("cuda")  # type: ignore
    pipeline.safety_checker = None  # type: ignore

    batches = [
        all_files[i : i + batch_size] for i in range(0, len(all_files), batch_size)
    ]

    tform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    # Create numpy seeded generator
    generator = np.random.default_rng(seed)
    logger.info("Starting inference...")

    for i, batch in tqdm(enumerate(batches)):
        loaded_images = load_images_from_paths(batch)

        inp = tform(loaded_images[0]).to("cuda").unsqueeze(0)

        # Sample images.
        batch_seed = int(generator.integers(0, 2**32 - 1))
        sampled_images = [
            sample_baseline(
                inp,
                seed=batch_seed,
                num_img=generated_per_input,
                pipeline=pipeline,
            )
        ]

        for input_file, images in zip(batch, sampled_images):
            input_name = os.path.basename(input_file)
            input_name = os.path.splitext(input_name)[0]
            for j, image in enumerate(images):
                output_name = f"{input_name}_{j}.jpg"
                output_file = os.path.join(output_path, output_name)
                image.save(output_file)

    logger.success("Done!")


if __name__ == "__main__":
    import fire

    fire.Fire(do_inference)
