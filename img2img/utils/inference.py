import os
import torch
from diffusers import StableDiffusionPipeline
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


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PILImage.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def sample_images(
    embeddings: torch.Tensor,
    seed: int | torch.Tensor = 42,
    num_img: int = 4,
    pipeline: StableDiffusionPipeline | None = None,
) -> list[list[Image]]:
    """Samples images using the embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        Embeddings to sample images from. (batch_size, a, b)
    seed : int or torch.Tensor, optional
        Seed for reproducibility, by default 42, or a torch tensor with
        shape (batch_size, a, b) for a custom latent vector.
    num_img : int, optional
        The number of images to sample for each embedding, by default 4
    pipeline : StableDiffusionPipeline, optional
        The pipeline to use for sampling, by default None

    Returns
    -------
    list[PIL.Image]
        List of images.
    """

    if pipeline is None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )  # type: ignore
        pipeline = pipeline.to("cuda")  # type: ignore
        created_pipeline = True
    else:
        created_pipeline = False

    if isinstance(seed, int):
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None
    else:
        generator = None
        latents = seed.to(torch.float16).to("cuda")

    results = [
        pipeline(
            prompt_embeds=emb.unsqueeze(0),  # type: ignore
            num_images_per_prompt=num_img,
            generator=generator,
            guidance_scale=16.5,
            latents=latents,  # type: ignore
        ).images  # type: ignore
        for emb in embeddings
    ]

    if created_pipeline:
        del pipeline

    return results  # type: ignore


def do_inference(
    data_path: str,
    model_config_path: str,
    checkpoint_path: str,
    seed: int = 42,
    generated_per_input: int = 4,
    batch_size: int = 4,
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

    logger.info(f"Found {len(all_files)} images.")

    all_files = sorted(all_files)
    all_files = all_files[offset : offset + total_inputs]

    # Load image embedder
    logger.info("Loading image embedder...")
    image_embedder = load_image_embedder()
    image_processor = image_embedder.get_processor()

    # Load model
    logger.info("Creating model...")
    model_objects, loaded_config = load_model(model_config_path)
    model = model_objects.model.to("cuda")

    # Load state dict
    logger.info("Loading checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path))

    # Create pipeline
    logger.info("Creating diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )  # type: ignore
    pipeline.safety_checker = None  # type: ignore
    pipeline = pipeline.to("cuda")  # type: ignore

    batches = [
        all_files[i : i + batch_size] for i in range(0, len(all_files), batch_size)
    ]

    # Create numpy seeded generator
    generator = np.random.default_rng(seed)
    logger.info("Starting inference...")

    for i, batch in tqdm(enumerate(batches)):
        loaded_images = load_images_from_paths(batch)
        processed_images = process_images(image_processor, loaded_images)
        image_embeddings = embed_images(image_embedder, processed_images)
        text_embeddings = model(image_embeddings)

        print(text_embeddings.shape)

        # Sample images.
        batch_seed = int(generator.integers(0, 2**32 - 1))
        sampled_images = sample_images(
            text_embeddings,
            seed=batch_seed,
            num_img=generated_per_input,
            pipeline=pipeline,
        )

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
