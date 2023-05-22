import torch
from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline, AutoencoderKL
import wandb
from loguru import logger


def sample_baseline(
    image: Image.Image, seed: int | torch.Tensor = 42, num_img: int = 4, pipeline=None
) -> list[Image.Image]:
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

    Returns
    -------
    list[PIL.Image]
        List of images.
    """

    if pipeline is None:
        pipeline = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
        )
        pipeline = pipeline.to("cuda")  # type: ignore
        pipeline = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5")
        created_pipeline = True
    else:
        created_pipeline = False

    if isinstance(seed, int):
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None
    else:
        generator = None
        latents = seed.to(torch.float16).to("cuda")

    results = pipeline(
        image=image,  # type: ignore
        num_images_per_prompt=num_img,
        generator=generator,
        latents=latents,  # type: ignore
        width=512,
        height=512,
    ).images  # type: ignore

    if created_pipeline:
        del pipeline

    return results


if __name__ == "__main__":
    images_path = "assets/images/"
    image_names = ["cat.png", "dog.jpg", "dragon.png", "math.jpg", "noir.png"]

    images = [images_path + name for name in image_names]

    seed = 42
    num_img = 4

    wandb.init(
        entity="img2img-eafit",
        project="img2img_reprojector_train",
        name="baseline",
        config={
            "seed": seed,
            "embeddings": images,
            "num_img": num_img,
        },
    )

    for path in images:
        logger.info(f"Sampling {path} images")
        with torch.no_grad():
            image = Image.open(path)
            results = sample_baseline(image, seed, num_img)
        for i, img in enumerate(results):
            wandb.log({path + "_" + str(i): wandb.Image(img)})
        logger.info(f"Done sampling {path} images")

    wandb.finish()
