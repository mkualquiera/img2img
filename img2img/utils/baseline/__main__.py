import torch
from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline
import wandb
from loguru import logger


def sample_baseline(
    image: Image.Image, seed: int | torch.Tensor = 42, num_img: int = 4
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

    pipeline = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        torch_dtype=torch.float16,
        revision="2ddbd90b14bc5892c19925b15185e561bc8e5d0a",
    )
    pipeline = pipeline.to("cuda")  # type: ignore

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
