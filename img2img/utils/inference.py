import torch
from diffusers import StableDiffusionPipeline
from PIL import Image as PILImage
from PIL.Image import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PILImage.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def sample_images(
    embeddings: torch.Tensor, seed: int | torch.Tensor = 42, num_img: int = 4
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

    Returns
    -------
    list[PIL.Image]
        List of images.
    """

    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")  # type: ignore

    if isinstance(seed, int):
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None
    else:
        generator = None
        latents = seed.to(torch.float16).to("cuda")

    results = [
        pipeline(
            prompt_embeds=emb,  # type: ignore
            num_images_per_prompt=num_img,
            generator=generator,
            guidance_scale=16.5,
            latents=latents,  # type: ignore
        ).images  # type: ignore
        for emb in embeddings
    ]

    del pipeline

    return results  # type: ignore
