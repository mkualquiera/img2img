from enum import Enum
import wandb
from img2img.utils.inference import sample_images
import torch

from diffusers import SchedulerMixin


class MixingStrategy(str, Enum):
    """Enum that defines the different mixing strategies.
    A mixing strategy defines how to mix one noise sample with another.
    """

    TOP_TO_BOTTOM = "top_to_bottom"
    INTERPOLATE = "interpolate"
    DITHER = "dither"


def generate_noise_sample(seed: int) -> torch.Tensor:
    """Generate a noise sample using the specified seed.
    The noise sample is a torch tensor with shape (1, 256, 256).
    """

    generator = torch.Generator().manual_seed(seed)

    noise = torch.randn(1, 4, 64, 64, generator=generator)

    # Return the noise sample.
    return noise.to("cuda")


def mix_top_to_bottom(
    noise_a: torch.Tensor, noise_b: torch.Tensor, percent: float
) -> torch.Tensor:
    """Mix two noise samples together using the top to bottom mixing strategy.
    The mixing strategy is defined by the percent parameter. If percent is 0.5,
    then the top half of noise_a is used, and the bottom half of noise_b is used.
    The resulting noise sample is returned.
    """

    # Determine the number of rows to use from noise_a and noise_b.
    num_rows_a = int(noise_a.shape[2] * percent)
    num_rows_b = noise_a.shape[2] - num_rows_a

    # Create the noise sample.
    noise = torch.zeros_like(noise_a)
    noise[:, :, :num_rows_a] = noise_a[:, :, :num_rows_a]
    noise[:, :, num_rows_a:] = noise_b[:, :, num_rows_a:]

    # Return the noise sample.
    return noise


def mix_interpolate(
    noise_a: torch.Tensor, noise_b: torch.Tensor, percent: float
) -> torch.Tensor:
    """Mix two noise samples together using the interpolate mixing strategy.
    The mixing strategy is defined by the percent parameter. If percent is 0.5,
    then the top half of noise_a is used, and the bottom half of noise_b is used.
    The resulting noise sample is returned.
    """

    # Create the noise sample.
    noise = noise_a * percent + noise_b * (1 - percent)

    # Return the noise sample.
    return noise


def mix_dither(
    noise_a: torch.Tensor, noise_b: torch.Tensor, percent: float
) -> torch.Tensor:
    """Mix two noise samples together using a simple dithering strategy.
    The mixing strategy is defined by the percent parameter. For example, if
    percent is 0.1 then a pixel from a will be used 1/10 of the time, and a pixel
    from b will be used 9/10 of the time.
    """

    def hash2d(x, y):
        return (x * 73856093) ^ (y * 19349663) ^ 83492791

    def deterministic_random(x, y):
        hashval = hash2d(x, y)

        # Take the last 8 bits of the hash.
        hashval &= 0xFF

        # Convert to a float in the range [0, 1).
        return hashval / 256.0

    # Create the noise sample.
    noise = torch.zeros_like(noise_a)
    for i in range(noise_a.shape[2]):
        for j in range(noise_a.shape[3]):
            if deterministic_random(i, j) < percent:
                noise[:, :, i, j] = noise_a[:, :, i, j]
            else:
                noise[:, :, i, j] = noise_b[:, :, i, j]

    # Return the noise sample.
    return noise


def test_mixing(
    embed_path: str,
    seed_a: int,
    seed_b: int,
    mixing_strategy: MixingStrategy,
    percent: float = 0.5,
):
    """Generate a noise sample using seed_a, and then generate another noise sample
    using seed_b. Mix the two noise samples together using the mixing strategy
    specified by mixing_strategy and the percent specified by percent. The resulting
    noise sample is used to generate an image, which is saved to wandb.
    """

    # Generate the noise samples.
    noise_a = generate_noise_sample(seed_a)
    noise_b = generate_noise_sample(seed_b)

    # Mix the noise samples together.
    if mixing_strategy == MixingStrategy.TOP_TO_BOTTOM:
        noise = mix_top_to_bottom(noise_a, noise_b, percent)
    elif mixing_strategy == MixingStrategy.INTERPOLATE:
        noise = mix_interpolate(noise_a, noise_b, percent)
    elif mixing_strategy == MixingStrategy.DITHER:
        noise = mix_dither(noise_a, noise_b, percent)

    # Start wandb run.
    wandb.init(
        project="img2img_noise_info",
        config={
            "embed_path": embed_path,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "mixing_strategy": mixing_strategy.value,
            "percent": percent,
        },
    )

    # Load the embedding.
    embed = torch.load(embed_path)

    # Add batch dimension to the embedding.
    embed = embed.unsqueeze(0)

    # Generate the image.
    img = sample_images(embed, seed=noise, num_img=1)[0][0]

    # Log the image.
    wandb.log({"image": wandb.Image(img)})

    # End wandb run.
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
        help="Path to the embedding to use for generating the image.",
    )
    parser.add_argument(
        "--seed_a",
        type=int,
        required=True,
        help="Seed to use for generating the first noise sample.",
    )
    parser.add_argument(
        "--seed_b",
        type=int,
        required=True,
        help="Seed to use for generating the second noise sample.",
    )
    parser.add_argument(
        "--mixing_strategy",
        type=MixingStrategy,
        required=True,
        help="Mixing strategy to use for mixing the two noise samples.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=0.5,
        help="Percent to use for mixing the two noise samples.",
    )
    args = parser.parse_args()

    test_mixing(
        args.embed_path,
        args.seed_a,
        args.seed_b,
        args.mixing_strategy,
        args.percent,
    )
