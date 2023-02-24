"""
This script intercepts requests to lexica.art and saves the images and text
to a file. Needs to be run with mitmproxy.
"""

import json
import os
import time

from mitmproxy import ctx, http


def response(flow: http.HTTPFlow) -> None:
    # The url will look like
    # https://lexica.art/api/trpc/prompts.infinitePrompts?batch=1&input={%220%22:{%22json%22:{%22text%22:%22%22,%22searchMode%22:%22images%22,%22source%22:%22search%22,%22cursor%22:0}}}

    if (
        "lexica.art" in flow.request.pretty_url
        and "infinitePrompts" in flow.request.pretty_url
    ):
        # Parse the json
        data = json.loads(flow.response.text)

        data = data[0]["result"]["data"]["json"]
        for i, prompt in enumerate(data["prompts"]):
            image = data["images"][i]
            id = image["id"]

            prompt_text = prompt["prompt"]

            with open(f"scripts/translator/data/{id}.txt", "a") as f:
                f.write(prompt_text)

    if "image.lexica.art" in flow.request.pretty_url:
        # Get the id
        id = flow.request.pretty_url.split("/")[-1]

        # Get the image
        image = flow.response.content

        with open(f"scripts/translator/data/{id}.webp", "wb") as f:
            f.write(image)
