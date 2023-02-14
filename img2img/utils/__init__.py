"""General utilities for img2img.
"""


from typing import Any


class NullScheduler:
    """A scheduler that does nothing."""

    def __init__(self, optimizer: Any) -> None:
        pass

    def step(self, *args, **kwargs):
        """Do nothing."""

    def state_dict(self):
        """Do nothing."""
        return {}

    def load_state_dict(self, *args, **kwargs):
        """Do nothing."""
