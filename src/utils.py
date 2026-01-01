"""
Utility functions for PixelArt Factory.
"""

import os
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO


def load_image(path: str) -> Image.Image:
    """Load an image from disk."""
    return Image.open(path)


def save_image(image: Image.Image, path: str) -> None:
    """Save an image to disk, creating directories if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Image saved to: {path}")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert a base64 string to PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data))


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode."""
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode (no alpha)."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def get_output_path(input_path: str, suffix: str = "_cleaned") -> str:
    """Generate output path from input path with a suffix."""
    path = Path(input_path)
    return str(path.parent / f"{path.stem}{suffix}{path.suffix}")


def validate_grid_size(size: int) -> int:
    """Validate that grid size is a power of 2 and reasonable."""
    valid_sizes = [16, 32, 64, 128, 256, 512, 1024]
    if size not in valid_sizes:
        closest = min(valid_sizes, key=lambda x: abs(x - size))
        print(f"Warning: Grid size {size} adjusted to {closest}")
        return closest
    return size

