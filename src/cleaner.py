"""
Pixel Art Cleaner - Cleans AI-generated pixel art by snapping to a grid and quantizing colors.

Based on the Spritefusion Pixel Snapper concept and custom K-Means quantization.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple, Optional

from .utils import ensure_rgb, ensure_rgba


def pixel_art_cleaner(
    image: Image.Image,
    target_grid: int = 256,
    palette_size: int = 24,
    downscale_factor: int = 4,
    remove_background: bool = True,
    background_threshold: int = 240,
) -> Image.Image:
    """
    Clean AI-generated pixel art by snapping to a grid and quantizing colors.
    
    Args:
        image: Input PIL Image.
        target_grid: Final image size (e.g., 256x256).
        palette_size: Maximum number of colors in the palette.
        downscale_factor: Factor for logical pixel size.
                          target_grid / downscale_factor = logical resolution.
                          E.g., 256 / 4 = 64x64 logical pixels.
        remove_background: Whether to make near-white backgrounds transparent.
        background_threshold: RGB threshold for background detection (0-255).
    
    Returns:
        Cleaned pixel art image with RGBA mode.
    """
    # Step 1: Convert to RGB for K-Means processing
    img_rgb = ensure_rgb(image)
    
    # Step 2: Resize to target grid first (normalize input size)
    img_resized = img_rgb.resize((target_grid, target_grid), Image.Resampling.LANCZOS)
    
    # Step 3: Downscale to logical resolution (forces the grid)
    logical_res = target_grid // downscale_factor
    # Use BILINEAR for smoother downscale that averages colors
    img_small = img_resized.resize((logical_res, logical_res), Image.Resampling.BILINEAR)
    
    # Step 4: Color quantization via K-Means
    img_array = np.array(img_small)
    h, w, channels = img_array.shape
    image_2d = img_array.reshape(h * w, channels)
    
    # Clamp palette size to number of unique colors
    unique_colors = len(np.unique(image_2d, axis=0))
    actual_palette_size = min(palette_size, unique_colors)
    
    print(f"Quantizing to {actual_palette_size} colors (from {unique_colors} unique)...")
    
    kmeans = KMeans(
        n_clusters=actual_palette_size,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(image_2d)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    # Rebuild image with quantized colors
    quantized_2d = centers[labels]
    quantized_array = quantized_2d.reshape(h, w, channels)
    img_quantized = Image.fromarray(quantized_array)
    
    # Step 5: Upscale back to target resolution using NEAREST (pixel-perfect)
    final_image = img_quantized.resize(
        (target_grid, target_grid),
        Image.Resampling.NEAREST
    )
    
    # Step 6: Convert to RGBA and optionally remove background
    final_image = ensure_rgba(final_image)
    
    if remove_background:
        final_image = make_background_transparent(final_image, background_threshold)
    
    return final_image


def make_background_transparent(
    image: Image.Image,
    threshold: int = 240
) -> Image.Image:
    """
    Make near-white pixels transparent.
    
    Args:
        image: Input RGBA image.
        threshold: RGB values above this become transparent.
    
    Returns:
        Image with transparent background.
    """
    image = ensure_rgba(image)
    data = image.getdata()
    
    new_data = []
    for pixel in data:
        r, g, b, a = pixel
        if r > threshold and g > threshold and b > threshold:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(pixel)
    
    image.putdata(new_data)
    return image


def clean_from_file(
    input_path: str,
    output_path: str,
    target_grid: int = 256,
    palette_size: int = 24,
    downscale_factor: int = 4,
    remove_background: bool = True,
) -> None:
    """
    Clean a pixel art image from file and save the result.
    
    Args:
        input_path: Path to input image.
        output_path: Path to save cleaned image.
        target_grid: Final image size.
        palette_size: Number of colors.
        downscale_factor: Logical pixel size factor.
        remove_background: Whether to remove white background.
    """
    print(f"Loading image: {input_path}")
    image = Image.open(input_path)
    
    print(f"Processing with grid={target_grid}, palette={palette_size}, factor={downscale_factor}")
    cleaned = pixel_art_cleaner(
        image,
        target_grid=target_grid,
        palette_size=palette_size,
        downscale_factor=downscale_factor,
        remove_background=remove_background,
    )
    
    cleaned.save(output_path)
    print(f"Cleaned image saved to: {output_path}")


def get_logical_resolution(target_grid: int, downscale_factor: int) -> Tuple[int, int]:
    """Calculate the logical resolution based on grid and downscale factor."""
    logical = target_grid // downscale_factor
    return (logical, logical)


def estimate_optimal_palette(image: Image.Image, max_colors: int = 64) -> int:
    """
    Estimate optimal palette size based on image complexity.
    
    Args:
        image: Input image.
        max_colors: Maximum palette size to consider.
    
    Returns:
        Suggested palette size.
    """
    # Sample the image at lower resolution
    sample = image.resize((64, 64), Image.Resampling.NEAREST)
    sample_rgb = ensure_rgb(sample)
    
    # Count unique colors
    arr = np.array(sample_rgb)
    pixels = arr.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    
    # Suggest a palette size (usually 50-75% of unique colors, capped)
    suggested = min(int(unique_colors * 0.6), max_colors)
    return max(8, suggested)  # At least 8 colors


class PixelArtCleaner:
    """
    Class-based interface for pixel art cleaning with configurable defaults.
    """
    
    def __init__(
        self,
        target_grid: int = 256,
        palette_size: int = 24,
        downscale_factor: int = 4,
        remove_background: bool = True,
        background_threshold: int = 240,
    ):
        self.target_grid = target_grid
        self.palette_size = palette_size
        self.downscale_factor = downscale_factor
        self.remove_background = remove_background
        self.background_threshold = background_threshold
    
    def clean(self, image: Image.Image) -> Image.Image:
        """Clean a single image."""
        return pixel_art_cleaner(
            image,
            target_grid=self.target_grid,
            palette_size=self.palette_size,
            downscale_factor=self.downscale_factor,
            remove_background=self.remove_background,
            background_threshold=self.background_threshold,
        )
    
    def clean_file(self, input_path: str, output_path: str) -> None:
        """Clean an image file and save."""
        clean_from_file(
            input_path,
            output_path,
            target_grid=self.target_grid,
            palette_size=self.palette_size,
            downscale_factor=self.downscale_factor,
            remove_background=self.remove_background,
        )
    
    def get_logical_size(self) -> Tuple[int, int]:
        """Get the logical pixel resolution."""
        return get_logical_resolution(self.target_grid, self.downscale_factor)

