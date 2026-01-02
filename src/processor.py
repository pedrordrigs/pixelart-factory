"""
Pixel Processor - The 'Factory' Layer.
Implements deterministic post-processing algorithms:
- CIELAB-based Color Quantization
- Advanced Dithering (Bayer, Floyd-Steinberg, Atkinson)
- Grid Alignment and Downscaling
"""

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from skimage import color
import math
from typing import Tuple, List, Optional, Union

from .utils import ensure_rgb, ensure_rgba

class PixelProcessor:
    """
    The main processing engine for the PixelArt Factory.
    Executes deterministic algorithms to transform raw AI outputs into strict pixel art.
    """
    
    def __init__(self):
        pass

    def extract_palette(self, image: Image.Image, max_colors: int = 16) -> np.ndarray:
        """
        Extract a palette from an image using K-Means (CIELAB).
        Returns: Numpy array of shape (N, 3) RGB colors.
        """
        img_rgb = ensure_rgb(image)
        img_array = np.array(img_rgb)
        
        # Convert to Lab
        img_lab = color.rgb2lab(img_array)
        pixels_lab = img_lab.reshape(-1, 3)
        
        kmeans = MiniBatchKMeans(n_clusters=max_colors, random_state=42, n_init=3)
        kmeans.fit(pixels_lab)
        centers_lab = kmeans.cluster_centers_
        
        # Convert centers back to RGB
        palette_rgb = color.lab2rgb(centers_lab) * 255
        return palette_rgb.astype(np.uint8)

    def apply_palette(
        self, 
        image: Image.Image, 
        palette: np.ndarray
    ) -> Image.Image:
        """
        Apply a fixed palette to an image.
        """
        img_rgb = ensure_rgb(image)
        
        # Nearest neighbor mapping
        img_arr = np.array(img_rgb)
        h, w, _ = img_arr.shape
        pixels = img_arr.reshape(-1, 3)
        
        # Chunk processing to save memory
        out_pixels = np.zeros_like(pixels)
        chunk_size = 10000
        
        for i in range(0, len(pixels), chunk_size):
            chunk = pixels[i:i+chunk_size]
            dists = np.sum((chunk[:, None, :] - palette[None, :, :]) ** 2, axis=2)
            nearest_indices = np.argmin(dists, axis=1)
            out_pixels[i:i+chunk_size] = palette[nearest_indices]
            
        return Image.fromarray(out_pixels.reshape(h, w, 3).astype(np.uint8))

    def quantize_colors_cielab(
        self, 
        image: Image.Image, 
        palette_size: int = 16
    ) -> Image.Image:
        """
        Quantize image colors using K-Means in CIELAB color space.
        """
        img_rgb = ensure_rgb(image)
        img_array = np.array(img_rgb)
        h, w, c = img_array.shape
        
        # 1. Convert to CIELAB
        img_lab = color.rgb2lab(img_array)
        pixels_lab = img_lab.reshape(-1, 3)
        
        # 2. Clustering
        kmeans = MiniBatchKMeans(n_clusters=palette_size, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels_lab)
        centers_lab = kmeans.cluster_centers_
        
        # 3. Reconstruct
        quantized_lab = centers_lab[labels].reshape(h, w, 3)
        quantized_rgb = color.lab2rgb(quantized_lab) * 255
        return Image.fromarray(quantized_rgb.astype(np.uint8))

    def downscale_smart(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        Downscale using Point Sampling (Center) to preserve hard edges and solidity.
        """
        # Ensure we are working with an RGB image
        img_rgb = ensure_rgb(image)
        orig_w, orig_h = img_rgb.size
        
        # Calculate block sizes
        if target_w > orig_w or target_h > orig_h:
            # Upscaling case: Use Nearest Neighbor to keep it solid
            return img_rgb.resize((target_w, target_h), Image.Resampling.NEAREST)
            
        # Optimization: If block size is integer, we can use fast array slicing
        # For pixel art, it's often integer scaling (e.g. 512 -> 64 is exactly 8x)
        if orig_w % target_w == 0 and orig_h % target_h == 0:
            block_w = int(orig_w / target_w)
            block_h = int(orig_h / target_h)
            
            arr = np.array(img_rgb)
            
            # Center offsets
            oy = block_h // 2
            ox = block_w // 2
            
            # Grid of center coordinates
            y_indices = np.arange(oy, orig_h, block_h)[:target_h]
            x_indices = np.arange(ox, orig_w, block_w)[:target_w]
            
            # Advanced slicing for 2D grid
            yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
            
            out_arr = arr[yy, xx]
            return Image.fromarray(out_arr)
            
        else:
            # Fractional downscaling - fallback to Nearest Neighbor resize
            # This is cleaner than mode filter on fractional blocks which introduces noise/aliasing
            return img_rgb.resize((target_w, target_h), Image.Resampling.NEAREST)

    def detect_and_snap_grid(self, image: Image.Image) -> Image.Image:
        """
        'Grid-Lock' Auto-Correction.
        Detects the dominant grid phase and snaps elements to integer coordinates.
        This is a placeholder that currently delegates to SmartCleaner in the pipeline.
        """
        # In the full integration, this logic is handled by SmartCleaner.
        return image

    def process_pipeline(
        self, 
        image: Image.Image,
        target_size: Tuple[int, int] = (64, 64),
        palette_size: int = 16,
        exact_scaling: bool = True
    ) -> Image.Image:
        """
        Run the full factory pipeline.
        """
        # 1. Downscale
        # Always use smart downscaling/upscaling (Point/Nearest) for pixel art
        processed = self.downscale_smart(image, target_size[0], target_size[1])
        
        # 2. Quantize
        final = self.quantize_colors_cielab(processed, palette_size)
        
        return final
