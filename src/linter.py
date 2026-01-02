"""
Pixel Linter - Quality Assurance for Pixel Art.
Checks for artifacts, palette integrity, and grid consistency.
"""

import numpy as np
from PIL import Image
from collections import Counter
from typing import Dict, Any, Tuple, List

from .utils import ensure_rgb

class PixelLinter:
    """
    Quality Assurance tool for pixel art.
    """
    
    def lint(self, image: Image.Image, expected_palette_size: int = 32) -> Dict[str, Any]:
        """
        Run all checks on an image.
        """
        results = {
            "palette_check": self.check_palette(image, expected_palette_size),
            "orphan_pixels": self.count_orphan_pixels(image),
            "grid_consistency": self.check_grid_consistency(image)
        }
        return results

    def check_palette(self, image: Image.Image, max_colors: int) -> Dict[str, Any]:
        """Verify palette size."""
        img = ensure_rgb(image)
        colors = img.getcolors(maxcolors=100000)
        num_colors = len(colors) if colors else 0
        
        return {
            "passed": num_colors <= max_colors,
            "count": num_colors,
            "max_allowed": max_colors
        }

    def count_orphan_pixels(self, image: Image.Image) -> int:
        """
        Count 'orphan' pixels (isolated pixels different from all 8 neighbors).
        These are usually noise.
        """
        img_arr = np.array(ensure_rgb(image))
        h, w, _ = img_arr.shape
        orphans = 0
        
        # Avoid boundary checks inside loop by ignoring 1-pixel border
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = img_arr[y, x]
                
                # Check 8 neighbors
                neighbors = [
                    img_arr[y-1, x-1], img_arr[y-1, x], img_arr[y-1, x+1],
                    img_arr[y,   x-1],                  img_arr[y,   x+1],
                    img_arr[y+1, x-1], img_arr[y+1, x], img_arr[y+1, x+1]
                ]
                
                is_orphan = True
                for n in neighbors:
                    if np.array_equal(center, n):
                        is_orphan = False
                        break
                
                if is_orphan:
                    orphans += 1
                    
        return orphans

    def remove_orphans(self, image: Image.Image) -> Image.Image:
        """
        Remove orphan pixels using a median filter or mode filter on 3x3 window.
        """
        # Simple implementation: if orphan, replace with mode of neighbors
        img_arr = np.array(ensure_rgb(image))
        h, w, _ = img_arr.shape
        out_arr = img_arr.copy()
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = img_arr[y, x]
                neighbors = [
                    img_arr[y-1, x-1], img_arr[y-1, x], img_arr[y-1, x+1],
                    img_arr[y,   x-1],                  img_arr[y,   x+1],
                    img_arr[y+1, x-1], img_arr[y+1, x], img_arr[y+1, x+1]
                ]
                
                # Check if orphan
                matches = 0
                for n in neighbors:
                    if np.array_equal(center, n):
                        matches += 1
                
                if matches == 0:
                    # Replace with most common neighbor
                    n_tuples = [tuple(n) for n in neighbors]
                    most_common = Counter(n_tuples).most_common(1)[0][0]
                    out_arr[y, x] = list(most_common)
                    
        return Image.fromarray(out_arr)

    def check_grid_consistency(self, image: Image.Image) -> Dict[str, Any]:
        """
        Check if the image respects a pixel grid.
        Calculates autocorrelation of gradients.
        """
        # Convert to grayscale
        gray = np.array(image.convert("L"), dtype=float)
        
        # Calculate gradients (simple difference)
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        
        # Sum gradients to get 1D signal
        signal_x = np.sum(grad_x, axis=0)
        signal_y = np.sum(grad_y, axis=1)
        
        def find_period(signal):
            # Autocorrelation
            n = len(signal)
            if n == 0: return 0
            
            # Remove mean
            signal = signal - np.mean(signal)
            
            # Correlate
            result = np.correlate(signal, signal, mode='full')
            result = result[n-1:] # Keep positive lags
            
            # Find peaks
            if len(result) < 2: return 0
            
            # Look for first strong peak
            search_range = result[2:n//2]
            if len(search_range) == 0: return 0
            
            lag = np.argmax(search_range) + 2
            return int(lag)

        period_x = find_period(signal_x)
        period_y = find_period(signal_y)
        
        return {
            "estimated_grid_x": period_x,
            "estimated_grid_y": period_y,
            "consistent": period_x > 1 and period_y > 1
        }

    def check_animation_jitter(self, frames: List[Image.Image], threshold: float = 0.1) -> Dict[str, Any]:
        """
        Check for excessive jitter between frames.
        Calculates pixel difference between consecutive frames.
        
        Args:
            frames: List of PIL images.
            threshold: Percentage of pixels that are allowed to change significantly.
            
        Returns:
            Dict with jitter statistics.
        """
        if len(frames) < 2:
            return {"jitter_score": 0.0, "status": "OK"}
            
        jitter_scores = []
        
        for i in range(len(frames) - 1):
            f1 = np.array(frames[i].convert("RGB"), dtype=int)
            f2 = np.array(frames[i+1].convert("RGB"), dtype=int)
            
            if f1.shape != f2.shape:
                return {"error": "Frame sizes mismatch"}
                
            # Absolute difference
            diff = np.abs(f1 - f2)
            # Sum channels
            diff_sum = np.sum(diff, axis=2)
            
            # Count pixels that changed significantly (> 20 intensity)
            changed_pixels = np.sum(diff_sum > 20)
            total_pixels = f1.shape[0] * f1.shape[1]
            
            score = changed_pixels / total_pixels
            jitter_scores.append(score)
            
        avg_jitter = np.mean(jitter_scores)
        
        return {
            "jitter_score": float(avg_jitter),
            "max_jitter": float(np.max(jitter_scores)),
            "status": "FAIL" if avg_jitter > threshold else "OK"
        }
