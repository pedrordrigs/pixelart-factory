"""
Smart Pixel Art Cleaner
Port of spritefusion-pixel-snapper (Rust) to Python.
Implements grid detection and dominant-color resampling.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class SmartConfig:
    k_colors: int = 16
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8


class SmartCleaner:
    def __init__(self, config: SmartConfig = SmartConfig()):
        self.config = config

    def process(self, image: Image.Image) -> Image.Image:
        """Main processing pipeline."""
        # 1. Quantize colors first (like in Rust implementation)
        # Convert to RGBA
        img = image.convert("RGBA")
        width, height = img.size
        
        # Quantize
        quantized = self.quantize_image(img)
        
        # 2. Compute profiles (gradients)
        profile_x, profile_y = self.compute_profiles(quantized)
        
        # 3. Estimate step sizes
        step_x_opt = self.estimate_step_size(profile_x)
        step_y_opt = self.estimate_step_size(profile_y)
        
        # 4. Resolve step sizes
        step_x, step_y = self.resolve_step_sizes(step_x_opt, step_y_opt, width, height)
        
        # 5. Walk profiles to find cuts
        raw_col_cuts = self.walk(profile_x, step_x, width)
        raw_row_cuts = self.walk(profile_y, step_y, height)
        
        # 6. Stabilize cuts (Simplified version of Rust's stabilize_both_axes)
        col_cuts = self.stabilize_cuts(profile_x, raw_col_cuts, width, raw_row_cuts, height)
        row_cuts = self.stabilize_cuts(profile_y, raw_row_cuts, height, raw_col_cuts, width)
        
        # 7. Resample
        result = self.resample(quantized, col_cuts, row_cuts)
        
        return result

    def quantize_image(self, img: Image.Image) -> Image.Image:
        """Reduce colors using K-Means."""
        if self.config.k_colors == 0:
            return img

        # Convert to numpy array
        arr = np.array(img)
        h, w, d = arr.shape
        
        # Filter out transparent pixels for training
        pixels = arr.reshape(-1, 4)
        opaque_mask = pixels[:, 3] > 0
        opaque_pixels = pixels[opaque_mask][:, :3] # RGB only
        
        if len(opaque_pixels) == 0:
            return img
            
        unique_colors = len(np.unique(opaque_pixels, axis=0))
        k = min(self.config.k_colors, unique_colors)
        
        if k < 2:
            return img

        print(f"Smart Quantizing to {k} colors...")
        
        # Use MiniBatchKMeans for speed if image is large, else KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=self.config.max_kmeans_iterations)
        kmeans.fit(opaque_pixels)
        centroids = kmeans.cluster_centers_.astype(np.uint8)
        
        # Map all pixels to nearest centroid
        # We need to process transparent pixels too (keep them transparent)
        
        # Create a new image array
        new_arr = np.zeros_like(arr)
        
        # Process opaque pixels
        labels = kmeans.predict(opaque_pixels)
        quantized_rgb = centroids[labels]
        
        # Reconstruct (this is slow in pure python, but numpy helps)
        # We need to put pixels back in place. 
        # Easier way: Predict all, then mask alpha.
        
        all_rgb = pixels[:, :3]
        all_labels = kmeans.predict(all_rgb)
        all_quantized = centroids[all_labels]
        
        final_pixels = np.zeros_like(pixels)
        final_pixels[:, :3] = all_quantized
        final_pixels[:, 3] = pixels[:, 3] # Keep original alpha
        
        # Mask transparent ones strictly? 
        # Rust impl keeps alpha as is but changes RGB. 
        # But for output we usually want full transparency if alpha is 0
        
        final_pixels = final_pixels.reshape(h, w, 4)
        return Image.fromarray(final_pixels.astype(np.uint8))

    def compute_profiles(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient profiles along X and Y axes."""
        # Convert to grayscale for gradient calc
        gray = np.array(img.convert("L")).astype(float)
        h, w = gray.shape
        
        col_proj = np.zeros(w)
        row_proj = np.zeros(h)
        
        # Compute gradients (simple difference)
        # Rust uses kernel [-1, 0, 1] effectively (right - left)
        
        # Gradient X
        # diff along axis 1 (columns)
        grad_x = np.abs(gray[:, 2:] - gray[:, :-2]) # skip borders
        # Sum down the rows
        col_proj[1:-1] = np.sum(grad_x, axis=0)
        
        # Gradient Y
        # diff along axis 0 (rows)
        grad_y = np.abs(gray[2:, :] - gray[:-2, :])
        row_proj[1:-1] = np.sum(grad_y, axis=1)
        
        return col_proj, row_proj

    def estimate_step_size(self, profile: np.ndarray) -> Optional[float]:
        """Estimate grid step size from profile peaks."""
        if len(profile) == 0:
            return None
            
        max_val = np.max(profile)
        if max_val == 0:
            return None
            
        threshold = max_val * self.config.peak_threshold_multiplier
        
        # Find local peaks
        peaks = []
        for i in range(1, len(profile) - 1):
            if (profile[i] > threshold and 
                profile[i] > profile[i-1] and 
                profile[i] > profile[i+1]):
                peaks.append(i)
                
        if len(peaks) < 2:
            return None
            
        # Filter peaks by distance
        clean_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - clean_peaks[-1] > (self.config.peak_distance_filter - 1):
                clean_peaks.append(p)
                
        if len(clean_peaks) < 2:
            return None
            
        # Compute diffs and median
        diffs = np.diff(clean_peaks)
        return float(np.median(diffs))

    def resolve_step_sizes(
        self, 
        sx: Optional[float], 
        sy: Optional[float], 
        w: int, 
        h: int
    ) -> Tuple[float, float]:
        """Decide final step sizes."""
        if sx is not None and sy is not None:
            ratio = sx / sy if sx > sy else sy / sx
            if ratio > self.config.max_step_ratio:
                smaller = min(sx, sy)
                return smaller, smaller
            else:
                avg = (sx + sy) / 2.0
                return avg, avg
        elif sx is not None:
            return sx, sx
        elif sy is not None:
            return sy, sy
        else:
            # Fallback
            fallback = max(1.0, min(w, h) / self.config.fallback_target_segments)
            return fallback, fallback

    def walk(self, profile: np.ndarray, step_size: float, limit: int) -> List[int]:
        """Walker algorithm to find cut positions."""
        cuts = [0]
        current_pos = 0.0
        
        search_window = max(
            step_size * self.config.walker_search_window_ratio,
            self.config.walker_min_search_window
        )
        mean_val = np.mean(profile) if len(profile) > 0 else 0
        
        while current_pos < limit:
            target = current_pos + step_size
            if target >= limit:
                cuts.append(limit)
                break
                
            start_search = int(max(target - search_window, current_pos + 1))
            end_search = int(min(target + search_window, limit))
            
            if end_search <= start_search:
                current_pos = target
                continue
                
            # Find max in window
            window = profile[start_search:end_search]
            if len(window) == 0:
                best_idx = int(target)
                best_val = 0
            else:
                rel_idx = np.argmax(window)
                best_idx = start_search + rel_idx
                best_val = window[rel_idx]
            
            if best_val > mean_val * self.config.walker_strength_threshold:
                cuts.append(best_idx)
                current_pos = float(best_idx)
            else:
                cuts.append(int(target))
                current_pos = target
                
        return sorted(list(set(cuts))) # Dedup and sort

    def stabilize_cuts(
        self, 
        profile: np.ndarray, 
        cuts: List[int], 
        limit: int, 
        sibling_cuts: List[int], 
        sibling_limit: int
    ) -> List[int]:
        """Refine cuts to be more uniform if needed."""
        # Simplified port: just ensure we have enough cuts and they aren't wildly wrong
        # If we have too few cuts, force uniform grid based on sibling or fallback
        
        # Sanitize
        cuts = sorted(list(set([max(0, min(x, limit)) for x in cuts])))
        if 0 not in cuts: cuts.insert(0, 0)
        if limit not in cuts: cuts.append(limit)
        
        min_required = self.config.min_cuts_per_axis
        if len(cuts) >= min_required:
            return cuts
            
        # Fallback to uniform
        cells = max(1, len(sibling_cuts) - 1)
        if sibling_limit > 0 and cells > 0:
            target_step = sibling_limit / cells
        else:
            target_step = limit / self.config.fallback_target_segments
            
        return self.snap_uniform_cuts(profile, limit, target_step)

    def snap_uniform_cuts(self, profile: np.ndarray, limit: int, target_step: float) -> List[int]:
        """Create uniform-ish cuts snapping to local peaks."""
        if target_step <= 0: return [0, limit]
        
        desired_cells = max(1, int(round(limit / target_step)))
        cell_width = limit / desired_cells
        
        search_window = max(
            cell_width * self.config.walker_search_window_ratio,
            self.config.walker_min_search_window
        )
        mean_val = np.mean(profile) if len(profile) > 0 else 0
        
        cuts = [0]
        for idx in range(1, desired_cells):
            target = cell_width * idx
            prev = cuts[-1]
            
            start = int(max(target - search_window, prev + 1))
            end = int(min(target + search_window, limit - 1))
            
            if end < start:
                best_idx = int(target)
            else:
                # Find best peak
                window = profile[start:end+1]
                if len(window) > 0:
                    best_idx = start + np.argmax(window)
                    best_val = window[np.argmax(window)]
                    
                    if best_val < mean_val * self.config.walker_strength_threshold:
                         best_idx = int(target)
                else:
                    best_idx = int(target)
            
            # Ensure strictly increasing
            if best_idx <= prev:
                best_idx = prev + 1
            if best_idx >= limit:
                break
                
            cuts.append(best_idx)
            
        cuts.append(limit)
        return sorted(list(set(cuts)))

    def resample(self, img: Image.Image, col_cuts: List[int], row_cuts: List[int]) -> Image.Image:
        """Resample image cells to single pixels using mode color."""
        cols = len(col_cuts) - 1
        rows = len(row_cuts) - 1
        
        if cols < 1 or rows < 1:
            return img
            
        arr = np.array(img)
        
        # New image size (logical pixels)
        # Note: The output of smart cleaner is often the "small" image (logical resolution)
        # But to be compatible with the pipeline, we might want to return it upscaled?
        # The Rust tool outputs the small logical image (e.g., 64x64).
        # But my pipeline expects `target_grid` size output usually.
        # I will generate the small one, and let the caller upscale if needed, 
        # OR I can return the small one and let the UI handle it.
        # To maintain API compatibility with `pixel_art_cleaner`, I should probably upscale it back 
        # if the user requested a specific size. 
        # BUT, `pixel_art_cleaner` takes `target_grid`. 
        # SmartCleaner figures out the grid itself. 
        # Let's return the logical image (small) and let the UI upscale it if desired.
        
        # Actually, let's upscale it back to the original size or nearest large size using Nearest Neighbor
        # to match the "Cleaned" look.
        
        out_arr = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        for r in range(rows):
            y_start = row_cuts[r]
            y_end = row_cuts[r+1]
            if y_end <= y_start: continue
            
            for c in range(cols):
                x_start = col_cuts[c]
                x_end = col_cuts[c+1]
                if x_end <= x_start: continue
                
                # Extract cell
                cell = arr[y_start:y_end, x_start:x_end]
                
                # Reshape to list of pixels
                pixels = cell.reshape(-1, 4)
                
                # Count frequencies
                # Numpy unique is a bit slow for this loop, but it's pure python port.
                # Optimization: just take the center pixel if too slow? 
                # No, mode is better.
                
                # Use a simple heuristic: random sampling or center if large, else full mode
                if len(pixels) > 100:
                    # Sample center 50%
                    center_y = (y_end - y_start) // 2
                    center_x = (x_end - x_start) // 2
                    # Just take center for speed in Python?
                    # The Rust one does full histogram.
                    # Let's try full unique, it shouldn't be too slow for typical pixel art sizes (256x256 -> 64x64)
                    pass
                
                # Fast mode finding
                # lexsort to sort rows, then find unique
                # or turn into void view
                
                # Simplest robust way:
                vals, counts = np.unique(pixels, axis=0, return_counts=True)
                mode_idx = np.argmax(counts)
                mode_pixel = vals[mode_idx]
                
                out_arr[r, c] = mode_pixel
                
        return Image.fromarray(out_arr)


