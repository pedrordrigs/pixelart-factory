"""
Smart Pixel Art Cleaner
Port of spritefusion-pixel-snapper (Rust) to Python.
Implements grid detection and dominant-color resampling.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class SmartConfig:
    k_colors: int = 16
    k_seed: int = 42
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8 # Lowered from 3.0 to catch more skew cases

class SmartCleaner:
    def __init__(self, config: SmartConfig = None):
        self.config = config if config else SmartConfig()

    def process(self, image: Image.Image) -> Image.Image:
        """Main processing pipeline matching Rust implementation."""
        # 1. Load and Validate
        img = image.convert("RGBA")
        width, height = img.size
        
        if width == 0 or height == 0:
             raise ValueError("Image dimensions cannot be zero")
        if width > 10000 or height > 10000:
             raise ValueError("Image dimensions too large")

        # 2. Quantize
        quantized_img = self.quantize_image(img)
        
        # 3. Compute Profiles
        profile_x, profile_y = self.compute_profiles(quantized_img)
        
        # 4. Estimate step sizes
        step_x_opt = self.estimate_step_size(profile_x)
        step_y_opt = self.estimate_step_size(profile_y)
        
        # 5. Resolve step sizes
        step_x, step_y = self.resolve_step_sizes(step_x_opt, step_y_opt, width, height)
        
        # 6. Walk profiles to find raw cuts
        raw_col_cuts = self.walk(profile_x, step_x, width)
        raw_row_cuts = self.walk(profile_y, step_y, height)
        
        # 7. Two-pass stabilization
        col_cuts, row_cuts = self.stabilize_both_axes(
            profile_x, profile_y,
            raw_col_cuts, raw_row_cuts,
            width, height
        )
        
        # 8. Resample
        output_img = self.resample(quantized_img, col_cuts, row_cuts)
        
        return output_img

    def quantize_image(self, img: Image.Image) -> Image.Image:
        if self.config.k_colors == 0:
            return img

        # Convert to numpy array
        arr = np.array(img)
        h, w, d = arr.shape
        
        # Reshape to pixels
        pixels = arr.reshape(-1, 4)
        
        # Filter opaque pixels (alpha > 0)
        # Rust: p[3] == 0 check
        opaque_mask = pixels[:, 3] > 0
        opaque_pixels = pixels[opaque_mask][:, :3] # RGB only
        
        n_pixels = len(opaque_pixels)
        if n_pixels == 0:
            return img
            
        k = min(self.config.k_colors, len(np.unique(opaque_pixels, axis=0)))
        if k < 1: # Should be at least 1 color if n_pixels > 0
            k = 1
            
        # Use MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.config.k_seed,
            n_init=3,
            max_iter=self.config.max_kmeans_iterations,
            batch_size=256
        )
        kmeans.fit(opaque_pixels)
        centroids = kmeans.cluster_centers_.astype(np.uint8)
        
        # Predict all opaque pixels
        # Create output array initialized with original
        new_pixels = pixels.copy()
        
        # Only update RGB of opaque pixels, keep Alpha
        labels = kmeans.predict(new_pixels[opaque_mask][:, :3])
        new_pixels[opaque_mask, :3] = centroids[labels]
        
        # Reshape back to image
        new_arr = new_pixels.reshape(h, w, 4)
        return Image.fromarray(new_arr)

    def compute_profiles(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        w, h = img.size
        if w < 3 or h < 3:
            raise ValueError("Image too small (minimum 3x3)")
            
        # Convert to grayscale weights: 0.299 R + 0.587 G + 0.114 B
        # Alpha is ignored in Rust impl (if p[3]==0 return 0.0 else gray)
        # We can simulate this
        
        arr = np.array(img).astype(float)
        # Calculate gray
        r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Apply alpha mask (0 if transparent)
        gray[a == 0] = 0.0
        
        col_proj = np.zeros(w)
        row_proj = np.zeros(h)
        
        # Kernel [-1, 0, 1] means abs(right - left)
        # For x in 1..w-1
        # left = gray(x-1, y), right = gray(x+1, y)
        
        # Vectorized Gradient X
        # gray is (h, w)
        # diff between col i+1 and i-1
        grad_x = np.abs(gray[:, 2:] - gray[:, :-2]) # (h, w-2)
        # Sum along height (axis 0)
        col_proj[1:-1] = np.sum(grad_x, axis=0)
        
        # Vectorized Gradient Y
        grad_y = np.abs(gray[2:, :] - gray[:-2, :]) # (h-2, w)
        # Sum along width (axis 1)
        row_proj[1:-1] = np.sum(grad_y, axis=1)
        
        return col_proj, row_proj

    def estimate_step_size(self, profile: np.ndarray) -> Optional[float]:
        if len(profile) == 0:
            return None
            
        max_val = np.max(profile)
        if max_val == 0:
            return None
            
        threshold = max_val * self.config.peak_threshold_multiplier
        
        # Find peaks
        # Rust: profile[i] > threshold && profile[i] > prev && profile[i] > next
        peaks = []
        for i in range(1, len(profile) - 1):
            if (profile[i] > threshold and 
                profile[i] > profile[i-1] and 
                profile[i] > profile[i+1]):
                peaks.append(i)
                
        if len(peaks) < 2:
            return None
            
        # Filter by distance
        clean_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - clean_peaks[-1] > (self.config.peak_distance_filter - 1):
                clean_peaks.append(p)
                
        if len(clean_peaks) < 2:
            return None
            
        # Compute diffs
        diffs = np.diff(clean_peaks)
        return float(np.median(diffs))

    def resolve_step_sizes(
        self, sx: Optional[float], sy: Optional[float], w: int, h: int
    ) -> Tuple[float, float]:
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
            fallback = max(1.0, min(w, h) / self.config.fallback_target_segments)
            return fallback, fallback

    def walk(self, profile: np.ndarray, step_size: float, limit: int) -> List[int]:
        if len(profile) == 0:
             raise ValueError("Cannot walk on empty profile")
             
        cuts = [0]
        current_pos = 0.0
        search_window = max(
            step_size * self.config.walker_search_window_ratio,
            self.config.walker_min_search_window
        )
        mean_val = np.mean(profile) if len(profile) > 0 else 0.0
        
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
            # indices in window are relative to start_search
            if len(window) == 0:
                best_idx = int(target)
                best_val = -1.0
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
                
        return sorted(list(set(cuts)))

    def stabilize_both_axes(
        self,
        profile_x: np.ndarray, profile_y: np.ndarray,
        raw_col_cuts: List[int], raw_row_cuts: List[int],
        width: int, height: int
    ) -> Tuple[List[int], List[int]]:
        
        col_cuts_pass1 = self.stabilize_cuts(
            profile_x, raw_col_cuts, width, raw_row_cuts, height
        )
        row_cuts_pass1 = self.stabilize_cuts(
            profile_y, raw_row_cuts, height, raw_col_cuts, width
        )
        
        col_cells = max(1, len(col_cuts_pass1) - 1)
        row_cells = max(1, len(row_cuts_pass1) - 1)
        
        col_step = width / col_cells
        row_step = height / row_cells
        
        step_ratio = col_step / row_step if col_step > row_step else row_step / col_step
        
        if step_ratio > self.config.max_step_ratio:
            target_step = min(col_step, row_step)
            
            if col_step > target_step * 1.2:
                final_col_cuts = self.snap_uniform_cuts(
                    profile_x, width, target_step
                )
            else:
                final_col_cuts = col_cuts_pass1
                
            if row_step > target_step * 1.2:
                final_row_cuts = self.snap_uniform_cuts(
                    profile_y, height, target_step
                )
            else:
                final_row_cuts = row_cuts_pass1
                
            return final_col_cuts, final_row_cuts
        else:
            return col_cuts_pass1, row_cuts_pass1

    def stabilize_cuts(
        self,
        profile: np.ndarray,
        cuts: List[int],
        limit: int,
        sibling_cuts: List[int],
        sibling_limit: int
    ) -> List[int]:
        if limit == 0: return [0]
        
        cuts = self.sanitize_cuts(cuts, limit)
        min_required = max(2, min(self.config.min_cuts_per_axis, limit + 1))
        
        axis_cells = len(cuts) - 1
        sibling_cells = max(0, len(sibling_cuts) - 1)
        
        sibling_has_grid = (sibling_limit > 0 and 
                            sibling_cells >= max(1, min_required - 1))
                            
        steps_skewed = False
        if sibling_has_grid and axis_cells > 0:
            axis_step = limit / axis_cells
            sibling_step = sibling_limit / sibling_cells
            step_ratio = axis_step / sibling_step
            if step_ratio > self.config.max_step_ratio or step_ratio < (1.0 / self.config.max_step_ratio):
                steps_skewed = True
                
        has_enough = len(cuts) >= min_required
        
        if has_enough and not steps_skewed:
            return cuts
            
        # Fallback target step
        if sibling_has_grid:
            target_step = sibling_limit / sibling_cells
        elif self.config.fallback_target_segments > 1:
            target_step = limit / self.config.fallback_target_segments
        elif axis_cells > 0:
            target_step = limit / axis_cells
        else:
            target_step = float(limit)
            
        if target_step <= 0: target_step = 1.0
        
        return self.snap_uniform_cuts(profile, limit, target_step)

    def sanitize_cuts(self, cuts: List[int], limit: int) -> List[int]:
        if limit == 0: return [0]
        
        # Filter and clamp
        new_cuts = []
        has_zero = False
        has_limit = False
        
        for val in cuts:
            if val == 0: has_zero = True
            if val >= limit:
                val = limit
                has_limit = True
            new_cuts.append(val)
            
        if not has_zero: new_cuts.append(0)
        if not has_limit: new_cuts.append(limit)
        
        return sorted(list(set(new_cuts)))

    def snap_uniform_cuts(self, profile: np.ndarray, limit: int, target_step: float) -> List[int]:
        if limit == 0: return [0]
        if limit == 1: return [0, 1]
        
        if target_step > 0:
            desired_cells = int(round(limit / target_step))
        else:
            desired_cells = 0
            
        min_required = max(2, min(self.config.min_cuts_per_axis, limit + 1))
        desired_cells = max(max(1, min_required - 1), desired_cells)
        desired_cells = min(desired_cells, limit)
        
        cell_width = limit / desired_cells
        search_window = max(
            cell_width * self.config.walker_search_window_ratio,
            self.config.walker_min_search_window
        )
        mean_val = np.mean(profile) if len(profile) > 0 else 0.0
        
        cuts = [0]
        for idx in range(1, desired_cells):
            target = cell_width * idx
            prev = cuts[-1]
            if prev + 1 >= limit: break
            
            start = int(max(target - search_window, prev + 1))
            end = int(min(target + search_window, limit - 1))
            
            if end < start:
                best_idx = int(target)
            else:
                window = profile[start:end+1]
                if len(window) > 0:
                    rel_idx = np.argmax(window)
                    best_idx = start + rel_idx
                    best_val = window[rel_idx]
                    
                    if best_val < mean_val * self.config.walker_strength_threshold:
                         # Fallback
                         fallback = int(round(target))
                         fallback = max(prev + 1, fallback)
                         fallback = min(limit - 1, fallback)
                         best_idx = fallback
                else:
                    best_idx = int(target)
                    
            if best_idx <= prev:
                best_idx = prev + 1
            if best_idx >= limit:
                break
                
            cuts.append(best_idx)
            
        cuts.append(limit)
        return self.sanitize_cuts(cuts, limit)

    def resample(self, img: Image.Image, col_cuts: List[int], row_cuts: List[int]) -> Image.Image:
        cols = len(col_cuts) - 1
        rows = len(row_cuts) - 1
        
        if cols < 1 or rows < 1:
            return img
            
        arr = np.array(img)
        # Output array (rows, cols, 4)
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
                pixels = cell.reshape(-1, 4)
                
                # Find mode color
                # Optimization: for very small cells (1x1), just take pixel
                if len(pixels) == 1:
                    out_arr[r, c] = pixels[0]
                    continue
                    
                # For larger cells, find most frequent
                # We can use np.unique, but it returns sorted unique elements
                vals, counts = np.unique(pixels, axis=0, return_counts=True)
                
                # We need to sort by count descending
                # Rust sorts by count desc, then pixel value asc (for stability)
                # We can just take argmax count
                if len(counts) > 0:
                    mode_idx = np.argmax(counts)
                    out_arr[r, c] = vals[mode_idx]
                    
        return Image.fromarray(out_arr)
