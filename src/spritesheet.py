"""
Spritesheet Generator - Create and process spritesheets.
"""

import os
from typing import List, Optional, Tuple, Union
from PIL import Image

from .generator import PixelArtGenerator
from .cleaner import pixel_art_cleaner


class SpritesheetGenerator:
    """
    Generate and manipulate spritesheets.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the spritesheet generator.
        
        Args:
            api_key: Google AI API key. Uses env var if not provided.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._generator = None
    
    @property
    def generator(self) -> PixelArtGenerator:
        """Lazy-load the AI generator."""
        if self._generator is None:
            self._generator = PixelArtGenerator(api_key=self.api_key)
        return self._generator
    
    def generate_spritesheet(
        self,
        subject: str,
        action: str = "idle",
        frames: int = 4,
        cols: int = 4,
        frame_size: int = 64,
        palette_size: int = 24,
        downscale_factor: int = 2,
        clean: bool = True,
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a spritesheet from a text prompt.
        """
        rows = (frames + cols - 1) // cols
        
        print(f"Generating {frames}-frame spritesheet ({cols}x{rows}) for '{subject}'...")
        
        # Generate the spritesheet
        raw_sheet = self.generator.generate_spritesheet(
            subject=subject,
            action=action,
            frames=frames,
            frame_size=frame_size,
            custom_positive=custom_positive,
            custom_negative=custom_negative,
            model_name=model_name,
        )
        
        if clean:
            print("Cleaning spritesheet...")
            # For spritesheets, we use smaller downscale to preserve details
            # We use max dimension as a hint for the grid size
            max_dim = max(raw_sheet.size)
            cleaned = pixel_art_cleaner(
                raw_sheet,
                target_grid=max_dim,
                palette_size=palette_size,
                downscale_factor=downscale_factor,
                remove_background=True,
            )
            return cleaned
        
        return raw_sheet
    
    def generate_from_reference(
        self,
        reference_path: Union[str, Image.Image],
        action: str = "idle",
        frames: int = 4,
        cols: int = 4,
        frame_size: int = 64,
        palette_size: int = 24,
        downscale_factor: int = 2,
        clean: bool = True,
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a spritesheet based on a reference image.
        """
        if isinstance(reference_path, str):
            reference = Image.open(reference_path)
        else:
            reference = reference_path
        
        # Improved prompt for reference generation
        prompt = (
            f"Based on the character/object in this reference image, "
            f"create a {frames}-frame {action} animation spritesheet. "
            f"Arrange frames in a single horizontal row. Maintain consistent character design."
        )
        
        if custom_positive:
            prompt += f", {custom_positive}"
        
        print(f"Generating spritesheet from reference...")
        
        # We don't force a size here to avoid stretching
        raw_sheet = self.generator.generate_from_reference(
            reference_image=reference,
            prompt=prompt,
            size=None, 
            custom_positive=custom_positive,
            custom_negative=custom_negative,
            model_name=model_name,
        )
        
        if clean:
            print("Cleaning spritesheet...")
            max_dim = max(raw_sheet.size)
            cleaned = pixel_art_cleaner(
                raw_sheet,
                target_grid=max_dim,
                palette_size=palette_size,
                downscale_factor=downscale_factor,
                remove_background=True,
            )
            return cleaned
        
        return raw_sheet
    
    def extract_frames(
        self,
        spritesheet: Image.Image,
        frame_size: int,
        cols: int,
        frames: int,
    ) -> List[Image.Image]:
        """
        Extract individual frames from a spritesheet.
        """
        extracted = []
        
        for i in range(frames):
            row = i // cols
            col = i % cols
            
            x = col * frame_size
            y = row * frame_size
            
            frame = spritesheet.crop((x, y, x + frame_size, y + frame_size))
            extracted.append(frame)
        
        return extracted
    
    def create_spritesheet_from_frames(
        self,
        frames: List[Image.Image],
        cols: int = 4,
        padding: int = 0,
    ) -> Image.Image:
        """
        Create a spritesheet from individual frame images.
        """
        if not frames:
            raise ValueError("No frames provided")
        
        frame_width, frame_height = frames[0].size
        rows = (len(frames) + cols - 1) // cols
        
        sheet_width = cols * (frame_width + padding) - padding
        sheet_height = rows * (frame_height + padding) - padding
        
        sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
        
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            
            x = col * (frame_width + padding)
            y = row * (frame_height + padding)
            
            if frame.mode != "RGBA":
                frame = frame.convert("RGBA")
            
            sheet.paste(frame, (x, y))
        
        return sheet
    
    def resize_spritesheet(
        self,
        spritesheet: Image.Image,
        scale: float = 2.0,
    ) -> Image.Image:
        """
        Resize a spritesheet using nearest neighbor (pixel-perfect).
        """
        new_width = int(spritesheet.width * scale)
        new_height = int(spritesheet.height * scale)
        
        return spritesheet.resize((new_width, new_height), Image.Resampling.NEAREST)


def create_spritesheet(
    frames: List[Image.Image],
    cols: int = 4,
    padding: int = 0,
) -> Image.Image:
    """
    Convenience function to create a spritesheet from frames.
    """
    generator = SpritesheetGenerator()
    return generator.create_spritesheet_from_frames(frames, cols, padding)


def split_spritesheet(
    spritesheet_path: str,
    frame_size: int,
    cols: int,
    frames: int,
    output_dir: str,
) -> List[str]:
    """
    Split a spritesheet into individual frame files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    spritesheet = Image.open(spritesheet_path)
    generator = SpritesheetGenerator()
    
    extracted = generator.extract_frames(spritesheet, frame_size, cols, frames)
    
    paths = []
    for i, frame in enumerate(extracted):
        path = os.path.join(output_dir, f"frame_{i:03d}.png")
        frame.save(path)
        paths.append(path)
        print(f"Saved: {path}")
    
    return paths


def combine_images_to_spritesheet(
    image_paths: List[str],
    output_path: str,
    cols: int = 4,
    padding: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Combine multiple image files into a spritesheet.
    """
    frames = []
    
    for path in image_paths:
        img = Image.open(path)
        if target_size:
            # We resize only if requested, using NEAREST to preserve pixels
            img = img.resize(target_size, Image.Resampling.NEAREST)
        frames.append(img)
    
    sheet = create_spritesheet(frames, cols, padding)
    sheet.save(output_path)
    print(f"Spritesheet saved to: {output_path}")
