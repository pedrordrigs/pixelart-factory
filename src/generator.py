"""
Image Generator - Uses Google's Generative AI to create pixel art images.
"""

import os
import io
import base64
from typing import Optional, Tuple
from PIL import Image, ImageOps

from google import genai
from google.genai import types


# Optimized prompt templates for pixel art generation - Refined based on game dev best practices
POSITIVE_PROMPT_TEMPLATE = """{subject}, {res_string} resolution, authentic pixel art style, game asset, high-contrast indexed colors, limited color palette, sharp crisp pixels, flat shading, clean thick outlines, orthographic side view, professional sprite design, retro 16-bit aesthetic, isolated on solid white background, no gradients, no blur, aliasing-free"""

NEGATIVE_PROMPT_TEMPLATE = """anti-aliasing, blur, fuzzy, noise, realistic, 3d render, vector, gradients, soft edges, compression artifacts, messy lines, photography, shadow on background, distorted, stretched, blurry, low resolution, interpolation, dithering, bloom, glow, semi-transparent pixels"""

# Spritesheet-specific prompt additions - Enhanced for frame consistency and game-ready alignment
SPRITESHEET_ADDITIONS = """, master spritesheet, animation atlas, sequential action frames, perfect tiled layout, uniform frame dimensions, consistent character design, frame-by-frame animation, identical style across all tiles, centered on grid, neutral pose, game-ready animation strip"""


class PixelArtGenerator:
    """
    Generate pixel art images using Google's Generative AI.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the generator with an API key.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash-exp"
    
    def _build_prompt(
        self,
        subject: str,
        size: Optional[Tuple[int, int]] = (512, 512),
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        is_spritesheet: bool = False,
    ) -> str:
        """
        Build the full prompt for image generation.
        """
        if size:
            width, height = size
            res_string = f"{width}x{height}"
        else:
            res_string = "high"
            
        positive = POSITIVE_PROMPT_TEMPLATE.format(subject=subject, res_string=res_string)
        
        if is_spritesheet:
            positive += SPRITESHEET_ADDITIONS
        
        if custom_positive:
            positive += f", {custom_positive}"
        
        negative = NEGATIVE_PROMPT_TEMPLATE
        if custom_negative:
            negative += f", {custom_negative}"
        
        # Combine positive and negative prompts
        full_prompt = f"{positive}. Avoid: {negative}"
        
        return full_prompt
    
    def _process_image_response(self, response, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Extract and process the image from the API response without stretching.
        """
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    image_data = part.inline_data.data
                    image = Image.open(io.BytesIO(image_data))
                    
                    if target_size:
                        # ImageOps.contain resizes to fit within target_size while PRESERVING aspect ratio.
                        image = ImageOps.contain(image, target_size, Image.Resampling.LANCZOS)
                    
                    return image
        
        raise RuntimeError("Failed to generate image. No image data in response.")

    def generate(
        self,
        prompt: str,
        size: Optional[Tuple[int, int]] = (512, 512),
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a pixel art image from a text prompt.
        """
        full_prompt = self._build_prompt(
            subject=prompt,
            size=size,
            custom_positive=custom_positive,
            custom_negative=custom_negative,
        )
        
        model = model_name or self.model_name
        print(f"Generating image with prompt: {full_prompt[:100]} using {model}...")
        
        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )

            response = self.client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=config
            )
            
            return self._process_image_response(response, target_size=size)
            
        except Exception as e:
            print(f"Generation failed with model {model}: {e}")
            raise e
    
    def generate_from_reference(
        self,
        reference_image: Image.Image,
        prompt: str,
        size: Optional[Tuple[int, int]] = (512, 512),
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a pixel art image based on a reference image.
        """
        full_prompt = self._build_prompt(
            subject=prompt,
            size=size,
            custom_positive=custom_positive,
            custom_negative=custom_negative,
        )
        
        full_prompt = f"Based on the reference image, create a consistent pixel art asset: {full_prompt}"
        
        model = model_name or self.model_name
        print(f"Generating from reference with prompt: {full_prompt[:100]} using {model}...")
        
        img_buffer = io.BytesIO()
        reference_image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/png"
        )
        
        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )
            
            response = self.client.models.generate_content(
                model=model,
                contents=[image_part, full_prompt],
                config=config
            )
            
            return self._process_image_response(response, target_size=size)
            
        except Exception as e:
            print(f"Reference generation failed with model {model}: {e}")
            raise e

    def generate_next_frame(
        self,
        reference_image: Image.Image,
        action: str,
        frame_idx: int,
        total_frames: int,
        size: Optional[Tuple[int, int]] = None,
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate the next sequential frame based on a reference image.
        """
        # Specific prompt for sequential generation
        prompt_core = (
            f"Generate the exact next sequential animation frame for action: '{action}'. "
            f"This is frame {frame_idx + 1} of a {total_frames}-frame loop. "
            f"Continue the movement logically from the provided reference image. "
            f"Maintain exact character consistency, scale, colors, and pixel style. "
            f"The image must be a single isolated sprite on a white background, not a spritesheet."
        )
        
        full_prompt = self._build_prompt(
            subject=prompt_core,
            size=size,
            custom_positive=custom_positive,
            custom_negative=custom_negative,
            is_spritesheet=False # We want a single frame
        )
        
        model = model_name or self.model_name
        print(f"Generating sequential frame {frame_idx + 1}/{total_frames} for '{action}' using {model}...")
        
        img_buffer = io.BytesIO()
        reference_image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/png"
        )
        
        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )
            
            response = self.client.models.generate_content(
                model=model,
                contents=[image_part, full_prompt],
                config=config
            )
            
            return self._process_image_response(response, target_size=size)
            
        except Exception as e:
            print(f"Sequential generation failed with model {model}: {e}")
            raise e
    
    def generate_spritesheet_prompt(
        self,
        subject: str,
        size: Optional[Tuple[int, int]],
        action: str = "idle",
        frames: int = 4,
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
    ) -> str:
        """
        Build a specialized prompt for spritesheet generation.
        """
        if size:
            width, height = size
            res_string = f"Target resolution {width}x{height} pixels. "
        else:
            res_string = ""

        # Highly specific spritesheet prompt for best AI performance and consistency
        positive = (
            f"High-quality game spritesheet of {subject} {action} animation. "
            f"{res_string}"
            f"A single horizontal row containing exactly {frames} separate animation frames. "
            f"Flat tiled view, atlas textures, consistent scale, fixed perspective, and character design across all {frames} frames. "
            f"No distortion, no stretching, perfect alignment. "
            f"Pixel art style, game asset, high-contrast indexed colors, limited color palette, sharp crisp pixels, flat shading, "
            f"clean thick outlines, side view, 16-bit aesthetic, "
            f"isolated on a solid white background with no shadows."
        )
        
        if custom_positive:
            positive += f", {custom_positive}"
        
        negative = NEGATIVE_PROMPT_TEMPLATE
        if custom_negative:
            negative += f", {custom_negative}"
            
        return f"{positive}. Avoid: {negative}"
    
    def generate_spritesheet(
        self,
        subject: str,
        action: str = "idle",
        frames: int = 4,
        frame_size: int = 64,
        custom_positive: Optional[str] = None,
        custom_negative: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a spritesheet image without forcing dimensions.
        """
        # Calculate intended spritesheet size for the prompt
        width = frame_size * frames
        height = frame_size
        
        prompt = self.generate_spritesheet_prompt(
            subject, (width, height), action, frames, custom_positive, custom_negative
        )
        
        model = model_name or self.model_name
        print(f"Generating {frames}-frame spritesheet for: {subject} ({action}) using {model}")
        
        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            
            # For spritesheets, we DO NOT pass a target_size. We want the raw AI output.
            return self._process_image_response(response, target_size=None)
            
        except Exception as e:
            print(f"Spritesheet generation failed with model {model}: {e}")
            raise e


def generate_image(
    prompt: str,
    api_key: Optional[str] = None,
    size: Optional[Tuple[int, int]] = (512, 512),
) -> Image.Image:
    """
    Convenience function to generate an image.
    """
    generator = PixelArtGenerator(api_key=api_key)
    return generator.generate(prompt, size=size)


def generate_from_file(
    reference_path: str,
    prompt: str,
    output_path: str,
    api_key: Optional[str] = None,
    size: Optional[Tuple[int, int]] = (512, 512),
) -> None:
    """
    Generate an image from a reference file and save.
    """
    generator = PixelArtGenerator(api_key=api_key)
    reference = Image.open(reference_path)
    
    result = generator.generate_from_reference(reference, prompt, size=size)
    result.save(output_path)
    print(f"Generated image saved to: {output_path}")
