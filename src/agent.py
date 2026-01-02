"""
PixelArt Agent - Orchestrator for the generation pipeline.
Combines Planner (LLM), Generator (Imagen/Gemini), Processor (Factory), and Critic (Vision).
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

from google import genai
from google.genai import types

from .generator import PixelArtGenerator
from .processor import PixelProcessor
from .linter import PixelLinter
from .smart_cleaner import SmartCleaner, SmartConfig
from .utils import save_image, ensure_rgb

class PixelArtAgent:
    """
    Autonomous agent for creating high-quality pixel art.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required for Agent.")
            
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        # Sub-modules
        self.generator = PixelArtGenerator(api_key=self.api_key)
        self.processor = PixelProcessor()
        self.linter = PixelLinter()
        self.smart_cleaner = SmartCleaner()

    def create_asset(
        self, 
        user_request: str, 
        output_path: str,
        iterations: int = 1
    ) -> Image.Image:
        """
        High-level entry point: Create an asset from a request.
        """
        print(f"Agent received request: {user_request}")
        
        # 1. Plan
        plan = self.plan_task(user_request)
        print(f"Plan: {json.dumps(plan, indent=2)}")
        
        current_image = None
        
        # 2. Execute & Refine Loop
        for i in range(iterations):
            print(f"--- Iteration {i+1}/{iterations} ---")
            
            if current_image is None:
                # Initial Generation
                prompt = plan.get('prompt', user_request)
                size = plan.get('size', (512, 512))
                current_image = self.generator.generate(prompt, size=size)
            else:
                # Refinement (if needed, e.g. based on critique)
                # For now, we just proceed to processing
                pass
            
            # 3. Process (Factory Layer)
            # Smart Clean (Grid Snapping)
            print("Applying Smart Grid Snapping...")
            current_image = self.smart_cleaner.process(current_image)
            
            # Factory Process (Quantize)
            print("Applying Factory Processing...")
            target_grid = plan.get('grid_size', 64)
            palette_size = plan.get('palette_size', 16)
            
            # Resize to target grid if smart cleaner didn't (SmartCleaner outputs logical res)
            # If current_image is small, we keep it. 
            
            current_image = self.processor.process_pipeline(
                current_image,
                target_size=(target_grid, target_grid),
                palette_size=palette_size,
                exact_scaling=False # Input is already snapped/logical from SmartCleaner
            )
            
            # 4. Critique
            print("Running Visual Critique...")
            critique = self.critique_image(current_image, plan)
            print(f"Critique: {critique}")
            
            if critique['score'] >= 8:
                print("Asset passes quality standards.")
                break
            else:
                print("Asset needs refinement (not fully implemented loop).")
                # In a full agent, we would use the critique to modify the prompt or inpaint.
                
        # Save
        save_image(current_image, output_path)
        return current_image

    def plan_task(self, request: str) -> Dict[str, Any]:
        """
        Use LLM to break down the request into technical specifications.
        """
        system_instruction = """
        You are an expert Pixel Art Technical Director.
        Analyze the user request and provide a JSON plan with:
        - prompt: Optimized image generation prompt (detailed, specific).
        - size: [width, height] for generation (usually 512 or 1024).
        - grid_size: Target logical grid size (e.g. 32, 64, 128).
        - palette_size: Recommended number of colors (e.g. 8, 16, 32).
        
        Output valid JSON only.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Request: {request}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Planning failed: {e}. Using defaults.")
            return {
                "prompt": request,
                "size": (512, 512),
                "grid_size": 64,
                "palette_size": 16
            }

    def critique_image(self, image: Image.Image, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Vision model to critique the image.
        """
        # Run Linter first for objective metrics
        linter_report = self.linter.lint(image, expected_palette_size=plan.get('palette_size', 32))
        
        # Prepare image for Vision model
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        
        prompt = f"""
        Analyze this pixel art image against the plan: {json.dumps(plan)}.
        Linter Report: {json.dumps(linter_report)}.
        
        Evaluate:
        1. Consistency of pixel grid (are pixels square and uniform?).
        2. Palette usage (is it muddy or clean?).
        3. Structural integrity (does the subject look correct?).
        
        Provide a JSON response with:
        - score: 1-10 integer.
        - issues: List of strings describing problems.
        - suggestion: String for how to fix.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Critique failed: {e}")
            return {"score": 5, "issues": ["Critique Error"], "suggestion": "Manual review required."}

    def harmonize_palettes(self, assets: List[Image.Image]) -> List[Image.Image]:
        """
        Feature: Dynamic Palette Harmonization.
        Extracts master palette from the first asset and applies it to others.
        """
        if not assets: return []
        
        master_img = assets[0]
        # Extract palette (e.g. 16 colors)
        palette = self.processor.extract_palette(master_img, max_colors=16)
        
        harmonized = [master_img]
        for img in assets[1:]:
            # Apply master palette to other images
            new_img = self.processor.apply_palette(img, palette)
            harmonized.append(new_img)
            
        return harmonized

import io
