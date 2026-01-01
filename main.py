#!/usr/bin/env python3
"""
PixelArt Factory - CLI Entry Point

Generate and clean pixel art using AI.
"""

import argparse
import os
import sys
from pathlib import Path

from src.cleaner import pixel_art_cleaner, PixelArtCleaner
from src.generator import PixelArtGenerator
from src.spritesheet import SpritesheetGenerator
from src.utils import load_image, save_image, validate_grid_size


def cmd_generate(args):
    """Generate pixel art from a text prompt."""
    print("=" * 50)
    print("PixelArt Factory - Generate")
    print("=" * 50)
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY or use --api-key")
        sys.exit(1)
    
    # Validate grid size
    grid_size = validate_grid_size(args.grid_size)
    
    print(f"Prompt: {args.prompt}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Pixel Size Factor: {args.pixel_size}")
    print(f"Palette: {args.palette} colors")
    if args.model_name:
        print(f"Model: {args.model_name}")
    print()
    
    # Initialize generator
    generator = PixelArtGenerator(api_key=api_key)
    
    # Generate image
    print("Generating image...")
    raw_image = generator.generate(
        prompt=args.prompt,
        size=(grid_size, grid_size),
        custom_positive=args.style,
        model_name=args.model_name,
    )
    
    # Save raw image if requested
    if args.save_raw:
        raw_path = Path(args.output).stem + "_raw.png"
        raw_image.save(raw_path)
        print(f"Raw image saved to: {raw_path}")
    
    # Clean the image unless --no-clean
    if not args.no_clean:
        print("Cleaning pixel art...")
        final_image = pixel_art_cleaner(
            raw_image,
            target_grid=grid_size,
            palette_size=args.palette,
            downscale_factor=args.pixel_size,
            remove_background=not args.keep_background,
        )
    else:
        final_image = raw_image
    
    # Save final image
    save_image(final_image, args.output)
    print(f"\nDone! Output saved to: {args.output}")


def cmd_clean(args):
    """Clean an existing pixel art image."""
    print("=" * 50)
    print("PixelArt Factory - Clean")
    print("=" * 50)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    grid_size = validate_grid_size(args.grid_size)
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Pixel Size Factor: {args.pixel_size}")
    print(f"Palette: {args.palette} colors")
    print()
    
    # Load image
    print("Loading image...")
    image = load_image(args.input)
    
    # Clean
    print("Cleaning pixel art...")
    cleaned = pixel_art_cleaner(
        image,
        target_grid=grid_size,
        palette_size=args.palette,
        downscale_factor=args.pixel_size,
        remove_background=not args.keep_background,
    )
    
    # Save
    save_image(cleaned, args.output)
    print(f"\nDone! Cleaned image saved to: {args.output}")


def cmd_spritesheet(args):
    """Generate a spritesheet from a prompt or reference image."""
    print("=" * 50)
    print("PixelArt Factory - Spritesheet")
    print("=" * 50)
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY or use --api-key")
        sys.exit(1)
    
    if not args.prompt and not args.input:
        print("Error: Either --prompt or --input is required")
        sys.exit(1)
    
    frame_size = validate_grid_size(args.grid_size)
    
    print(f"Frames: {args.frames}")
    print(f"Columns: {args.cols}")
    print(f"Frame Size: {frame_size}x{frame_size}")
    print(f"Action: {args.action}")
    if args.model_name:
        print(f"Model: {args.model_name}")
    print()
    
    # Initialize generator
    generator = SpritesheetGenerator(api_key=api_key)
    
    if args.input and os.path.exists(args.input):
        # Generate from reference
        print(f"Generating from reference: {args.input}")
        spritesheet = generator.generate_from_reference(
            reference_path=args.input,
            action=args.action,
            frames=args.frames,
            cols=args.cols,
            frame_size=frame_size,
            palette_size=args.palette,
            downscale_factor=args.pixel_size,
            clean=not args.no_clean,
            model_name=args.model_name,
        )
    else:
        # Generate from prompt
        if not args.prompt:
            print("Error: --prompt required when --input is not provided or doesn't exist")
            sys.exit(1)
        
        print(f"Generating from prompt: {args.prompt}")
        spritesheet = generator.generate_spritesheet(
            subject=args.prompt,
            action=args.action,
            frames=args.frames,
            cols=args.cols,
            frame_size=frame_size,
            palette_size=args.palette,
            downscale_factor=args.pixel_size,
            clean=not args.no_clean,
            model_name=args.model_name,
        )
    
    # Save
    save_image(spritesheet, args.output)
    print(f"\nDone! Spritesheet saved to: {args.output}")


def cmd_split(args):
    """Split a spritesheet into individual frames."""
    print("=" * 50)
    print("PixelArt Factory - Split Spritesheet")
    print("=" * 50)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    from src.spritesheet import split_spritesheet
    
    print(f"Input: {args.input}")
    print(f"Frame Size: {args.frame_size}")
    print(f"Columns: {args.cols}")
    print(f"Frames: {args.frames}")
    print(f"Output Dir: {args.output_dir}")
    print()
    
    paths = split_spritesheet(
        spritesheet_path=args.input,
        frame_size=args.frame_size,
        cols=args.cols,
        frames=args.frames,
        output_dir=args.output_dir,
    )
    
    print(f"\nDone! Extracted {len(paths)} frames to: {args.output_dir}")


def cmd_combine(args):
    """Combine multiple images into a spritesheet."""
    print("=" * 50)
    print("PixelArt Factory - Combine Images")
    print("=" * 50)
    
    from src.spritesheet import combine_images_to_spritesheet
    
    # Get input files
    image_paths = []
    for path in args.inputs:
        if os.path.isdir(path):
            # Add all images from directory
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_paths.append(os.path.join(path, f))
        elif os.path.isfile(path):
            image_paths.append(path)
    
    if not image_paths:
        print("Error: No valid image files found")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images")
    print(f"Columns: {args.cols}")
    print(f"Output: {args.output}")
    
    target_size = None
    if args.frame_size:
        target_size = (args.frame_size, args.frame_size)
        print(f"Frame Size: {args.frame_size}x{args.frame_size}")
    
    print()
    
    combine_images_to_spritesheet(
        image_paths=image_paths,
        output_path=args.output,
        cols=args.cols,
        padding=args.padding,
        target_size=target_size,
    )
    
    print(f"\nDone! Spritesheet saved to: {args.output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PixelArt Factory - Generate and clean pixel art using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate pixel art from text
  python main.py generate --prompt "a warrior character" --output warrior.png
  
  # Clean an existing image
  python main.py clean --input messy.png --output clean.png --palette 16
  
  # Generate a spritesheet
  python main.py spritesheet --prompt "slime monster" --action walk --frames 4
  
  # Split a spritesheet into frames
  python main.py split --input sheet.png --frame-size 64 --cols 4 --frames 8
  
  # Combine images into a spritesheet
  python main.py combine ./frames/ --output sheet.png --cols 4
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ========================
    # Generate Command
    # ========================
    gen_parser = subparsers.add_parser("generate", help="Generate pixel art from text prompt")
    gen_parser.add_argument("--prompt", "-p", required=True, help="Text description of the image")
    gen_parser.add_argument("--output", "-o", default="output.png", help="Output file path")
    gen_parser.add_argument("--grid-size", "-g", type=int, default=256,
                           help="Final image size (default: 256)")
    gen_parser.add_argument("--pixel-size", "-x", type=int, default=4,
                           help="Downscale factor for logical pixel size (default: 4)")
    gen_parser.add_argument("--palette", "-c", type=int, default=24,
                           help="Number of colors in palette (default: 24)")
    gen_parser.add_argument("--style", "-s", help="Additional style terms")
    gen_parser.add_argument("--api-key", help="Google AI API key")
    gen_parser.add_argument("--no-clean", action="store_true",
                           help="Skip pixel art cleaning step")
    gen_parser.add_argument("--save-raw", action="store_true",
                           help="Also save the raw generated image")
    gen_parser.add_argument("--keep-background", action="store_true",
                           help="Don't make white background transparent")
    gen_parser.add_argument("--model-name", help="Specific AI model to use")
    gen_parser.set_defaults(func=cmd_generate)
    
    # ========================
    # Clean Command
    # ========================
    clean_parser = subparsers.add_parser("clean", help="Clean an existing pixel art image")
    clean_parser.add_argument("--input", "-i", required=True, help="Input image path")
    clean_parser.add_argument("--output", "-o", default="cleaned.png", help="Output file path")
    clean_parser.add_argument("--grid-size", "-g", type=int, default=256,
                             help="Target grid size (default: 256)")
    clean_parser.add_argument("--pixel-size", "-x", type=int, default=4,
                             help="Downscale factor (default: 4)")
    clean_parser.add_argument("--palette", "-c", type=int, default=24,
                             help="Number of colors (default: 24)")
    clean_parser.add_argument("--keep-background", action="store_true",
                             help="Don't make white background transparent")
    clean_parser.set_defaults(func=cmd_clean)
    
    # ========================
    # Spritesheet Command
    # ========================
    sheet_parser = subparsers.add_parser("spritesheet", help="Generate a spritesheet")
    sheet_parser.add_argument("--prompt", "-p", help="Text description")
    sheet_parser.add_argument("--input", "-i", help="Reference image path")
    sheet_parser.add_argument("--output", "-o", default="spritesheet.png", help="Output file path")
    sheet_parser.add_argument("--action", "-a", default="idle",
                             help="Animation action (idle, walk, run, attack, jump)")
    sheet_parser.add_argument("--frames", "-f", type=int, default=4,
                             help="Number of frames (default: 4)")
    sheet_parser.add_argument("--cols", type=int, default=4,
                             help="Columns in spritesheet (default: 4)")
    sheet_parser.add_argument("--grid-size", "-g", type=int, default=64,
                             help="Size of each frame (default: 64)")
    sheet_parser.add_argument("--pixel-size", "-x", type=int, default=2,
                             help="Downscale factor (default: 2)")
    sheet_parser.add_argument("--palette", "-c", type=int, default=24,
                             help="Number of colors (default: 24)")
    sheet_parser.add_argument("--api-key", help="Google AI API key")
    sheet_parser.add_argument("--no-clean", action="store_true",
                             help="Skip cleaning step")
    sheet_parser.add_argument("--model-name", help="Specific AI model to use")
    sheet_parser.set_defaults(func=cmd_spritesheet)
    
    # ========================
    # Split Command
    # ========================
    split_parser = subparsers.add_parser("split", help="Split spritesheet into frames")
    split_parser.add_argument("--input", "-i", required=True, help="Spritesheet image path")
    split_parser.add_argument("--output-dir", "-o", default="frames",
                             help="Output directory (default: frames)")
    split_parser.add_argument("--frame-size", "-s", type=int, required=True,
                             help="Size of each frame")
    split_parser.add_argument("--cols", type=int, required=True,
                             help="Number of columns in spritesheet")
    split_parser.add_argument("--frames", "-f", type=int, required=True,
                             help="Number of frames to extract")
    split_parser.set_defaults(func=cmd_split)
    
    # ========================
    # Combine Command
    # ========================
    combine_parser = subparsers.add_parser("combine", help="Combine images into spritesheet")
    combine_parser.add_argument("inputs", nargs="+", help="Input image files or directories")
    combine_parser.add_argument("--output", "-o", default="combined.png",
                               help="Output file path")
    combine_parser.add_argument("--cols", type=int, default=4,
                               help="Number of columns (default: 4)")
    combine_parser.add_argument("--frame-size", "-s", type=int,
                               help="Resize frames to this size")
    combine_parser.add_argument("--padding", type=int, default=0,
                               help="Padding between frames (default: 0)")
    combine_parser.set_defaults(func=cmd_combine)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
