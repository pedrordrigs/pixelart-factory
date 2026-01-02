import unittest
import numpy as np
from PIL import Image
from src.processor import PixelProcessor
from src.linter import PixelLinter

class TestFactoryPipeline(unittest.TestCase):
    def setUp(self):
        # Create a dummy image (gradient)
        self.width = 128
        self.height = 128
        self.image = Image.new('RGB', (self.width, self.height))
        
        # Create a gradient
        for y in range(self.height):
            for x in range(self.width):
                r = int(x / self.width * 255)
                g = int(y / self.height * 255)
                b = 128
                self.image.putpixel((x, y), (r, g, b))
                
        self.processor = PixelProcessor()
        self.linter = PixelLinter()

    def test_quantization(self):
        # Quantize to 4 colors
        result = self.processor.process_pipeline(
            self.image, 
            target_size=(32, 32), 
            palette_size=4
        )
        self.assertEqual(result.size, (32, 32))
        
        # Check palette size
        colors = result.getcolors()
        self.assertLessEqual(len(colors), 4)

    def test_linter(self):
        # Create an image with an orphan pixel
        img = Image.new('RGB', (10, 10), (0, 0, 0))
        img.putpixel((5, 5), (255, 255, 255)) # Orphan
        
        report = self.linter.lint(img)
        self.assertEqual(report['orphan_pixels'], 1)
        
        # Fix
        fixed = self.linter.remove_orphans(img)
        report_fixed = self.linter.lint(fixed)
        self.assertEqual(report_fixed['orphan_pixels'], 0)

if __name__ == '__main__':
    unittest.main()
