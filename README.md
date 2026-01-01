# 👾 PixelArt Factory

**PixelArt Factory** is a professional-grade tool for generating, cleaning, and animating pixel art using Google's latest Generative AI models (Gemini 2.0, 2.5, and 3 Pro). It combines state-of-the-art AI generation with a robust post-processing pipeline to ensure your assets are game-ready.

---

## ✨ Key Features

- **🚀 AI Generation**: Create pixel art from text prompts using Gemini's native image generation.
- **🧼 Smart Cleaning**: An intelligent cleaning algorithm (ported from Rust) that detects your pixel grid, snaps pixels to a perfect alignment, and quantizes colors for a crisp look.
- **🎞️ Spritesheet Creator**: Generate sequential animation frames with high consistency. Optimized for 2D game development.
- **📐 Never Stretch**: Intelligent aspect-ratio preservation ensures your art is never distorted, regardless of target resolution.
- **🎨 Reactive UI**: Real-time cleaning and upscaling preview using Streamlit.
- **🛠️ Game-Ready Prompts**: Built-in optimized prompt templates for 16-bit, 32-bit, and authentic game assets.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/pedrordrigs/pixelart-factory.git
cd pixelart-factory
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API Key
Obtain a Google AI Studio API key from [Google AI Studio](https://aistudio.google.com/).
You can set it as an environment variable or enter it directly in the app.
```bash
# Windows
set GOOGLE_API_KEY=your_key_here
# Linux/Mac
export GOOGLE_API_KEY=your_key_here
```

---

## 🚀 Usage

### Web Application
Run the interactive Streamlit app:
```bash
streamlit run app.py
```

### CLI Mode (Advanced)
Generate an image directly from the terminal:
```bash
python main.py generate --prompt "A cyber knight" --model-name gemini-2.0-flash-exp
```

---

## 💡 Prompting Tips for Better Pixel Art

For optimal results, follow these game-dev oriented tips:
- **Resolution**: Include the resolution in your prompt (e.g., `64x64 resolution`).
- **Style**: Use keywords like `authentic pixel art style`, `indexed colors`, and `flat shading`.
- **Negative Prompts**: Avoid `gradients`, `blur`, and `noise`.

---

## 📂 Project Structure

- `app.py`: The main Streamlit web interface.
- `src/generator.py`: AI interaction logic (Gemini/Google GenAI).
- `src/smart_cleaner.py`: Advanced grid-snapping and color quantization algorithm.
- `src/spritesheet.py`: Spritesheet layout and frame management.
- `src/utils.py`: Image processing utilities.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

Made with ❤️ by [pedrordrigs](https://github.com/pedrordrigs)
