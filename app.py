import streamlit as st
import os
import io
from PIL import Image
import numpy as np

# Set page config first
st.set_page_config(
    page_title="PixelArt Factory",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.generator import PixelArtGenerator
from src.smart_cleaner import SmartCleaner, SmartConfig
from src.spritesheet import SpritesheetGenerator

# --- Sidebar ---
st.sidebar.title("👾 PixelArt Factory")

mode = st.sidebar.radio("Mode", ["Generate", "Clean Image", "Spritesheet", "Sequential Spritesheet"])

api_key = st.sidebar.text_input("Google API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Model Selector - Removed Imagen models
st.sidebar.markdown("### AI Model")
model_name = st.sidebar.selectbox(
    "Select Model",
    options=[
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ],
    index=0,
    help="Select the AI model. These Gemini models support native image generation."
)

st.sidebar.header("Cleaning Settings")

# Only Smart Mode settings now
st.sidebar.caption("Auto-detects grid and snaps pixels.")

# Reactive settings (changes trigger rerun)
smart_k = st.sidebar.slider("Quantization Colors", 2, 64, 16, help="Number of colors to reduce to.")
upscale_factor = st.sidebar.slider("Upscale Output", 1, 8, 4, help="Scale the final pixel art up (nearest neighbor).")

# --- Functions ---

def clean_image_reactive(image, k_colors, scale_factor):
    """
    Apply Smart Cleaning and resize back to a multiple of logical size.
    Preserves aspect ratio and avoids stretching.
    """
    # 1. Smart Clean (returns logical resolution image)
    config = SmartConfig(k_colors=k_colors)
    cleaner = SmartCleaner(config)
    
    # logical_img is usually small (e.g. 64x64 for a 256x256 input)
    logical_img = cleaner.process(image)
    
    # 2. Upscale
    # We simply multiply dimensions by scale_factor
    w, h = logical_img.size
    new_w, new_h = w * scale_factor, h * scale_factor
    
    return logical_img.resize((new_w, new_h), Image.Resampling.NEAREST)

# --- Main Content ---

if mode == "Generate":
    st.title("Generate Pixel Art")
    
    prompt = st.text_area("Prompt", "A brave knight, pixel art style")
    
    # Generation Settings
    col1, col2 = st.columns(2)
    with col1:
        custom_positive = st.text_input("Extra Style Tags", "terraria style, vibrant")
        gen_size = st.select_slider("Generation Resolution", options=[256, 512, 1024], value=512)
    with col2:
        custom_negative = st.text_input("Negative Prompts", "blur, noise")
        
    if st.button("Generate New"):
        if not api_key:
            st.error("Please provide a Google API Key in the sidebar.")
        else:
            with st.spinner("Generating..."):
                try:
                    generator = PixelArtGenerator(api_key=api_key)
                    # Pass the selected model explicitly
                    
                    raw_image = generator.generate(
                        prompt, 
                        size=(gen_size, gen_size),
                        custom_positive=custom_positive,
                        custom_negative=custom_negative,
                        model_name=model_name
                    )
                    st.session_state['gen_raw_image'] = raw_image
                    
                except Exception as e:
                    st.error(f"Error with model {model_name}: {e}")
                    if model_name != "gemini-2.0-flash-exp":
                        st.info("Tip: Try switching back to 'gemini-2.0-flash-exp' if other models fail.")

    # Display & Real-time Cleaning
    if 'gen_raw_image' in st.session_state:
        raw_image = st.session_state['gen_raw_image']
        
        # Clean reactive
        cleaned_image = clean_image_reactive(raw_image, smart_k, upscale_factor)
        
        col_raw, col_clean = st.columns(2)
        with col_raw:
            st.subheader("Raw AI Output")
            st.image(raw_image, use_container_width=True)
            
        with col_clean:
            st.subheader("Cleaned Pixel Art")
            st.image(cleaned_image, use_container_width=True, caption=f"Size: {cleaned_image.size}")
            
            buf = io.BytesIO()
            cleaned_image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="pixel_art.png",
                mime="image/png"
            )

elif mode == "Clean Image":
    st.title("Clean Existing Image")
    
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        # Load once
        image = Image.open(uploaded_file)
        
        # Clean reactive
        cleaned_image = clean_image_reactive(image, smart_k, upscale_factor)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Cleaned")
            st.image(cleaned_image, use_container_width=True, caption=f"Size: {cleaned_image.size}")
            
            buf = io.BytesIO()
            cleaned_image.save(buf, format="PNG")
            st.download_button(
                label="Download Cleaned",
                data=buf.getvalue(),
                file_name="cleaned_pixel_art.png",
                mime="image/png"
            )

elif mode == "Spritesheet":
    st.title("Generate Spritesheet")
    
    # Reference upload
    st.markdown("### 1. Reference (Optional)")
    reference_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], help="Upload an image to base the style/character on.")
    
    st.markdown("### 2. Prompt Settings")
    prompt = st.text_input("Character/Subject", "A robot")
    action = st.text_input("Action", "walking")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        custom_positive = st.text_input("Extra Style Tags", "terraria style, vibrant", key="sheet_pos")
    with col_s2:
        custom_negative = st.text_input("Negative Prompts", "blur, noise", key="sheet_neg")
        
    st.markdown("### 3. Layout Settings")
    c1, c2, c3 = st.columns(3)
    frames = c1.number_input("Frames", 2, 32, 4)
    cols = c2.number_input("Columns", 1, 16, 4)
    # Increased frame size limits
    frame_size_hint = c3.select_slider("Approx Frame Size (px)", [32, 64, 128, 256, 512, 1024], value=64)
    
    if st.button("Generate Sheet"):
        if not api_key:
            st.error("API Key required")
        else:
            with st.spinner("Generating Spritesheet..."):
                try:
                    sheet_gen = SpritesheetGenerator(api_key=api_key)
                    # Use the selected model
                    
                    if reference_file:
                        # Generate from reference
                        ref_image = Image.open(reference_file)
                        raw_sheet = sheet_gen.generate_from_reference(
                            reference_path=ref_image,
                            action=action,
                            frames=frames,
                            cols=cols,
                            frame_size=frame_size_hint,
                            clean=False,
                            custom_positive=custom_positive,
                            custom_negative=custom_negative,
                            model_name=model_name
                        )
                    else:
                        # Generate from prompt
                        raw_sheet = sheet_gen.generate_spritesheet(
                            subject=prompt,
                            action=action,
                            frames=frames,
                            cols=cols,
                            frame_size=frame_size_hint,
                            clean=False,
                            custom_positive=custom_positive,
                            custom_negative=custom_negative,
                            model_name=model_name
                        )
                    
                    st.session_state['sheet_raw_image'] = raw_sheet
                    
                except Exception as e:
                    st.error(f"Error with model {model_name}: {e}")
                    if model_name != "gemini-2.0-flash-exp":
                        st.info("Tip: Try switching back to 'gemini-2.0-flash-exp' if other models fail.")

    # Display & Real-time Cleaning
    if 'sheet_raw_image' in st.session_state:
        raw_sheet = st.session_state['sheet_raw_image']
        
        # Clean reactive (preserves aspect ratio!)
        cleaned_sheet = clean_image_reactive(raw_sheet, smart_k, upscale_factor)
        
        st.subheader("Result")
        st.image(cleaned_sheet, use_container_width=True, caption=f"Cleaned Spritesheet ({cleaned_sheet.size})")
        
        col_raw, col_dwn = st.columns(2)
        with col_raw:
            with st.expander("See Raw Output"):
                st.image(raw_sheet, use_container_width=True)
        
        with col_dwn:
            buf = io.BytesIO()
            cleaned_sheet.save(buf, format="PNG")
            st.download_button("Download Sheet", buf.getvalue(), "spritesheet.png", "image/png")

elif mode == "Sequential Spritesheet":
    st.title("Sequential Spritesheet Generation")
    st.markdown("Generate animation frames one-by-one from a starting image.")
    
    # 1. Input Image (Required)
    st.markdown("### 1. Input Frame")
    seq_input_file = st.file_uploader("Upload Starting Frame", type=["png", "jpg", "jpeg"], key="seq_input")
    
    if seq_input_file:
        st.image(seq_input_file, width=128, caption="Frame 1 (Start)")
    
    # 2. Settings
    st.markdown("### 2. Animation Settings")
    seq_action = st.text_input("Action Name", "walking cycle", key="seq_action", help="Describe the movement.")
    
    c1, c2 = st.columns(2)
    seq_frames = c1.number_input("Total Frames", 2, 16, 4, key="seq_frames", help="Total frames including the first one.")
    seq_res = c2.select_slider("Frame Resolution", [32, 64, 128, 256, 512], value=64, key="seq_res")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        custom_positive = st.text_input("Extra Style Tags", "consistent character", key="seq_pos")
    with col_s2:
        custom_negative = st.text_input("Negative Prompts", "morphing, style change", key="seq_neg")
    
    # 3. Generate
    if st.button("Generate Sequence"):
        if not api_key:
            st.error("API Key required")
        elif not seq_input_file:
            st.error("Please upload a starting frame first.")
        else:
            with st.spinner(f"Generating {seq_frames} frames sequentially... This may take a moment."):
                try:
                    sheet_gen = SpritesheetGenerator(api_key=api_key)
                    start_img = Image.open(seq_input_file)
                    
                    # Call sequential generation
                    sheet, zip_bytes = sheet_gen.generate_sequential_spritesheet(
                        reference_path=start_img,
                        action=seq_action,
                        frames=seq_frames,
                        frame_size=seq_res,
                        custom_positive=custom_positive,
                        custom_negative=custom_negative,
                        model_name=model_name
                    )
                    
                    st.session_state['seq_sheet_image'] = sheet
                    st.session_state['seq_zip_data'] = zip_bytes
                    
                except Exception as e:
                    st.error(f"Error: {e}")

    # 4. Results
    if 'seq_sheet_image' in st.session_state:
        sheet = st.session_state['seq_sheet_image']
        zip_data = st.session_state['seq_zip_data']
        
        # Clean reactive
        cleaned_sheet = clean_image_reactive(sheet, smart_k, upscale_factor)
        
        st.subheader("Sequential Result")
        st.image(cleaned_sheet, use_container_width=True, caption="Combined Sequence")
        
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            buf = io.BytesIO()
            cleaned_sheet.save(buf, format="PNG")
            st.download_button("Download Sheet (PNG)", buf.getvalue(), "seq_spritesheet.png", "image/png")
            
        with c_d2:
            st.download_button("Download All Frames (ZIP)", zip_data, "frames.zip", "application/zip")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using PixelArt Factory")
