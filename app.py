import streamlit as st
import os
import io
import json
from PIL import Image
import numpy as np
from google.genai import types

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
from src.processor import PixelProcessor
from src.linter import PixelLinter
from src.agent import PixelArtAgent

# --- Sidebar ---
st.sidebar.title("👾 PixelArt Factory")

mode = st.sidebar.radio(
    "Mode", 
    ["Agent (Autonomous)", "Generate", "Clean Image", "Spritesheet", "Sequential Spritesheet"]
)

api_key = st.sidebar.text_input("Google API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Model Selector
st.sidebar.markdown("### AI Model")
model_name = st.sidebar.selectbox(
    "Select Model",
    options=[
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ],
    index=0,
    help="Select the AI model. Gemini 2.0 Flash is recommended for speed/agent."
)

st.sidebar.markdown("---")
st.sidebar.header("🏭 Factory (Post-Processing)")

# Sidebar strictly for POST-PROCESSING parameters
palette_size = st.sidebar.slider("Post-Process Palette Size", 2, 256, 16, help="Colors to quantize to in the Factory step.")
smart_snap = st.sidebar.checkbox("Smart Grid Snap", value=True, 
                               help="Auto-detect grid size for cleanup.")
output_resolution = st.sidebar.select_slider(
    "Output Resolution",
    options=[64, 128, 256, 512, 1024, 2048],
    value=512,
    help="Final upscale resolution for the asset."
)
upscale_view = st.sidebar.slider("View Scale", 1, 8, 4)
show_lint = st.sidebar.checkbox("Show QA Report", value=False)

# --- Helper Functions ---

def run_factory_pipeline(image, grid_size, p_size, smart_snap, output_res):
    """
    Runs the full factory pipeline with robust handling for Smart Mode.
    grid_size: The logical grid size hint (from Generation or User input)
    output_res: The target resolution for the final image.
    """
    current_image = image
    
    # 1. Determine Logical Grid Size & Clean
    if smart_snap:
        # Use SmartCleaner to find the 'true' grid resolution.
        # We use the requested palette size for the cleaning quantization step.
        cleaner = SmartCleaner(SmartConfig(k_colors=p_size)) 
        smart_result = cleaner.process(current_image)
        smart_w, smart_h = smart_result.size
        
        # Check if detected resolution is significantly lower than user request (detail loss check)
        # If user requested 128 (e.g.) and Smart Snap found 64, we might lose text/details.
        # Threshold: if requested is > 1.2x detected, assume fallback needed.
        if grid_size > smart_w * 1.2:
             # Fallback to manual resize to preserve requested detail
             logical_w, logical_h = grid_size, grid_size
             current_image = image.resize((logical_w, logical_h), Image.Resampling.NEAREST)
             exact_scaling = True
        else:
             current_image = smart_result
             logical_w, logical_h = smart_w, smart_h
             exact_scaling = False
        
    else:
        # Force destructive resize to target logical grid (grid_size)
        # This makes the image 64x64 (or whatever grid_size is)
        logical_w, logical_h = grid_size, grid_size
        
        # Resize raw image to logical grid first to enforce pixelation
        current_image = image.resize((logical_w, logical_h), Image.Resampling.NEAREST)
        exact_scaling = True

    # 2. Factory Processor (Quantize)
    # At this point, current_image is at logical resolution
    processor = PixelProcessor()
    
    # Process pipeline essentially just quantizes now since we handled scaling
    processed_logical = processor.process_pipeline(
        current_image,
        target_size=(logical_w, logical_h),
        palette_size=p_size,
        exact_scaling=False # We handled scaling manually above
    )
    
    # 3. Final Upscale to Output Resolution
    # We want to upscale the logical image (e.g. 64x64) to output_res (e.g. 1024x1024)
    # preserving the pixel art look (Nearest Neighbor)
    
    if output_res:
        # Calculate aspect ratio
        aspect = logical_w / logical_h
        target_w = output_res
        target_h = int(output_res / aspect)
        
        # Or should output_res be the max dimension? 
        # The select_slider suggests square options, but let's fit to max dimension
        if logical_w > logical_h:
            target_w = output_res
            target_h = int(output_res / aspect)
        else:
            target_h = output_res
            target_w = int(output_res * aspect)
            
        final_image = processed_logical.resize((target_w, target_h), Image.Resampling.NEAREST)
    else:
        final_image = processed_logical
    
    return final_image

def upscale_image(image, factor):
    w, h = image.size
    return image.resize((w * factor, h * factor), Image.Resampling.NEAREST)

def display_linter_report(image, p_size):
    linter = PixelLinter()
    report = linter.lint(image, expected_palette_size=p_size)
    
    st.markdown("### 📋 QA Report")
    c1, c2, c3 = st.columns(3)
    
    passed = report['palette_check']['passed']
    count = report['palette_check']['count']
    c1.metric("Palette", f"{count}/{p_size}", "PASS" if passed else "FAIL", delta_color="normal" if passed else "inverse")
    
    orphans = report['orphan_pixels']
    c2.metric("Orphan Pixels", orphans, "CLEAN" if orphans == 0 else "NOISY", delta_color="inverse" if orphans > 0 else "normal")
    
    grid_consistent = report['grid_consistency']['consistent']
    c3.metric("Grid Lock", "YES" if grid_consistent else "NO", delta_color="normal" if grid_consistent else "inverse")

def optimize_prompt(api_key, user_prompt, perspective=None, model_name="gemini-2.0-flash-exp"):
    """Use Gemini to optimize the user prompt for pixel art generation."""
    # Force a text-capable model for prompt engineering
    text_model = "gemini-2.0-flash-exp" 
    
    agent = PixelArtAgent(api_key=api_key, model_name=text_model)
    
    perspective_text = f"The user has explicitly requested a {perspective}." if perspective else ""
    
    system_instruction = f"""
    You are an AI Prompt Engineer for Pixel Art. 
    Rewrite the user's request into a high-quality, descriptive prompt for an AI image generator (like Imagen 3).
    
    CRITICAL: You MUST respect the user's requested perspective: {perspective if perspective else 'Any'}.
    {perspective_text}
    
    Include specific pixel art terms: 'clean outlines', 'limited palette', 'high contrast', 'no gradients', '16-bit aesthetic', '{perspective} view'.
    Return ONLY the optimized prompt text.
    """
    try:
        response = agent.client.models.generate_content(
            model=text_model,
            contents=f"Optimize this prompt for pixel art: {user_prompt}",
            config=types.GenerateContentConfig(system_instruction=system_instruction)
        )
        return response.text.strip()
    except Exception as e:
        st.warning(f"Prompt optimization failed: {e}")
        return user_prompt

# --- Main Content ---

if mode == "Agent (Autonomous)":
    st.title("🤖 PixelArt Autonomous Agent")
    st.caption("Describe what you want, and the Agent will plan, generate, process, and review the asset.")
    
    c_p1, c_p2 = st.columns([2, 1])
    with c_p1:
        prompt_input = st.text_area("What should the agent build?", "A cyberpunk street food vendor stall, night time, neon lights")
        enhance_prompt = st.checkbox("Enhance Prompt (AI)", value=True, help="Use Gemini to optimize the prompt before generation.")
    with c_p2:
        st.markdown("### Generation Settings")
        # Generation Parameters MOVED TO PAGE
        gen_perspective = st.selectbox("Perspective", ["Orthographic Side", "Isometric", "Top-Down", "Front View", "3/4 View"], index=0)
        gen_grid_size = st.select_slider("Target Resolution", [16, 32, 64, 128, 256, 512, 1024], value=64, help="Logical grid size used for generation hints.")
        gen_palette_hint = st.slider("Generation Palette Hint", 2, 256, 16, key="agent_palette_hint", help="Suggests color count to the Agent/Model.")
    
    if st.button("🚀 Launch Agent"):
        if not api_key:
            st.error("Agent requires API Key.")
        else:
            agent = PixelArtAgent(api_key=api_key, model_name=model_name)
            status_container = st.status("Agent Working...", expanded=True)
            
            try:
                # 1. Planning & Optimization
                status_container.write("🧠 Planning task...")
                
                current_prompt = prompt_input
                
                # Explicit Prompt Enhancement Step
                if enhance_prompt:
                    status_container.write("✨ Optimizing prompt with AI...")
                    current_prompt = optimize_prompt(api_key, prompt_input, perspective=gen_perspective, model_name="gemini-2.0-flash-exp")
                
                # Plan technical details based on the (possibly optimized) prompt
                plan = agent.plan_task(current_prompt)
                
                # Ensure the plan uses our optimized prompt
                plan['prompt'] = current_prompt
                
                # Override plan with UI constraints
                plan['grid_size'] = gen_grid_size
                plan['palette_size'] = gen_palette_hint
                
                # Show plan details
                status_container.markdown(f"**Plan:** Grid `{plan.get('grid_size')}`, Palette `{plan.get('palette_size')}` colors")
                status_container.markdown(f"**Optimized Prompt:** `{plan.get('prompt')}`")
                
                # 2. Generation
                status_container.write("🎨 Generating base asset...")
                raw_image = agent.generator.generate(
                    plan.get('prompt', prompt_input), 
                    size=plan.get('size', (512, 512)),
                    perspective=gen_perspective.lower(),
                    model_name=model_name,
                    logical_resolution=gen_grid_size
                )
                st.session_state['agent_raw'] = raw_image
                # Store the generation grid size to use in pipeline
                st.session_state['agent_target_grid'] = gen_grid_size
                
                status_container.update(label="Generation Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status_container.update(label="Mission Failed", state="error")
                st.error(f"Agent Error: {e}")

    if 'agent_raw' in st.session_state:
        st.divider()
        raw_image = st.session_state['agent_raw']
        # Use the stored target grid from generation, or default
        target_grid = st.session_state.get('agent_target_grid', 64)
        
        # Run pipeline with Sidebar Post-Process settings + Page Generation settings
        with st.spinner("🏭 Running Factory Post-Processing..."):
            # Use the Sidebar palette_size for the actual Factory processing
            final_image = run_factory_pipeline(raw_image, target_grid, palette_size, smart_snap, output_resolution)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            view_img = upscale_image(final_image, upscale_view)
            st.image(view_img, caption=f"Final Asset ({final_image.size})", use_container_width=True)
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            st.download_button("Download Asset", buf.getvalue(), "agent_asset.png", "image/png")
        with c2:
            st.image(raw_image, caption="Raw AI Generation", use_container_width=True)
        if show_lint:
            display_linter_report(final_image, palette_size)

elif mode == "Generate":
    st.title("Generate Pixel Art")
    
    c_p1, c_p2 = st.columns([2, 1])
    with c_p1:
        prompt_input = st.text_area("Prompt", "A brave knight, pixel art style")
        use_ai_opt = st.checkbox("AI Prompt Optimization", value=True)
        custom_positive = st.text_input("Style Tags", "game asset")
        custom_negative = st.text_input("Negative", "blur, noise, realistic")
    with c_p2:
        st.markdown("### Generation Settings")
        gen_perspective = st.selectbox("Perspective", ["Orthographic Side", "Isometric", "Top-Down", "Front View", "3/4 View"], index=0)
        gen_grid_size = st.select_slider("Target Resolution", [16, 32, 64, 128, 256, 512, 1024], value=64)
        gen_palette_hint = st.slider("Generation Palette Hint", 2, 256, 16, key="gen_palette_hint")
    
    if st.button("Generate"):
        if not api_key:
            st.error("API Key required.")
        else:
            with st.spinner("Generating..."):
                final_prompt = prompt_input
                if use_ai_opt:
                    final_prompt = optimize_prompt(api_key, prompt_input, perspective=gen_perspective, model_name=model_name)
                    st.info(f"🚀 **Optimized Prompt:** {final_prompt}")
                
                gen = PixelArtGenerator(api_key=api_key)
                raw = gen.generate(
                    final_prompt, 
                    size=(512, 512), 
                    perspective=gen_perspective.lower(),
                    custom_positive=custom_positive, 
                    custom_negative=custom_negative, 
                    model_name=model_name,
                    logical_resolution=gen_grid_size
                )
                st.session_state['gen_raw'] = raw
                st.session_state['gen_target_grid'] = gen_grid_size
                st.session_state['gen_final_prompt'] = final_prompt

    if 'gen_raw' in st.session_state:
        st.divider()
        
        # Show Generation Details
        if 'gen_final_prompt' in st.session_state:
            with st.expander("📝 Generation Details", expanded=False):
                st.info(f"**Used Prompt:** {st.session_state['gen_final_prompt']}")

        raw = st.session_state['gen_raw']
        target_grid = st.session_state.get('gen_target_grid', 64)
        
        with st.spinner("🏭 Running Factory Post-Processing..."):
            final = run_factory_pipeline(raw, target_grid, palette_size, smart_snap, output_resolution)
        
        c1, c2 = st.columns(2)
        with c1:
            view = upscale_image(final, upscale_view)
            st.image(view, caption=f"Factory Output ({final.size})", use_container_width=True)
            buf = io.BytesIO()
            final.save(buf, format="PNG")
            st.download_button("Download (Native)", buf.getvalue(), "asset.png", "image/png")
        with c2:
            st.image(raw, caption="Raw Input", use_container_width=True)
        if show_lint:
            display_linter_report(final, palette_size)

elif mode == "Clean Image":
    st.title("Clean Existing Image")
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    # For Clean Mode, we need target grid on page as it's the primary "Goal"
    clean_target_grid = st.select_slider("Target Resolution Hint", [16, 32, 64, 128, 256, 512, 1024], value=64)
    # clean_palette_size = st.slider("Palette Size", 2, 256, 16, key="clean_palette_size") # Removed per request to use sidebar
    
    if uploaded:
        image = Image.open(uploaded)
        st.session_state['clean_raw'] = image
        
    if 'clean_raw' in st.session_state:
        st.divider()
        raw = st.session_state['clean_raw']
        # Use sidebar post-process settings + on-page target grid
        final = run_factory_pipeline(raw, clean_target_grid, palette_size, smart_snap, output_resolution)
        
        c1, c2 = st.columns(2)
        with c1:
            view = upscale_image(final, upscale_view)
            st.image(view, caption=f"Cleaned ({final.size})", use_container_width=True)
            buf = io.BytesIO()
            final.save(buf, format="PNG")
            st.download_button("Download Cleaned", buf.getvalue(), "cleaned.png", "image/png")
        with c2:
            st.image(raw, caption="Original", use_container_width=True)
        if show_lint:
            display_linter_report(final, palette_size)

elif mode == "Spritesheet":
    st.title("Generate Spritesheet")
    
    c_p1, c_p2 = st.columns([2, 1])
    with c_p1:
        prompt_input = st.text_input("Character", "Slime monster")
        action = st.text_input("Action", "idle")
        use_ai_opt = st.checkbox("AI Prompt Optimization", value=True, key="ss_ai_opt")
    with c_p2:
        st.markdown("### Sheet Settings")
        frames = st.number_input("Frames", 2, 32, 4)
        gen_perspective = st.selectbox("Perspective", ["Orthographic Side", "Isometric", "Top-Down", "Front View", "3/4 View"], index=0, key="ss_persp")
        gen_grid_size = st.select_slider("Frame Resolution", [16, 32, 64, 128, 256], value=64, key="ss_grid")
        gen_palette_hint = st.slider("Generation Palette Hint", 2, 256, 16, key="ss_palette_hint")
    
    if st.button("Generate Sheet"):
        if not api_key:
            st.error("API Key required.")
        else:
            with st.spinner("Generating..."):
                final_prompt = prompt_input
                if use_ai_opt:
                    final_prompt = optimize_prompt(api_key, f"{prompt_input} {action} spritesheet", perspective=gen_perspective, model_name=model_name)
                    st.info(f"🚀 **Optimized Prompt:** {final_prompt}")

                gen = SpritesheetGenerator(api_key=api_key)
                
                raw_sheet = gen.generate_spritesheet(
                    subject=final_prompt, 
                    action=action, 
                    frames=frames, 
                    frame_size=gen_grid_size, 
                    clean=False,
                    model_name=model_name,
                    custom_positive=f"{gen_perspective.lower()} view"
                )
                st.session_state['sheet_raw'] = raw_sheet
                st.session_state['sheet_target_grid'] = gen_grid_size

    if 'sheet_raw' in st.session_state:
        st.divider()
        raw = st.session_state['sheet_raw']
        target_grid = st.session_state.get('sheet_target_grid', 64)
        
        # For spritesheets, target_grid is usually frame size, but pipeline expects total image size intent?
        # No, SmartCleaner is good at handling sheets. 
        # But if we force resize, we should probably be careful.
        # Let's pass the raw sheet dimensions as target if smart snap is on, 
        # or calculate total width based on frames if manual.
        
        # Simple heuristic: trust Smart Snap for sheets mostly.
        # If manual, we assume user wants to keep resolution roughly same as generation intent
        
        final = run_factory_pipeline(raw, target_grid, palette_size, smart_snap, output_resolution)
        st.image(upscale_image(final, upscale_view), use_container_width=True, caption=f"Result ({final.size})")
        buf = io.BytesIO()
        final.save(buf, format="PNG")
        st.download_button("Download Sheet", buf.getvalue(), "sheet.png", "image/png")

elif mode == "Sequential Spritesheet":
    st.title("Sequential Generation")
    
    start_file = st.file_uploader("Start Frame", type=['png', 'jpg'])
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        action_seq = st.text_input("Action", "walking")
        frames_seq = st.number_input("Frames", 2, 16, 4)
    with c_s2:
        gen_grid_size = st.select_slider("Frame Resolution", [16, 32, 64, 128, 256], value=64, key="seq_grid")
        # gen_palette_size = st.slider("Palette Size", 2, 256, 16, key="seq_palette_size") # Removed per request
    
    if st.button("Generate Sequence"):
        if not api_key or not start_file:
            st.error("Missing Info")
        else:
            with st.spinner("Generating sequence..."):
                gen = SpritesheetGenerator(api_key=api_key)
                start_img = Image.open(start_file)
                
                sheet, zip_data = gen.generate_sequential_spritesheet(
                    start_img, action_seq, frames_seq, 
                    frame_size=gen_grid_size,
                    model_name=model_name
                )
                st.session_state['seq_sheet'] = sheet
                st.session_state['seq_zip'] = zip_data
                st.session_state['seq_target_grid'] = gen_grid_size

    if 'seq_sheet' in st.session_state:
        st.divider()
        sheet = st.session_state['seq_sheet']
        target_grid = st.session_state.get('seq_target_grid', 64)
        
        final = run_factory_pipeline(sheet, target_grid, palette_size, smart_snap, output_resolution)
        st.image(upscale_image(final, upscale_view), caption="Sequential Sheet", use_container_width=True)
        b1 = io.BytesIO()
        final.save(b1, format="PNG")
        st.download_button("Download Sheet", b1.getvalue(), "seq_sheet.png", "image/png")
        st.download_button("Download Frames ZIP", st.session_state['seq_zip'], "frames.zip", "application/zip")

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ by PixelArt Factory")
