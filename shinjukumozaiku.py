import cv2
import numpy as np
import io
import streamlit as st
from PIL import Image
import easyocr
from transformers import pipeline
from streamlit_image_coordinates import streamlit_image_coordinates
from huggingface_hub import login  # <-- ADDED IMPORT


def load_css():
	st.markdown("""
    <style>
        /* Neon text glow for main title */
        h1 {
            color: #fff !important;
            text-shadow:
                0 0 5px #fff,
                0 0 10px #fff,
                0 0 20px #ff2a6d,
                0 0 40px #ff2a6d,
                0 0 80px #ff2a6d;
            font-family: 'Courier New', Courier, monospace;
            text-transform: uppercase;
            letter-spacing: 4px;
        }

        /* Glitchy/Neon subtext */
        p, .st-emotion-cache-16idsys p {
            color: #d1f7ff !important;
            text-shadow: 0 0 3px #05d9e8;
        }

        /* Style the standard Streamlit buttons to look futuristic */
        div.stButton > button {
            background-color: transparent !important;
            color: #05d9e8 !important;
            border: 2px solid #05d9e8 !important;
            box-shadow: 0 0 10px #05d9e8 inset, 0 0 10px #05d9e8;
            transition: all 0.2s ease-in-out;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: bold;
        }

        div.stButton > button:hover {
            background-color: #05d9e8 !important;
            color: #01012b !important;
            box-shadow: 0 0 20px #05d9e8 inset, 0 0 20px #05d9e8;
            transform: scale(1.02);
        }

        /* The primary download button gets the Pink/Magenta treatment */
        div.stButton > button[kind="primary"] {
            color: #ff2a6d !important;
            border: 2px solid #ff2a6d !important;
            box-shadow: 0 0 10px #ff2a6d inset, 0 0 10px #ff2a6d;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #ff2a6d !important;
            color: #fff !important;
            box-shadow: 0 0 20px #ff2a6d inset, 0 0 20px #ff2a6d;
        }

        /* Container styling for the image */
        [data-testid="stImage"] {
            border: 2px solid #ff2a6d;
            box-shadow: 0 0 15px #ff2a6d;
            border-radius: 4px;
            padding: 5px;
            background: #000;
        }
    </style>
    """, unsafe_allow_html=True)


# --- Config & Initialization ---
st.set_page_config(page_title="ShinjukuMozaiku - A Reliable Privacy Guard", page_icon="🌃", layout="centered")
load_css()


@st.cache_resource(show_spinner="Booting PyTorch Vision & Hugging Face NLP Cores...")
def load_ai_models():
    # 1. Securely fetch the token from Streamlit Secrets
    hf_token = st.secrets.get("HF_TOKEN")
    
    if hf_token:
        login(token=hf_token)
    else:
        st.warning("System Alert: HF_TOKEN missing. Operating at restricted bandwidth.")
    
    # EasyOCR replaces PaddleOCR
    reader = easyocr.Reader(['en'], gpu=False)
    # Hugging Face NER to intelligently identify sensitive information
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return reader, ner


ocr_reader, hf_ner = load_ai_models()


# --- Core Logic ---
def detect_boxes(image_np):
	"""Run EasyOCR to find text, then Hugging Face NER to flag sensitive data."""
	# Read text from image
	result = ocr_reader.readtext(image_np)
	boxes = []
	h, w = image_np.shape[:2]
	
	for i, (bbox, text, confidence) in enumerate(result):
		# Extract bounding box coordinates
		xs = [pt[0] for pt in bbox]
		ys = [pt[1] for pt in bbox]
		x_min, y_min = max(0, int(min(xs)) - 5), max(0, int(min(ys)) - 5)
		x_max, y_max = min(w, int(max(xs)) + 5), min(h, int(max(ys)) + 5)
		
		# Hugging Face AI check: Is this text sensitive?
		entities = hf_ner(text)
		is_sensitive = len(entities) > 0  # True if it found a Person, Location, etc.
		
		boxes.append({
			"id": i,
			"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min,
			"active": is_sensitive,  # Smart default based on NLP
			"text": text
		})
	return boxes


def draw_overlays(image_np, boxes):
	"""Draw semi-transparent Pink (active) or Cyan (inactive) boxes."""
	overlay = image_np.copy()
	for b in boxes:
		# BGR Colors for OpenCV: Hot Pink for active (blur), Neon Cyan for inactive (keep)
		color = (109, 42, 255) if b["active"] else (232, 217, 5)
		cv2.rectangle(overlay, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, -1)
	# Blend overlay with original image (alpha=0.4 mimics your CSS opacity)
	return cv2.addWeighted(overlay, 0.4, image_np, 0.6, 0)


def apply_blur(image_np, boxes):
	"""Apply Gaussian Blur to active regions."""
	img = image_np.copy()
	for b in boxes:
		if b["active"]:
			x, y, w, h = b["x"], b["y"], b["w"], b["h"]
			roi = img[y:y + h, x:x + w]
			if roi.size > 0:
				img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (51, 51), 30)
	return img


# --- UI / Frontend ---
st.title("🌃 ShinjukuMozaiku - A Privacy Tool")
st.markdown(
	"Upload target visual data. AI Auto-Flags sensitive text. Tap highlighted sectors to toggle localized obfuscation. Execute download sequence when ready.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
	# 1. State Management for new uploads
	if "last_upload" not in st.session_state or st.session_state.last_upload != uploaded_file.name:
		st.session_state.last_upload = uploaded_file.name
		st.session_state.last_click = None  # Reset click tracker
		
		# Read and convert image
		file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
		img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		st.session_state.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		# Run OCR + NLP
		with st.spinner("Extracting structural data & scanning for sensitive entities..."):
			st.session_state.boxes = detect_boxes(st.session_state.img_rgb)
	
	# 2. Render Interactive Image
	display_img = draw_overlays(st.session_state.img_rgb, st.session_state.boxes)
	click_data = streamlit_image_coordinates(Image.fromarray(display_img), key="interactive_img")
	
	# 3. Handle Clicks (Toggle Box State)
	if click_data is not None:
		click_id = f"{click_data['x']}_{click_data['y']}"
		# Only process if this is a new click
		if st.session_state.get("last_click") != click_id:
			st.session_state["last_click"] = click_id
			cx, cy = click_data["x"], click_data["y"]
			
			# Check which box was clicked
			for b in st.session_state.boxes:
				if b["x"] <= cx <= b["x"] + b["w"] and b["y"] <= cy <= b["y"] + b["h"]:
					b["active"] = not b["active"]
					st.rerun()  # Force UI refresh to update colors
					break
	
	# 4. Toolbar & Download Section
	active_count = sum(1 for b in st.session_state.boxes if b["active"])
	st.info(
		f"Detected **{len(st.session_state.boxes)}** text nodes. AI Auto-Targeted **{active_count}** for obfuscation.")
	
	col1, col2 = st.columns([1, 1])
	with col1:
		if st.button("🗑️ Abort / Discard", use_container_width=True):
			st.session_state.clear()
			st.rerun()
	
	with col2:
		# Generate final image in memory for download
		blurred_img = apply_blur(st.session_state.img_rgb, st.session_state.boxes)
		buf = io.BytesIO()
		Image.fromarray(blurred_img).save(buf, format="JPEG")
		
		st.download_button(
			label="⬇️ Execute Obfuscation & Download",
			data=buf.getvalue(),
			file_name="shinjuku-mozaiku-protected.jpg",
			mime="image/jpeg",
			type="primary",
			use_container_width=True
		)