"""
Siddhant AI Studio - text to image generator (Streamlit)
Engine: Hugging Face FLUX (HF_TOKEN) with Pollinations fallback. Auto-retry on every engine.
"""

import io
import os
import random
import time
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# ---------------------------------------------------------------- settings
ASPECTS = {
    "Square  1:1": (1024, 1024),
    "Landscape  16:9": (1344, 768),
    "Portrait  9:16": (768, 1344),
    "Classic  4:3": (1152, 896),
}

STYLES = {
    "None": "",
    "Photorealistic": "ultra realistic photograph, 85mm lens, natural lighting, sharp focus, highly detailed",
    "Cinematic": "cinematic still, dramatic lighting, film grain, depth of field, 8k",
    "Anime": "anime style illustration, vibrant colors, clean line art, studio quality",
    "3D Render": "3D render, octane render, soft studio lighting, high detail, smooth materials",
    "Digital Art": "digital painting, concept art, rich colors, detailed brush strokes, artstation",
    "Watercolor": "watercolor painting, soft washes, paper texture, delicate details",
    "Pixel Art": "pixel art, 16-bit, retro game style, crisp pixels",
}

MAX_RETRIES = 3
HISTORY_LIMIT = 12


# ---------------------------------------------------------------- helpers
class GenError(Exception):
    def __init__(self, message, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def to_png_bytes(raw: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise GenError("Server ne valid image nahi di.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def classify_http(status: int) -> GenError:
    if status == 429:
        return GenError("Limit/quota poora ho gaya (429).", retryable=True)
    if status in (500, 502, 503, 504):
        return GenError(f"Server abhi busy hai ({status}).", retryable=True)
    if status in (401, 403):
        return GenError("Key/token galat ya permission nahi hai.", retryable=False)
    if status == 402:
        return GenError("Is service ka free credit khatam ho gaya (402).", retryable=False)
    return GenError(f"Unexpected error (HTTP {status}).", retryable=True)


# ---------------------------------------------------------------- backends
def gen_pollinations(prompt, width, height, seed, model):
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {
        "width": width,
        "height": height,
        "seed": seed,
        "model": model,
        "nologo": "true",
    }
    try:
        r = requests.get(url, params=params, timeout=120)
    except requests.Timeout:
        raise GenError("Pollinations ne time pe jawab nahi diya (timeout).")
    except requests.RequestException:
        raise GenError("Internet / connection problem.")
    if r.status_code != 200:
        raise classify_http(r.status_code)
    if not r.headers.get("content-type", "").startswith("image"):
        raise GenError("Pollinations ne image ki jagah kuch aur bheja.")
    return to_png_bytes(r.content)


def gen_huggingface(prompt, width, height, seed):
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN, timeout=120)
    try:
        img = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=width,
            height=height,
            seed=seed,
        )
    except Exception as e:
        raw = str(e).replace("\n", " ")[:300]
        msg = raw.lower()
        if "429" in msg or "too many requests" in msg:
            raise GenError(f"Hugging Face rate limit (429). Asli error: {raw}", retryable=True)
        if "402" in msg or "payment required" in msg or "credits" in msg:
            raise GenError(f"Hugging Face credit/quota (402). Asli error: {raw}", retryable=False)
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            raise GenError(f"HF token galat ya permission nahi. Asli error: {raw}", retryable=False)
        raise GenError(f"Hugging Face error. Asli error: {raw}", retryable=True)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def enhance_prompt(text: str) -> str:
    """Hindi/Hinglish prompt ko detailed English prompt mein badalta hai. Fail ho toh original."""
    instruction = (
        "Rewrite the following image idea as one detailed English prompt for an AI image "
        "generator. Keep the meaning, add helpful visual detail (subject, setting, lighting, "
        "mood). Reply with ONLY the prompt, max 60 words. Idea: " + text
    )
    try:
        r = requests.get("https://text.pollinations.ai/" + quote(instruction), timeout=30)
        out = r.text.strip().strip('"')
        if r.status_code == 200 and 5 < len(out) < 700:
            return out
    except requests.RequestException:
        pass
    return text


def build_backends(prompt, width, height, seed):
    backends = []
    if HF_TOKEN:
        backends.append(("Hugging Face FLUX", lambda: gen_huggingface(prompt, width, height, seed)))
    backends.append(("Pollinations FLUX", lambda: gen_pollinations(prompt, width, height, seed, "flux")))
    backends.append(("Pollinations Turbo", lambda: gen_pollinations(prompt, width, height, seed, "turbo")))
    return backends


def generate_with_fallback(prompt, width, height, seed, status):
    errors = []
    for name, fn in build_backends(prompt, width, height, seed):
        for attempt in range(1, MAX_RETRIES + 1):
            status.update(label=f"Rendering your image...  (attempt {attempt}/{MAX_RETRIES})")
            try:
                return fn(), name
            except GenError as e:
                errors.append(f"{name}: {e}")
                if not e.retryable:
                    break
                if attempt < MAX_RETRIES:
                    time.sleep(2**attempt + random.random())
    raise GenError("\n".join(f"- {line}" for line in dict.fromkeys(errors)), retryable=False)


# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@500;600;700;800&display=swap');

:root {
    --bg: #07070d;
    --panel: rgba(255,255,255,0.035);
    --border: rgba(255,255,255,0.09);
    --text: #ecebf5;
    --muted: #8d8ca3;
    --accent1: #7c5cff;
    --accent2: #22d3ee;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(900px 500px at 12% -10%, rgba(124,92,255,0.22), transparent 60%),
        radial-gradient(800px 480px at 95% 5%, rgba(34,211,238,0.14), transparent 60%),
        var(--bg);
    color: var(--text);
}

/* hide default streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
.stAppDeployButton, [data-testid="stStatusWidget"] { display: none !important; visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 880px; padding-top: 2.2rem; padding-bottom: 3rem; }

/* hero */
.hero { text-align: center; padding: 1.2rem 0 1.8rem 0; }
.badge {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 600;
    color: #cfc8ff; border: 1px solid rgba(124,92,255,0.45);
    background: rgba(124,92,255,0.12);
}
.hero h1 {
    font-family: 'Sora', sans-serif; font-weight: 800; letter-spacing: -0.03em;
    font-size: clamp(2.3rem, 6vw, 3.8rem); line-height: 1.08; margin: 1rem 0 0.7rem 0; padding: 0;
    color: #fff;
}
.hero h1 span {
    background: linear-gradient(100deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--muted); font-size: 1.05rem; margin: 0 auto; max-width: 520px; }

/* labels */
label, [data-testid="stWidgetLabel"] p {
    color: var(--muted) !important; font-size: 0.82rem !important; font-weight: 500 !important;
    letter-spacing: 0.02em;
}

/* inputs */
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: var(--panel) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 14px !important;
    font-size: 1rem !important; transition: border-color .2s, box-shadow .2s;
}
.stTextArea textarea { padding: 1rem 1.1rem !important; min-height: 120px; }
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent1) !important; box-shadow: 0 0 0 3px rgba(124,92,255,0.22) !important;
}
[data-baseweb="select"] > div {
    background: var(--panel) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text) !important;
}

/* primary button */
.stButton > button, [data-testid="stDownloadButton"] > button {
    border-radius: 14px; font-weight: 600; letter-spacing: 0.01em; padding: 0.8rem 1.4rem;
    transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, var(--accent1), #5b8cff 55%, var(--accent2));
    color: #fff; border: none; box-shadow: 0 10px 30px -10px rgba(124,92,255,0.7);
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px); filter: brightness(1.08); }
.stButton > button[kind="primary"]:active { transform: translateY(0); }
[data-testid="stDownloadButton"] > button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--accent1); color: #fff; }

/* image result */
[data-testid="stImage"] img { border-radius: 20px; border: 1px solid var(--border);
    box-shadow: 0 30px 80px -30px rgba(0,0,0,0.85); }

/* status + expander + alerts */
[data-testid="stStatusWidget"], details { border-radius: 14px !important; }
[data-testid="stExpander"], [data-testid="stStatus"] {
    background: var(--panel); border: 1px solid var(--border) !important; border-radius: 14px;
}
[data-testid="stAlert"] { border-radius: 14px; }

/* sidebar */
[data-testid="stSidebar"] {
    background: rgba(10,10,18,0.92); border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.side-title {
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; color: #fff; margin-bottom: .2rem;
}
.side-sub { color: var(--muted); font-size: 0.78rem; margin-bottom: 1.1rem; }
.section-label {
    color: var(--muted); font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
    font-weight: 600; margin: 1.4rem 0 .6rem 0;
}
.pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 500; color: #bff5e0;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
}
.pill i { width: 7px; height: 7px; border-radius: 50%; background: #34d399; display: inline-block; }
.pill.off { color: #fde1a8; background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); }
.pill.off i { background: #fbbf24; }

.meta { color: var(--muted); font-size: 0.8rem; padding-top: .7rem; }

/* footer */
.credit {
    text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 3.5rem;
    padding-top: 1.4rem; border-top: 1px solid var(--border); letter-spacing: 0.03em;
}
.credit b {
    font-family: 'Sora', sans-serif; font-weight: 600;
    background: linear-gradient(100deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
</style>
"""

# ---------------------------------------------------------------- app
st.set_page_config(page_title="Siddhant AI Studio", page_icon="✦", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None

# sidebar
with st.sidebar:
    st.markdown('<div class="side-title">Studio Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Fine-tune how your image is created</div>', unsafe_allow_html=True)

    style = st.selectbox("Style", list(STYLES.keys()))
    aspect = st.selectbox("Format", list(ASPECTS.keys()))
    negative = st.text_input("Exclude", placeholder="blurry, text, watermark")
    auto_english = st.toggle(
        "Smart prompt enhancement",
        value=True,
        help="Understands Hindi and Hinglish, and enriches your prompt automatically.",
    )
    fixed_seed = st.number_input(
        "Seed  (0 = random)", min_value=0, max_value=2_147_483_647, value=0
    )

    st.markdown("", unsafe_allow_html=True)
    if HF_TOKEN:
        st.markdown('<div class="pill"><i></i>Premium engine active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill off"><i></i>Add HF_TOKEN in .env</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="section-label">Recent creations</div>', unsafe_allow_html=True)
        for item in st.session_state.history:
            st.image(item["png"], caption=item["prompt"][:48], use_container_width=True)

# hero
st.markdown(
    """
    <div class="hero">
        <div class="badge">AI Studio &nbsp;·&nbsp; Crafted by Siddhant Singh</div>
        <h1>Imagine it. <span>Create it.</span></h1>
        <p>Turn your words into stunning, high-quality visuals in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

user_prompt = st.text_area(
    "Describe your image",
    placeholder="A futuristic flying sports car above the Taj Mahal at golden hour...",
    height=130,
)

generate = st.button("Generate Image", type="primary", use_container_width=True)

if generate:
    text = user_prompt.strip()
    if not text:
        st.warning("Please describe the image you'd like to create.")
    else:
        width, height = ASPECTS[aspect]
        seed = int(fixed_seed) if fixed_seed else random.randint(1, 2_147_483_647)

        with st.status("Preparing...", expanded=False) as status:
            final_prompt = text
            if auto_english:
                status.update(label="Refining your prompt...")
                final_prompt = enhance_prompt(text)
            if STYLES[style]:
                final_prompt += ", " + STYLES[style]
            if negative.strip():
                final_prompt += f". Avoid: {negative.strip()}"

            try:
                png, backend = generate_with_fallback(final_prompt, width, height, seed, status)
                status.update(label="Your image is ready", state="complete")
                st.session_state.last = {
                    "png": png,
                    "prompt": text,
                    "final": final_prompt,
                    "backend": backend,
                    "seed": seed,
                    "format": aspect.split("  ")[0],
                    "style": style,
                }
                st.session_state.history.insert(0, {"png": png, "prompt": text})
                st.session_state.history = st.session_state.history[:HISTORY_LIMIT]
            except GenError as e:
                status.update(label="Could not create the image", state="error")
                st.session_state.last = None
                st.error("We couldn't create your image right now. Please try again in a minute.")
                with st.expander("Technical details"):
                    st.code(str(e), language=None)

last = st.session_state.last
if last:
    st.image(last["png"], use_container_width=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download PNG",
            data=last["png"],
            file_name=f"siddhant_ai_{last['seed']}.png",
            mime="image/png",
            use_container_width=True,
        )
    with c2:
        st.markdown(
            f'<div class="meta">{last["format"]} &nbsp;·&nbsp; {last["style"]} &nbsp;·&nbsp; Seed {last["seed"]}</div>',
            unsafe_allow_html=True,
        )
    with st.expander("View final prompt"):
        st.write(last["final"])

st.markdown(
    '<div class="credit">Designed &amp; built by <b>Siddhant Singh</b> &nbsp;·&nbsp; © 2026</div>',
    unsafe_allow_html=True,
)
