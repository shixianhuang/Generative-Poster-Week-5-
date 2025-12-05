# =====================================================
# Step 7 — Interactive UI (Streamlit Version, Colab-stable)
# Converts ipywidgets workflow to a Streamlit web app.
# Author: HUANG SHIXIAN
# Course: Arts and Advanced Big Data
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random, math, colorsys, io
import pandas as pd
from typing import List, Tuple, Optional

st.set_page_config(page_title="Generative Abstract Poster — Step 7", layout="wide")

# ------------------------------
# Helpers: numeric + color utils
# ------------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def hsv_to_rgb_tuple(h: float, s: float, v: float) -> Tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(clamp01(h), clamp01(s), clamp01(v))
    return (r, g, b)

# ------------------------------
# Palettes
# ------------------------------
def pastel_palette(k: int = 6) -> List[Tuple[float, float, float]]:
    """Soft, muted colors: low saturation, high value."""
    hues = np.linspace(0, 1, k, endpoint=False)
    np.random.shuffle(hues)
    return [hsv_to_rgb_tuple(h, np.random.uniform(0.20, 0.45), np.random.uniform(0.85, 0.98))
            for h in hues]

def vivid_palette(k: int = 6) -> List[Tuple[float, float, float]]:
    """Strong, high-contrast colors: high saturation, medium-high value."""
    hues = np.linspace(0, 1, k, endpoint=False)
    np.random.shuffle(hues)
    return [hsv_to_rgb_tuple(h, np.random.uniform(0.80, 1.00), np.random.uniform(0.70, 0.95))
            for h in hues]

def mono_palette(k: int = 6, base_hue: float = 0.58) -> List[Tuple[float, float, float]]:
    """Monochrome shades around a single hue (default: blue)."""
    return [hsv_to_rgb_tuple(base_hue,
                             np.random.uniform(0.25, 0.85),
                             np.random.uniform(0.55, 0.98))
            for _ in range(k)]

def random_palette(k: int = 6) -> List[Tuple[float, float, float]]:
    return [(random.random(), random.random(), random.random()) for _ in range(k)]

def csv_palette_from_df(df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """
    Accept columns named r,g,b (or R,G,B) in either 0-1 or 0-255 scale.
    Returns list of (r,g,b) in [0,1].
    """
    cols = {c.lower(): c for c in df.columns}
    needed = {"r","g","b"}
    if not needed.issubset(cols.keys()):
        raise ValueError("CSV must include columns: r,g,b (0-1 or 0-255).")
    r = df[cols["r"]].to_numpy()
    g = df[cols["g"]].to_numpy()
    b = df[cols["b"]].to_numpy()
    # Try to detect scale
    max_val = max(float(r.max()), float(g.max()), float(b.max()))
    if max_val > 1.001:  # likely 0-255 scale
        r, g, b = r/255.0, g/255.0, b/255.0
    palette = [(clamp01(float(r[i])), clamp01(float(g[i])), clamp01(float(b[i])))
               for i in range(len(df))]
    return palette

# ------------------------------
# Geometry: wobbly circle ("blob")
# ------------------------------
def blob(center=(0.5, 0.5), r=0.3, points=220, wobble=0.15):
    angles = np.linspace(0, 2*math.pi, points)
    radii  = r * (1 + wobble * (np.random.rand(points) - 0.5))
    x = center[0] + radii*np.cos(angles)
    y = center[1] + radii*np.sin(angles)
    return x, y

# ------------------------------
# Core drawing (Step 6 API compatible)
# ------------------------------
def draw_poster(
    n_layers: int = 8,
    wobble: float = 0.15,
    palette_mode: str = "pastel",
    seed: int = 0,
    figsize: Tuple[int, int] = (6, 8),
    title: str = "Generative Poster — Step 7",
    subtitle: Optional[str] = "Interactive UI (Streamlit)"
):
    """
    Compatible with the Step 6 signature in your notebook.
    Renders a single poster and returns a matplotlib Figure.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Select palette
    if palette_mode == "pastel":
        palette = pastel_palette(7)
    elif palette_mode == "vivid":
        palette = vivid_palette(7)
    elif palette_mode == "mono":
        palette = mono_palette(7, base_hue=0.58)
    elif palette_mode == "random":
        palette = random_palette(7)
    else:
        # When palette_mode == "csv", we will replace this outside after reading file
        palette = pastel_palette(7)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.axis("off")
    ax.set_facecolor((0.98, 0.98, 0.97))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        x, y = blob(center=(cx, cy), r=rr, wobble=wobble)
        color = random.choice(palette)
        alpha = random.uniform(0.28, 0.62)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))

    ax.text(0.05, 0.95, title, fontsize=18, weight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.05, 0.91, subtitle, fontsize=11, transform=ax.transAxes)

    plt.tight_layout()
    return fig

# ------------------------------
# Streamlit UI (replacing ipywidgets)
# ------------------------------
st.title("🎨 Generative Abstract Poster — Step 7 (Interactive UI)")

with st.sidebar:
    st.header("Controls")
    # Match your ipywidgets:
    layers = st.slider("Layers", min_value=3, max_value=20, value=8, step=1)
    wobble = st.slider("Wobble", min_value=0.01, max_value=0.60, value=0.15, step=0.01)
    palette_mode = st.selectbox("Palette", ["pastel", "vivid", "mono", "random", "csv"], index=0)
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)

    # Figure size (inches)
    width  = st.slider("Width (inches)", min_value=4, max_value=12, value=6, step=1)
    height = st.slider("Height (inches)", min_value=5, max_value=16, value=8, step=1)

    csv_help = None
    uploaded_csv = None
    if palette_mode == "csv":
        st.markdown("Upload CSV with columns **r,g,b** in either 0–1 or 0–255 range.")
        uploaded_csv = st.file_uploader("Palette CSV", type=["csv"])

col1, col2 = st.columns([2,1])

if st.button("Generate Poster", type="primary"):
    # Build palette override for CSV mode
    csv_palette = None
    csv_note = ""
    if palette_mode == "csv":
        if uploaded_csv is None:
            st.warning("CSV palette selected but no file uploaded. Falling back to pastel.")
        else:
            try:
                df = pd.read_csv(uploaded_csv)
                csv_palette = csv_palette_from_df(df)
                csv_note = f" (CSV palette: {len(csv_palette)} colors)"
            except Exception as e:
                st.error(f"Failed to read CSV: {e}. Falling back to pastel.")

    # If using csv_palette, temporarily monkey-patch palette choice by wrapping draw_poster
    if csv_palette is not None:
        def draw_with_csv(**kwargs):
            # Call draw_poster but force palette afterward by redrawing fills
            # Simpler approach: temporarily set palette_mode to 'random' but override choices
            random.seed(kwargs.get("seed", 0))
            np.random.seed(kwargs.get("seed", 0))

            fig = plt.figure(figsize=kwargs.get("figsize", (6,8)))
            ax = plt.gca()
            ax.axis("off")
            ax.set_facecolor((0.98,0.98,0.97))
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            for _ in range(kwargs.get("n_layers", 8)):
                cx, cy = random.random(), random.random()
                rr = random.uniform(0.15, 0.45)
                x, y = blob(center=(cx, cy), r=rr, wobble=kwargs.get("wobble", 0.15))
                color = random.choice(csv_palette)
                alpha = random.uniform(0.28, 0.62)
                ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))

            ax.text(0.05, 0.95, kwargs.get("title", "Generative Poster — Step 7"),
                    fontsize=18, weight="bold", transform=ax.transAxes)
            sub = kwargs.get("subtitle", "Interactive UI (Streamlit)") + csv_note
            ax.text(0.05, 0.91, sub, fontsize=11, transform=ax.transAxes)
            plt.tight_layout()
            return fig

        fig = draw_with_csv(
            n_layers=layers,
            wobble=wobble,
            seed=seed,
            figsize=(width, height),
        )
    else:
        fig = draw_poster(
            n_layers=layers,
            wobble=wobble,
            palette_mode=palette_mode,
            seed=seed,
            figsize=(width, height),
            title="Generative Poster — Step 7",
            subtitle=f"Interactive UI (Streamlit) — palette={palette_mode}{csv_note}",
        )

    with col1:
        st.pyplot(fig)

        # Download PNG
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
        st.download_button(
            "Download PNG",
            data=buf.getvalue(),
            file_name=f"poster_step7_{palette_mode}.png",
            mime="image/png"
        )

    with col2:
        st.markdown("### Notes")
        st.write("- This Streamlit app mirrors your Step 7 ipywidgets controls.")
        st.write("- **Seed** ensures reproducibility (same inputs → same output).")
        st.write("- **CSV palette**: provide `r,g,b` columns (0–1 or 0–255).")
        st.write("- Use *Download PNG* to export at 220 DPI.")
