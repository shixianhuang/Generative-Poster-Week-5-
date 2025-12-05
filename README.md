# 🎨 Generative Abstract Poster — Step 7 (Interactive UI, Streamlit)

This project converts the **ipywidgets** UI from Step 7 into a **Streamlit** web app.  
It preserves the same controls (layers, wobble, palette, seed, size) and adds:
- CSV palette upload (columns: `r,g,b`, values in `0–1` or `0–255`)
- One-click **Download PNG** (220 DPI)

---

## 🌈 Features
- Adjustable **Layers / Wobble / Seed**
- Palettes: **pastel / vivid / mono / random / csv**
- **Figure size** controls (inches)
- **PNG download** button for exporting posters
- Fully reproducible with the same seed

---

## 🧠 How it maps from ipywidgets → Streamlit
- `IntSlider / FloatSlider / Dropdown` → `st.slider / st.selectbox / st.number_input`
- `interactive_output(draw_poster, {...})` → direct call to `draw_poster(...)` on button click
- Size changes → set `figsize=(width, height)` before drawing
- CSV palette → `st.file_uploader` + parsing with **pandas**

---

## ⚙️ Installation & Run
```bash
git clone https://github.com/yourusername/generative-poster-step7.git
cd generative-poster-step7
pip install -r requirements.txt
streamlit run app.py
