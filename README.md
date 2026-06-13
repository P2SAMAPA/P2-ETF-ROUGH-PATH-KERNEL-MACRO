# Rough Path Signature Kernel with Macro Conditioning

Applies rough path theory (Lyons 1998) to ETF returns. Computes truncated path signatures (depth 4) of rolling windows, then uses a product Gaussian kernel on signature + macro variables (VIX, DXY, yields) in a kernel ridge regression to predict next‑day returns. The per‑ETF score is the predicted return.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Path signature (Chen 1958) up to depth 4
- Product Gaussian kernel: K = K_sig * K_macro
- Kernel ridge regression (precomputed kernel)
- Score = predicted next‑day return
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-rough-path-kernel-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to O(n²) kernel matrix)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High predicted return → ETF is expected to rise.
- Low or negative → expected to fall.

## Requirements

See `requirements.txt`.
