# SC-NAS: Stabilizing DARTS via Dual Spectral Normalization

Official code for the arXiv paper: [arXiv link once submitted]  
**Authors**: Pollob Hussain (Independent Researcher)

## Overview
SC-NAS is a simple, 3-line modification to DARTS that eliminates performance collapse and reduces std dev from 2.14% to 0.12% on CIFAR-10. It uses dual spectral normalization for Lipschitz continuity in bilevel optimization.

**Key Results**:
- CIFAR-10: 97.58% ± 0.12% (10 runs)
- CIFAR-100: 83.21%
- ImageNet Top-1: 75.3%

## Quick Start
1. Clone: `git clone https://github.com/GPollob/SC-NAS.git`
2. Install: `pip install -r requirements.txt` (PyTorch 2.0+, torchsn)
3. Run search: `python search.py --epochs 50`
4. Reproduce Table 1: `python evaluate_stability.py` (outputs mean/std across 10 seeds)

## Method (3 Lines)
- Operation-level: `W = W / spectral_norm(W)` in conv layers.
- Arch-level: `alpha = alpha / norm(alpha, p=2)` before softmax.

## Files
- `darts.py`: Modified DARTS supernet with SN.
- `search.py`: Full search script.
- `evaluate.py`: Retrain discovered arch.

License: MIT (or CC BY 4.0). Questions? Email: aikovenv@gmail.com
