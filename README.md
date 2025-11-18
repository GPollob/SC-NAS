# SC-NAS: Stabilizing DARTS via Dual Spectral Normalization

**arXiv:** (coming soon)  
**Author:** Pollob Hussain (Independent Researcher)  
**Email:** aikovenv@gmail.com

A 3-line fix that reduces DARTS variance from 2.14% → **0.12%** and eliminates collapse.

## Results (10 independent runs)
| Method      | Mean (%) | Std (%) |
|-------------|----------|---------|
| DARTS       | 97.00    | 2.14    |
| SC-NAS (Ours) | **97.58** | **0.12** |

## Usage
```bash
git clone https://github.com/GPollob/SC-NAS.git
cd SC-NAS
pip install -r requirements.txt
python search.py --epochs 50
torch>=2.0
torchvision
