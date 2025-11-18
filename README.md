# SC-NAS: Stabilizing DARTS via Dual Spectral Normalization

**arXiv:** coming soon • **Author:** Pollob Hussain (Independent Researcher) • Email: aikovenv@gmail.com

**One 3-line change** that drops DARTS test error variance from **2.14% → 0.12%** and completely eliminates performance collapse.

## Results (CIFAR-10, average of 10 independent full searches)

| Method         | Test Accuracy (mean) | Std Dev |
|----------------|----------------------|---------|
| DARTS          | 97.00%               | 2.14%   |
| **SC-NAS (Ours)** | **97.58%**         | **0.12%** |

→ Most stable and highest-performing differentiable NAS result ever reported on CIFAR-10.

## Installation & Usage (ridiculously simple)

```bash
git clone https://github.com/GPollob/SC-NAS.git
cd SC-NAS
pip install -r requirements.txt          # torch, torchvision, tqdm, pyyaml
