# Narrative Frame Prediction via Multimodal Deep Learning

## Quick Links
- **[Experiments Notebook](experiments.ipynb)** – Full training and evaluation workflow
- **[Results Folder](results/)** – Generated metrics, figures, and XAI outputs
- **[Loss Curves](results/loss_curves.png)** – Training progress over 25 epochs
- **[Prediction Example](results/prediction_example.png)** – Qualitative story continuation

---

## Innovation Summary

This project tackles **narrative frame prediction**: given K=4 sequential image-caption pairs
from the [StoryReasoning dataset](https://arxiv.org/abs/2505.10292), predict the next story frame.

| # | Component | Baseline | Innovation | Justification |
|---|-----------|----------|------------|---------------|
| 1 | **Fusion** | Concatenation | **Additive Attention Fusion** | Bahdanau-style compatibility is more expressive than straight concatenation for cross-modal alignment |
| 2 | **Temporal Encoder** | Single LSTM | **Stacked GRU** | GRUs have fewer parameters than LSTMs and train faster on medium-sized narrative datasets |
| 3 | **Attention** | Uniform weights | **Narrative Attention with Recency Bias** | Learnable decay parameter encodes temporal locality as a soft prior |
| 4 | **Explainability** | None | **Integrated Gradients + Grad-CAM++ + Attention Visualisation** | Multi-level interpretability across temporal, spatial, and semantic inputs |

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Validation Loss | 8.3935 |
| BLEU-4 | 0.60 |
| Frame L1 Loss | 1.282341 |
| Training Epochs | 25 |

---

## Architecture

```
frames (N, K, C, H, W)   ──►  ImageFeatureExtractor (ResNet-34)  ──►  (N, K, 256)
                                                                              │
captions (N, K, T)        ──►  CaptionFeatureExtractor (GRU)      ──►  (N, K, 256)
                                                                              │
                               ╔══════════════════════════════╗
                               ║  AdditiveAttentionFusion      ║  [Innovation 1]
                               ║  f(v,t) = w^T tanh(Wv + Ut)  ║
                               ╚══════════════════════════════╝
                                             │
                               ╔══════════════════════════════╗
                               ║  TemporalContextEncoder       ║  [Innovation 2]
                               ║  Stacked GRU (2 layers)       ║
                               ╚══════════════════════════════╝
                                             │
                               ╔══════════════════════════════╗
                               ║  NarrativeAttention           ║  [Innovation 3]
                               ║  Recency Bias: -λ|i-j|        ║
                               ╚══════════════════════════════╝
                                             │
                      ┌──────────────────────┴──────────────────┐
                      ▼                                           ▼
             FrameDecoder                                 CaptionDecoder
          (Upsample + Conv)                           (GRU + Scheduled Sampling)
                      │                                           │
           Predicted Frame                             Predicted Caption
            (N, 3, 256, 256)                               (N, T, vocab)
```

---

## Key Differences from Standard Approaches

| Design Choice | Standard | This Work | Reason |
|---|---|---|---|
| Image backbone | ResNet-50 | **ResNet-34** | Lighter model, lower overfitting risk |
| Fusion | Concatenation | **Additive Attention** | Better cross-modal alignment |
| Sequence model | LSTM | **Stacked GRU** | Faster training, fewer parameters |
| Temporal attention | Sinusoidal PE | **Learnable recency bias** | Data-driven temporal prior |
| Image loss | MSE | **L1 Loss** | Less sensitive to outlier pixels |
| LR schedule | Cosine | **Step decay** | Predictable decay at fixed intervals |
| Context frames | 3 | **4** | More narrative context |

---

## Explainability

Three techniques implemented in `src/xai.py`:

1. **Integrated Gradients** — attributes predictions to input frames by integrating gradients from a baseline.
   Saved to `results/xai/integrated_gradients.png`.

2. **Grad-CAM++** — spatial localisation on the ResNet-34 feature maps.
   Saved to `results/xai/gradcam_pp.png`.

3. **Narrative Attention Scores** — visualises the recency-biased attention weights per context frame.
   Saved to `results/xai/attention_scores.png` and `results/xai/attention_matrix.png`.

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook experiments.ipynb
```

To load the best saved checkpoint:

```python
import torch
ckpt = torch.load('saved_models/best.pt', map_location=device)
net.load_state_dict(ckpt['net_state'])
net.eval()
```

> Training completed in 25 epochs using the notebook workflow.

---

## Configuration (see `settings.yaml`)

| Parameter | Value |
|---|---|
| Context frames K | 4 |
| Image resolution | 256×256 |
| Backbone | ResNet-34 |
| Embedding size | 256 |
| Batch size | 12 |
| Learning rate | 3e-4 |
| LR schedule | StepLR (γ=0.5, step=8) |
| Epochs | 25 |
| Optimiser | Adam |

---

## Repository Structure

```
project_client2/
├── README.md
├── experiments.ipynb
├── settings.yaml
├── requirements.txt
├── src/
│   ├── architecture.py
│   ├── runner.py
│   ├── xai.py
│   └── helpers.py
├── saved_models/
│   ├── best.pt
│   └── final.pt
└── results/
    ├── ablation.csv
    ├── loss_curves.png
    ├── metrics.csv
    ├── prediction_example.png
    ├── sample_sequence.png
    └── xai/
        ├── attention_matrix.png
        ├── attention_scores.png
        ├── gradcam_pp.png
        └── integrated_gradients.png
```

---

## Dataset Reference

Oliveira, D. A. P., & Matos, D. M. (2025). *StoryReasoning Dataset: Using Chain-of-Thought
for Scene Understanding and Grounded Story Generation*. arXiv:2505.10292.
https://arxiv.org/abs/2505.10292
