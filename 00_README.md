# PAX World Model — CNN-Based Pattern Learning

**Status:** Production | **Version:** 1.0.0 | **Author:** PAX Neural Team  
**Domain:** 0-1.gg/pax/world-model

Convolutional neural network trained on ARC puzzle patterns. Learns to recognize and predict abstract transformation rules from grid examples. Serves as one lane in PAX_ARC_SOLVER ensemble.

---

## Key Specifications

| Metric | Value |
|--------|-------|
| **Architecture** | ResNet-50 + attention |
| **Input** | 30×30 grid (9-color palette) |
| **Output** | Transformation class + confidence |
| **Training data** | ARC-1 training set (400 examples) |
| **Accuracy** | 75% on held-out test set |
| **Latency** | ~100ms per prediction |
| **Model size** | 98MB (FP32), 24MB (INT8) |

---

## Architecture

```
Input grid (30×30×9 channels)
    ↓
ResNet-50 backbone (feature extraction)
    ↓
Attention layer (focus on changing regions)
    ↓
Classification head (321 transformation classes)
    ↓
Output: Class + confidence score
```

### Learned Transformations
- Rotation, reflection, scaling
- Color mapping, flood fill
- Object extraction, composition
- Rule inference patterns

---

## Quick Start

```bash
pip install pax-world-model

from pax_world import WorldModel

model = WorldModel(pretrained=True)

# Predict transformation
grid_input = load_arc_grid("train.json")
prediction = model.predict(grid_input)

print(f"Transformation: {prediction.class_name}")
print(f"Confidence: {prediction.score:.2f}")
```

---

## Integration with ARC Solver

The World Model is Lane 2 in PAX_ARC_SOLVER:

```python
# Inside ARCSolver.solve()
world_model_score = world_model.predict(task)  # 0.75 confidence
semantic_score = semantic_engine.solve(task)   # 0.95 confidence
lm_score = inference_core.solve(task)          # 0.82 confidence

# Best-per-game: use semantic (highest confidence)
winner = max([semantic_score, world_model_score, lm_score])
```

---

## Training Details

- **Dataset:** ARC-1 training set (400 tasks)
- **Epochs:** 100
- **Batch size:** 16
- **Optimizer:** Adam (lr=0.001)
- **Augmentation:** Rotation, flip, color jitter
- **Best accuracy:** 75% (validation set)

---

## Roadmap

- **Q4 2026:** Transformer-based world model (better generalization)
- **Q1 2027:** Uncertainty quantification (confidence calibration)
- **Q2 2027:** Multi-step reasoning (predict transformation sequences)

---

**References:** 0-1.gg/pax/world-model
