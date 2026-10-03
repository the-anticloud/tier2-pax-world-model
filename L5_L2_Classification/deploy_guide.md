# Deploy Guide — PAX_WORLD_MODEL
**Stack:** Python 3.11, numpy, PAX 27B, PyTorch 2.10+, AIOSS_FORMAT | Air-gap capable

## Prerequisites
Anticloud core stack installed. PAX 27B weights (pax-27b-q4.gguf). AIOSS_FORMAT.

## Install
```bash
pip install anticloud-pax-world-model
```

## AIOSS Integration
```bash
aioss init --module PAX_WORLD_MODEL --output ./pax_world_model.aioss
```

## Air-Gap
```bash
pip download -r requirements.txt -d ./wheels/
pip install --no-index --find-links ./wheels/ -r requirements.txt
```

## PAX Harness Wiring
```python
from anticloud_pax import PAXHarness
harness = PAXHarness(model_path="./pax-27b-q4.gguf", module="PAX_WORLD_MODEL",
                     aioss_chain="./pax_world_model.aioss",
                     classification="L5_NARROW_L2_GENERAL")
```

## Verification
```bash
aioss verify --chain ./pax_world_model.aioss --verbose
python -m pax_world_model.tests.smoke
```
