# EdgeTAM CoreML Export

Export EdgeTAM to CoreML format for iOS/macOS deployment.

## Quick Export

```bash
python coreml/export_to_coreml.py \
  --sam2_cfg sam2/configs/edgetam.yaml \
  --sam2_checkpoint checkpoints/edgetam.pt
```

This creates three CoreML models in `./coreml_models/`:
- `edgetam_image_encoder.mlpackage` (9.6MB)
- `edgetam_prompt_encoder.mlpackage` (2.0MB)
- `edgetam_mask_decoder.mlpackage` (9.8MB)

## Usage Example

```python
import coremltools as ct
from PIL import Image

# Load models
image_encoder = ct.models.MLModel("coreml_models/edgetam_image_encoder.mlpackage")
prompt_encoder = ct.models.MLModel("coreml_models/edgetam_prompt_encoder.mlpackage")
mask_decoder = ct.models.MLModel("coreml_models/edgetam_mask_decoder.mlpackage")

# Segment with point prompt
image = Image.open("image.jpg").resize((1024, 1024))
encoder_out = image_encoder.predict({"image": image})

# Add your point and generate mask
# See coreml_models/inference_example.py for complete example
```

## Performance Benchmark

Run benchmark with: `python coreml/benchmark_coreml.py`

Note: This is a limited test on synthetic data. Real-world performance may vary.

### Results

| Metric | PyTorch | CoreML | Difference |
|--------|---------|--------|------------|
| Speed | 40.1ms | 39.2ms | -0.9ms |
| Quality | IoU 0.9897 | IoU 0.9893 | -0.0004 |
| Size | 54MB | 21.4MB | -32.6MB |

## Requirements

- PyTorch
- coremltools
- EdgeTAM checkpoint

The CoreML export maintains identical segmentation quality while being faster and 60% smaller for mobile deployment.