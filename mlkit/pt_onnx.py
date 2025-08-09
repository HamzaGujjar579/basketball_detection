import torch
import torch.nn as nn
from ultralytics import YOLO

# Step 1: Load YOLOv8 model
model = YOLO('best.pt').model
model.eval()

# Step 2: Wrap to accept NHWC input
class NHWCWrapper(nn.Module):
    def __init__(self, original_model):
        super(NHWCWrapper, self).__init__()
        self.model = original_model

    def forward(self, x):
        # Convert NHWC [B, H, W, C] to NCHW [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return self.model(x)

wrapped_model = NHWCWrapper(model).eval()

# Step 3: Export to ONNX using dummy NHWC input
dummy_input_nhwc = torch.randn(1, 640, 640, 3)

# Run a forward pass to make sure it works
output = wrapped_model(dummy_input_nhwc)

# Export the model to ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input_nhwc,
    "best_nhwc.onnx",
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=13
)

print("✅ Exported NHWC-compatible model as best_nhwc.onnx")
