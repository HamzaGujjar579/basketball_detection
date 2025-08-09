import torch
import torch.nn as nn
from ultralytics import YOLO

# Step 1: Load YOLOv8 model properly
model = YOLO('best.pt').model
model.eval()

# Step 2: Wrap model to accept NHWC input
class NHWCWrapper(nn.Module):
    def __init__(self, original_model):
        super(NHWCWrapper, self).__init__()
        self.model = original_model

    def forward(self, x):
        # Convert NHWC [B, H, W, C] → NCHW [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return self.model(x)

# Wrap and set model to eval mode
wrapped_model = NHWCWrapper(model).eval()

# Step 3: Print input shape using a dummy NHWC tensor
dummy_input_nhwc = torch.randn(1, 640, 640, 3)  # NHWC input
output = wrapped_model(dummy_input_nhwc)
print(f"✅ Input shape: {dummy_input_nhwc.shape}")
# If output is a tuple (as in YOLOv8), access the first element
if isinstance(output, tuple):
    print(f"✅ Output shape: {output[0].shape}")
else:
    print(f"✅ Output shape: {output.shape}")
# Step 4: Save the NHWC-compatible model as .pt again
torch.save(wrapped_model, 'best_nhwc.pt')
print("✅ Saved wrapped model as best_nhwc.pt with NHWC support.")
