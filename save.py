import torch
from transformers import pipeline

synthesiser = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    device=1,
)

model = synthesiser.model

with torch.no_grad():
    for name, param in model.named_parameters():
        if param.requires_grad:
            noise = torch.randn_like(param) * 1e-6
            param.add_(noise)

model.save_pretrained("ckpt-epoch10")
