import torch
from transformers import pipeline

# Загружаем пайплайн
synthesiser = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    device=1,
)

model = synthesiser.model  # <-- достаем модель из пайплайна

# Проходим по параметрам модели и зашумляем их
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.requires_grad:
            noise = torch.randn_like(param) * 1e-6
            param.add_(noise)

model.save_pretrained("ckpt-epoch10")
