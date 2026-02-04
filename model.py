import re
import torch
import torch.nn as nn
import timm
from transformers import AutoModel


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Текстовая модель
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        # Визуальная модель
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # Проекция в общее пространство
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.EMB_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.EMB_DIM)

        # Регрессор
        self.regressor = nn.Sequential(
            nn.Linear(config.EMB_DIM*2, config.EMB_DIM),
            nn.LayerNorm(config.EMB_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.EMB_DIM, 1)
        )

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        # Конкатенация эмбеддингов вместо умножения
        fused_emb = torch.cat([text_emb, image_emb], dim=1)
        out = self.regressor(fused_emb)
        return out.squeeze(1)