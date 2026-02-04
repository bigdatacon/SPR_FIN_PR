import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from functools import partial

from model import MultimodalCalorieModel, set_requires_grad
# import MultimodalCalorieModel, set_requires_grad
from dataset import MultimodalDataset, collate_fn
# get_transforms
from Config import Config


def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets))


def validate(model, loader, device):
    model.eval()
    maes = []

    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["image"].to(device)
            )
            labels = batch["label"].to(device)
            maes.append(mae(preds, labels).item())

    return float(np.mean(maes))


def train():
    cfg = Config()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)

    model = MultimodalCalorieModel(cfg).to(DEVICE)

    # 🔥 Freeze / unfreeze
    set_requires_grad(model.text_model, cfg.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, cfg.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {"params": model.text_model.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_model.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.regressor.parameters(), "lr": cfg.HEAD_LR},
    ])

    criterion = nn.L1Loss()

    train_tfms = get_transforms(cfg, "train")
    val_tfms = get_transforms(cfg, "val")

    train_ds = MultimodalDataset(cfg, train_tfms)
    val_ds = MultimodalDataset(cfg, val_tfms, ds_type="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    best_mae = float("inf")

    for epoch in range(cfg.EPOCHS):
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}", flush=True)
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            preds = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["image"].to(DEVICE)
            )
            labels = batch["label"].to(DEVICE)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_mae = validate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | "
            f"Train MAE: {train_loss/len(train_loader):.2f} | "
            f"Val MAE: {val_mae:.2f}",
            flush=True
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"✅ Best model saved (MAE={best_mae:.2f})")

        if val_mae <= cfg.MAE_THRESHOLD:
            print("🎯 TARGET ACHIEVED (MAE ≤ 50)")
            break
