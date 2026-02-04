import torch
from torch.utils.data import Dataset
from PIL import Image
import timm
import pandas as pd
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, config, transforms, ds_type="train", max_samples=None):
        # Загружаем CSV
        df = pd.read_csv(config.DATA_CSV)  # dish.csv
        ingr_df = pd.read_csv(config.INGR_CSV)  # ingredients.csv
        if max_samples:
            df = df[:max_samples]  # берем только первые max_samples

        # Словарь id -> название ингредиента
        self.ingr_map = dict(zip(
            ingr_df["id"].astype(str).str.zfill(10),
            ingr_df["ingr"]
        ))

        # Фильтруем по split
        self.df = df[df["split"] == ds_type].reset_index(drop=True)
        self.transforms = transforms
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # Генерируем текст из ингредиентов
        self.df["text"] = self.df["ingredients"].apply(self.ingredients_to_text)
        # Пути к изображениям
        self.df["image_path"] = self.df["dish_id"].apply(lambda x: f"{x}/rgb.png")
        # Label
        self.df["label"] = self.df["total_calories"]

    def ingredients_to_text(self, ingr_string):
        ids = ingr_string.split(";")
        names = []
        for ingr_id in ids:
            num_id = ingr_id.replace("ingr_", "")
            if num_id in self.ingr_map:
                names.append(self.ingr_map[num_id])
        return ", ".join(names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        text = row["text"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        img_path = f"{self.transforms.config.IMAGE_ROOT}/{row['image_path']}" \
            if hasattr(self.transforms, 'config') else f"data/images/{row['image_path']}"
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image=np.array(image))["image"]

        return {"text": text, "image": image, "label": label}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    tokenized_input = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    return {
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "label": labels
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.RandomCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0)
        ])
    else:
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.0)
        ])
    return transforms
