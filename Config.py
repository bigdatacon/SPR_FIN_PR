class Config:
    # ===== Модели =====
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # ===== Обучение =====
    BATCH_SIZE = 16
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    HEAD_LR = 5e-4
    EPOCHS = 1
    DROPOUT = 0.1
    EMB_DIM = 256

    # ===== Данные =====
    DATA_CSV = "data/dish.csv"
    INGR_CSV = "data/ingredients.csv"
    IMAGE_ROOT = "data/images"

    # ===== Сохранение =====
    SAVE_PATH = "best_model.pth"

    # ===== Target =====
    MAE_THRESHOLD = 50.0
