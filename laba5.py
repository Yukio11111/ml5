import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

!pip install -q transformers pytorch_metric_learning kagglehub
!pip install -q torchao --upgrade

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from transformers import AutoModel`
import kagglehub

from google.colab import drive

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 280
EMB_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 15
LR = 2e-5
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# DATASET DOWNLOAD
# ============================================================
print("\nDownloading dataset...")
os.environ["KAGGLE_USERNAME"] = ""
os.environ["KAGGLE_KEY"] = ""

dataset_path = kagglehub.competition_download("dl-lab-5-metric-learning")
BASE_DIR = Path(dataset_path)

TRAIN_ROOT = BASE_DIR / "train" / "train"
TEST_ROOT = BASE_DIR / "test_kaggle" / "test_kaggle"
INPUT_SUBMISSION_PATH = BASE_DIR / "submission.csv"

drive.mount("/content/drive")
SAVE_DIR = Path("/content/drive/MyDrive/laba5")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = SAVE_DIR / "best_dinov2_msloss.pth"
OUTPUT_SUBMISSION_PATH = SAVE_DIR / "submission_dinov2_msloss.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ============================================================
# TRANSFORMS
# ============================================================
class SquarePad:
    def __init__(self, fill=128): self.fill = fill
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = (max_wh - w) // 2, (max_wh - h) // 2
        return TF.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), fill=self.fill)

train_transform = T.Compose([
    SquarePad(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = T.Compose([
    SquarePad(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# DATASET CLASS
# ============================================================
class ProductDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

print("\nPreparing data...")
all_classes = sorted([p.name for p in TRAIN_ROOT.iterdir() if p.is_dir()])
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

train_paths, train_labels = [], []
val_paths, val_labels = [], []

for cls_name in all_classes:
    cls_idx = class_to_idx[cls_name]
    imgs = [p for p in (TRAIN_ROOT / cls_name).glob("*.*") if p.suffix.lower() in IMAGE_EXTS]
    random.shuffle(imgs)

    if len(imgs) < 3: continue

    val_count = max(2, int(len(imgs) * 0.2))
    val_imgs = imgs[:val_count]
    train_imgs = imgs[val_count:]

    if len(train_imgs) < 2: continue

    train_paths.extend(train_imgs); train_labels.extend([cls_idx] * len(train_imgs))
    val_paths.extend(val_imgs); val_labels.extend([cls_idx] * len(val_imgs))

sampler = MPerClassSampler(labels=train_labels, m=4, batch_size=BATCH_SIZE, length_before_new_iter=len(train_paths))

train_loader = DataLoader(ProductDataset(train_paths, train_labels, train_transform), batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, drop_last=True, pin_memory=True)
val_loader = DataLoader(ProductDataset(val_paths, val_labels, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# MODEL: DINOv2
# ============================================================
class DinoMetricModel(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.fc = nn.Linear(768, emb_size)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        embeddings = self.fc(cls_token)
        return F.normalize(embeddings, p=2, dim=1)

print(f"\nInitializing {MODEL_NAME}...")
model = DinoMetricModel(EMB_SIZE).to(DEVICE)

# ============================================================
# LOSS: MultiSimilarity
# ============================================================
miner = miners.MultiSimilarityMiner()
loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# ============================================================
# VALIDATION FUNCTION
# ============================================================
def calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals=(0.0001,)):
    thresholds = np.quantile(neg_dist, fmr_vals)
    fnmr = np.array([(pos_dist > t).mean() for t in thresholds], dtype=np.float32)
    return float(fnmr[0])

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_embeddings, all_labels = [], []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device)
        with torch.cuda.amp.autocast():
            embs = model(images)
        all_embeddings.append(embs.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    sim = all_embeddings @ all_embeddings.T
    dist = (1 - sim).numpy()

    labels_np = all_labels.numpy()
    same = labels_np[:, None] == labels_np[None, :]
    eye = np.eye(len(labels_np), dtype=bool)
    valid = ~eye

    pos_dist = dist[same & valid]
    neg_dist = dist[~same & valid]

    fnmr = calc_fnmr_at_fmr(pos_dist, neg_dist)

    pos_count = len(pos_dist) // 2
    neg_count = len(neg_dist) // 2

    return fnmr, pos_count, neg_count

# ============================================================
# TRAINING
# ============================================================
print("\nStarting Training...")
best_fnmr = 1.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_pairs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    scheduler.step()

    val_fnmr, pos_count, neg_count = validate(model, val_loader, DEVICE)

    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f} | Val FNMR@1e-4={val_fnmr:.4f}")
    print(f"   [Stats] Pos Pairs: {pos_count:,} | Neg Pairs: {neg_count:,}")

    if val_fnmr < best_fnmr:
        best_fnmr = val_fnmr
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Best model saved! FNMR={best_fnmr:.4f}")

# ============================================================
# INFERENCE
# ============================================================
print(f"\nLoading best model (FNMR={best_fnmr:.4f}) for inference...")
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

submission = pd.read_csv(INPUT_SUBMISSION_PATH)[["id", "file_1", "file_2"]].copy()

filename_to_path = {p.name: p for p in TEST_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
needed_files = set(submission["file_1"].astype(str)) | set(submission["file_2"].astype(str))
needed_paths = [filename_to_path[x] for x in sorted(needed_files)]

print(f"Extracting test embeddings...")
test_embeddings = {}

with torch.no_grad():
    for start in tqdm(range(0, len(needed_paths), BATCH_SIZE)):
        batch_paths = needed_paths[start : start + BATCH_SIZE]
        images_tensor = torch.stack([eval_transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)

        with torch.cuda.amp.autocast():
            embs = model(images_tensor).cpu().numpy()

        for i, path in enumerate(batch_paths):
            test_embeddings[path.name] = embs[i].astype(np.float32)

print("Calculating similarities...")
similarities = []
for row in tqdm(submission.itertuples(index=False), total=len(submission)):
    sim = float(np.dot(test_embeddings[row.file_1], test_embeddings[row.file_2]))
    similarities.append(sim)

submission["similarity"] = (np.array(similarities) + 1.0) / 2.0
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print(f"Submission saved to: {OUTPUT_SUBMISSION_PATH}")