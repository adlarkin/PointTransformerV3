import torch
import torch.nn as nn
import numpy as np
from model import PointTransformerV3
from stability_predictor import StabilityPredictor
from torch.utils.data import Dataset, DataLoader

_PC_SIZE = 256
_EPOCHS = 50
_SEED: int | None = 42

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pc_data(file: str, pc_size: int, seed: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _resample(points, size, rng: np.random.Generator):
        indices = rng.choice(len(points), size, replace=len(points) < size)
        return points[indices]

    rng = np.random.default_rng(seed=seed)

    data = np.load(file, allow_pickle=True)
    semantic_names_to_ids = data["semantic_names_to_ids"].item()
    pointclouds = data["pointclouds"]
    semantic_ids = data["semantic_ids"]
    stability_labels = data["stability_labels"]
    env_list, obj_list = [], []

    for pc, ids in zip(pointclouds, semantic_ids):
        env_points = pc[ids == semantic_names_to_ids["environment"]]
        obj_points = pc[ids == semantic_names_to_ids["object"]]

        env_list.append(torch.from_numpy(_resample(env_points, pc_size, rng)).float())
        obj_list.append(torch.from_numpy(_resample(obj_points, pc_size, rng)).float())

    return torch.stack(env_list), torch.stack(obj_list), torch.from_numpy(stability_labels).float()

class PointCloudPairDataset(Dataset):
    def __init__(self, file_path: str, pc_size: int, seed: int | None):
        self.env_pcs, self.obj_pcs, self.pc_pair_labels = load_pc_data("isaac_pc_data.npz", pc_size, seed)
        assert self.env_pcs.shape == self.obj_pcs.shape
        assert len(self.env_pcs) == len(self.pc_pair_labels)

    def __len__(self):
        return len(self.env_pcs)

    def __getitem__(self, idx):
        return self.env_pcs[idx], self.obj_pcs[idx], self.pc_pair_labels[idx]


# NOTE: setting enable_flash=False because my GPU does not support it.
# Running with enable_flash=True gives the following error:
#   "FlashAttention only supports Ampere GPUs or newer."
# TODO: go back to commented out model initialization below once GPU is upgraded?
#
#backbone = PointTransformerV3(in_channels=3, cls_mode=True).to(_DEVICE)
backbone = PointTransformerV3(
    in_channels=3,
    cls_mode=True,
    enable_flash=False,      # Disables the requirement for Ampere GPUs
    upcast_attention=True,   # Improves stability when flash is disabled
    upcast_softmax=True      # Improves stability when flash is disabled
).to(_DEVICE)

# the feature_dim extraction below assumes the backbone is in cls_mode.
# feature_dim is the dimension of the backbone output (pointcloud embedding)
assert backbone.cls_mode
last_stage = backbone.enc[-1]
last_block = last_stage[-1]
feature_dim = last_block.channels
classification_head = StabilityPredictor(feature_dim=feature_dim).to(_DEVICE)

optimizer = torch.optim.AdamW(
    list(backbone.parameters()) + list(classification_head.parameters()),
    lr=1e-3,
)
criterion = nn.BCEWithLogitsLoss()

def train_step(env_batch, obj_batch, label_batch):
    backbone.train()
    classification_head.train()
    optimizer.zero_grad()

    # Ensure shape is [B, 3, N]
    env_batch = env_batch.permute(0, 2, 1).to(_DEVICE)
    obj_batch = obj_batch.permute(0, 2, 1).to(_DEVICE)

    # Concatenate for parallel processing [2*B, 3, 256]
    combined_input = torch.cat([env_batch, obj_batch], dim=0)

    # Forward through PTv3 backbone.
    # model(xyz) handles serialization and sparsification internally
    all_features = backbone(combined_input)
    assert all_features.shape == ((len(env_batch) + len(obj_batch)), feature_dim)

    env_features, obj_features = torch.chunk(all_features, 2, dim=0)

    logits = classification_head(env_features, obj_features)

    loss = criterion(logits, label_batch.view_as(logits).to(_DEVICE))
    loss.backward()
    optimizer.step()
    return loss.item()

dataset = PointCloudPairDataset("isaac_pc_data.npz", _PC_SIZE, _SEED)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop
for i in range(_EPOCHS):
    loss = 0
    for _, (batch_env, batch_obj, batch_labels) in enumerate(train_loader):
        loss += train_step(batch_env, batch_obj, batch_labels)
    print(f"Epoch {i+1} average loss: {loss / len(train_loader):.6f}")
print()

# Test trained models
# Since the same data is used for train / test, model should be "perfect"
# (trying to see if we can overfit to prove that model architecture / pipeline works)
env_pcs, obj_pcs, pc_pair_labels = load_pc_data("isaac_pc_data.npz", _PC_SIZE, _SEED)
env_batch = env_pcs.permute(0, 2, 1).to(_DEVICE)
obj_batch = obj_pcs.permute(0, 2, 1).to(_DEVICE)
combined_input = torch.cat([env_batch, obj_batch], dim=0)
all_features = backbone(combined_input)
env_features, obj_features = torch.chunk(all_features, 2, dim=0)
logits = classification_head(env_features, obj_features)
probabilities = torch.sigmoid(logits)
print(f"Stability scores:\n{probabilities}")
print(f"Ground-Truth Stability labels:\n{dataset.pc_pair_labels}")
