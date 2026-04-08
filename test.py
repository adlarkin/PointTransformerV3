import torch
import torch.nn as nn
import numpy as np
from model import PointTransformerV3
from stability_predictor import StabilityPredictor

_PC_SIZE = 256
_EPOCHS = 100

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: figure out how to seed the random indexing / permutations below
# (for reproducability)

def load_pc_data(file: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(file, allow_pickle=True)
    metadata = data["metadata"].item()
    features = data["features"]
    labels = data["labels"]

    env_list, obj_list = [], []

    for pc, ids in zip(features, labels):
        env_points = pc[ids == metadata["environment"]]
        obj_points = pc[ids == metadata["object"]]

        def resample(points, size):
            indices = np.random.choice(len(points), size, replace=len(points) < size)
            return points[indices]

        env_list.append(torch.from_numpy(resample(env_points, _PC_SIZE)).float())
        obj_list.append(torch.from_numpy(resample(obj_points, _PC_SIZE)).float())

    return torch.stack(env_list), torch.stack(obj_list)

env_pcs, obj_pcs = load_pc_data("isaac_pc_data.npz")

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

# TODO: extract feaure_dim programmatically from backbone so that it's not hardcoded?
classification_head = StabilityPredictor(feature_dim=512).to(_DEVICE)

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

    # Forward through PTv3 backbone
    # model(xyz) handles serialization and sparsification internally
    all_features = backbone(combined_input)

    env_features, obj_features = torch.chunk(all_features, 2, dim=0)

    logits = classification_head(env_features, obj_features)

    loss = criterion(logits, label_batch.view_as(logits).to(_DEVICE))
    loss.backward()
    optimizer.step()
    return loss.item()

# Temporary hardcoded stability labels for the pointcloud data
# TODO: save this info along with the pointcloud data so that it can be
# loaded with the dataset
pc_classification_labels = torch.cat((torch.ones(5), torch.zeros(5)))

# Training Loop
for i in range(_EPOCHS):
    # Select a random batch (here size 4)
    idx = torch.randint(0, env_pcs.shape[0], (4,))
    batch_env = env_pcs[idx]
    batch_obj = obj_pcs[idx]
    batch_classification_labels = pc_classification_labels[idx]

    loss = train_step(batch_env, batch_obj, batch_classification_labels)
    print(f"Epoch {i+1} loss: {loss:.6f}")
print()

# Test trained models
# Since the same data is used for train / test, model should be "perfect"
# (trying to see if we can overfit to prove that model architecture / pipeline works)
env_batch = env_pcs.permute(0, 2, 1).to(_DEVICE)
obj_batch = obj_pcs.permute(0, 2, 1).to(_DEVICE)
combined_input = torch.cat([env_batch, obj_batch], dim=0)
all_features = backbone(combined_input)
env_features, obj_features = torch.chunk(all_features, 2, dim=0)
logits = classification_head(env_features, obj_features)
probabilities = torch.sigmoid(logits)
print(f"Stability scores:\n{probabilities}")
print(f"Ground-Truth Stability labels:\n{pc_classification_labels}")
