import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_one_epoch(model, dataloader, criterion, optimizer, device, metrics=None):
    model.train()
    logs = {"loss": 0.0}
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        logs["loss"] += loss.item()
    logs["loss"] /= len(dataloader)
    return logs

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, metrics=None):
    model.eval()
    logs = {"dice_loss": 0.0}
    for images, masks in tqdm(dataloader, desc="Validation"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        logs["dice_loss"] += loss.item()
    logs["dice_loss"] /= len(dataloader)
    return logs
