import torch

def save_model(model, optimizer, epoch, val_loss, path):
    state = {
        "model_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss
    }
    torch.save(state, path)
    print(f"[INFO] Model saved to {path}")

def load_model(model, optimizer=None, path="checkpoints/best_model.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_dict"])
    print(f"[INFO] Model loaded from {path}")
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"], checkpoint["val_loss"]
    
    return model
    