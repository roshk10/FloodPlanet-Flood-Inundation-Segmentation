from pathlib import Path

import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one complete epoch.

    Returns:
        average training loss
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels,
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate weighted loss
        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Training loader contains no samples."
        )

    return total_loss / total_samples


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
):
    """
    Save model and optimizer state.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )