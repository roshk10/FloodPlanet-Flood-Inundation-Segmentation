from torch.utils.data import DataLoader

from dataset import FloodPlanetDataset


# Published paper setting
BATCH_SIZE = 8


def create_dataloader(
    root_dir,
    events,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
):
    """
    Create a PyTorch DataLoader for FloodPlanet.

    Parameters
    ----------
    root_dir : str
        Location of the FloodPlanet dataset.

    events : list[str]
        Flood events to include.

    shuffle : bool
        Shuffle samples when True.
        Normally True for training and False for evaluation.

    batch_size : int
        Number of patches processed together.

    num_workers : int
        Number of worker processes used to load data.
        We start with 0 for reliable debugging.

    pin_memory : bool
        Helps transfer tensors to CUDA efficiently.
    """

    dataset = FloodPlanetDataset(
        root_dir=root_dir,
        events=events,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader