from torch.utils.data import random_split
from data import FrameDataset, VideoDataset


def load_data(main_path, transform, is_split=False, max_frames=16, train_ratio=0.8):
    """
    Loads and prepares the dataset for video-based classification.

    This function creates a FrameDataset from the given directory of frames,
    wraps it into a VideoDataset (grouping frames into videos), and optionally
    splits the dataset into training and validation subsets.

    :param main_path: Path to the main directory containing subdirectories for each label.
    :param transform: Transformation pipeline to be applied to each frame.
    :param is_split: Boolean flag indicating whether to split the dataset into
                    training and validation sets (default: False).
    :param max_frames: Maximum number of frames per video. Videos will be padded
                    or truncated to this length (default: 16).
    :param train_ratio: Proportion of the dataset to use for training when splitting
                        (default: 0.8).
    :return:
        - If is_split=True: A tuple (train_dataset, val_dataset).
        - If is_split=False: A single VideoDataset instance containing the entire dataset.
    """
    if is_split:
        frame_dataset = FrameDataset(main_path=main_path, transform=transform)
        Video_dataset = VideoDataset(frame_dataset=frame_dataset, max_frames=max_frames)

        train_size = int(train_ratio * len(Video_dataset))
        val_size = len(Video_dataset) - train_size

        train_dataset, val_dataset = random_split(Video_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    frame_dataset = FrameDataset(main_path=main_path, transform=transform)
    Video_dataset = VideoDataset(frame_dataset=frame_dataset, max_frames=max_frames)

    return Video_dataset
