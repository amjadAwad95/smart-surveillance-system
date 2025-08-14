from torch.utils.data import Dataset


class EncodedVideoDataset(Dataset):
    """
    EncodedVideoDataset is a PyTorch Dataset that wraps an existing video dataset.
    It processes each video by permuting the frame dimensions and converting the label
    into a binary format (0 for 'NormalVideos', 1 for all other labels).
    """

    def __init__(self, video_dataset):
        """
        Initializes the EncodedVideoDataset.
        :param video_dataset: An existing dataset object that returns (frames, label) pairs.
        """
        self.video_dataset = video_dataset

    def __len__(self):
        """
        Returns the total number of videos in the dataset.
        :return: Length of the dataset.
        """
        return len(self.video_dataset)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single video sample.
        :param idx: Index of the video to retrieve.
        :return: A tuple (frames, label) where:
                - frames: Tensor of video frames with dimensions permuted from
                        (time, channels, height, width) to (channels, time, height, width).
                - label: Integer (0 for 'NormalVideos', 1 for all other labels).
        """
        frames, label = self.video_dataset[idx]
        frames = frames.permute(1, 0, 2, 3)
        label = 0 if label == "NormalVideos" else 1

        return frames, label
