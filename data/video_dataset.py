import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    VideoDataset is a PyTorch Dataset that groups frames into videos based on labels and part numbers.
    Each video is padded or truncated to a fixed number of frames.
    """
    def __init__(self, frame_dataset, max_frames=16):
        """
        Initializes the VideoDataset.
        :param frame_dataset: An instance of FrameDataset containing the frames.
        :param max_frames: The maximum number of frames per video. Videos will be padded or truncated to this length.
        """
        self.frame_dataset = frame_dataset
        self.max_frames = max_frames
        video_dict = defaultdict(list)

        for frame in frame_dataset.dataset:
            video_dict[(frame['label'], frame['part_number'])].append(frame)

        self.videos = [sorted(value, key=lambda x:x['frame_idx']) for value in video_dict.values()]
        

    def __len__(self):
        """
        Returns the total number of videos in the dataset.
        :return: Length of the dataset.
        """
        return len(self.videos)
    

    def __pad_or_truncate(self, frames):
        """
        Pads or truncates the list of frames to ensure it has a fixed length.
        :param frames: List of frames to be padded or truncated.
        :return: Padded or truncated list of frames.
        """
        length = len(frames)

        if length > self.max_frames:
            indices = np.linspace(0, length - 1, self.max_frames, dtype=int)
            return [frames[i] for i in indices]
        elif length < self.max_frames:
            padding = [frames[-1]] * (self.max_frames - length)
            return frames + padding
        
        return frames
    

    def __getitem__(self, idx):
        """
        Returns a single video and its associated label.
        :param idx: Index of the video to retrieve.
        :return: A tuple containing a tensor of frames and the label.
        """
        video=self.videos[idx]
        video = self.__pad_or_truncate(video)

        frames = []
        label = video[0]["label"]

        for frame in video:
            image = self.frame_dataset[frame["index"]][0]
            frames.append(image)
            
        frames = torch.stack(frames)
        return frames, label
        