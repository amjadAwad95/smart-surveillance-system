import torch
from torch.utils.data import Dataset
from itertools import groupby



class VideoDataset(Dataset):
    """
    VideoDataset is a PyTorch Dataset that groups frames into videos based on labels and part numbers.
    Each video is padded or truncated to a fixed number of frames.
    """
    def __init__(self, frame_dataset, max_frames=128):
        """
        Initializes the VideoDataset.
        :param frame_dataset: An instance of FrameDataset containing the frames.
        :param max_frames: The maximum number of frames per video. Videos will be padded or truncated to this length.
        """
        self.frame_dataset = frame_dataset
        self.max_frames = max_frames
        self.videos = []

        for _, video_group in groupby(frame_dataset.dataset, key=lambda x: (x['label'], x['part_number'])):
            self.videos.append(list(video_group))

        for idx, video in enumerate(self.videos):
            self.videos[idx] = sorted(video, key=lambda x: x['frame_idx'])
        

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
            return frames[:self.max_frames]
        elif length < self.max_frames:
            padding =[frames[-1]] * (self.max_frames - length)
            return frames + padding
        
        return frames
    

    def __getitem__(self, idx):
        """
        Returns a single video and its associated label.
        :param idx: Index of the video to retrieve.
        :return: A tuple containing a tensor of frames and the label.
        """
        video=self.videos[idx]

        frames = []
        label = video[0]["label"]

        for frame in video:
            image, _, _, _ = self.frame_dataset[frame["index"]]
            frames.append(image)
            
        frames = self.__pad_or_truncate(frames)
        frames = torch.stack(frames)
    
        return frames, label
        