import os
import re
from PIL import Image
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """
    FrameDataset is a PyTorch Dataset that loads frames from a directory structure.
    Each frame is associated with a label, part number, and frame index.
    The dataset is structured such that each label has its own subdirectory containing frames.
    """
    def __init__(self, main_path, transform=None):
        """
        Initializes the FrameDataset.
        :param main_path: Path to the main directory containing subdirectories for each label.
        :param transform: Optional transform to be applied on a sample.
        """
        self.main_path = main_path
        self.transform = transform

        self.regex_patterns = [
            re.compile(r"([A-Za-z]+[0-9]+_x[0-9]+)_(\d+)\.png"),
            re.compile(r"([A-Za-z]+_[A-Za-z]+[0-9]+_x[0-9]+)_(\d+)\.png"),
            re.compile(r"([A-Za-z]+_[A-Za-z]+_[0-9]+_x[0-9]+)_(\d+)\.png"),
        ]

        self.dataset = []

        paths = [f"{main_path}/{label}" for label in os.listdir(main_path)]

        index = 0

        for path in paths:
            label = path.split("/")[-1]

            for image in os.listdir(path):
                for pattern in self.regex_patterns:
                    match = pattern.match(image)
                    if match:
                        underscore_count = image.count("_")
                        if underscore_count == 2:
                            split = image.split("_")
                            part = split[0]
                            part_number = re.findall(r"\d+", part)[0]
                            frame_idx = int(split[-1].replace(".png", ""))
                            self.dataset.append(
                                {
                                    "label": label,
                                    "part_number": part_number,
                                    "image": image,
                                    "frame_idx": frame_idx,
                                    "index": index,
                                }
                            )
                            index += 1
                        elif underscore_count == 3:
                            split = image.split("_")
                            part = split[1]
                            part_number = re.findall(r"\d+", part)[0]
                            frame_idx = int(split[-1].replace(".png", ""))
                            self.dataset.append(
                                {
                                    "label": label,
                                    "part_number": part_number,
                                    "image": image,
                                    "frame_idx": frame_idx,
                                    "index": index,
                                }
                            )
                            index += 1
                        else:
                            split = image.split("_")
                            part = split[2]
                            part_number = part
                            frame_idx = int(split[-1].replace(".png", ""))
                            self.dataset.append(
                                {
                                    "label": label,
                                    "part_number": part_number,
                                    "image": image,
                                    "frame_idx": frame_idx,
                                    "index": index,
                                }
                            )
                            index += 1
                        break

    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        :return: Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single frame and its associated label, part number, and frame index.
        :param idx: Index of the frame to retrieve.
        :return: A tuple containing the image, label, part number, and frame index.
        """
        frame_info = self.dataset[idx]
        label = frame_info["label"]
        part_number = frame_info["part_number"]
        frame_idx = frame_info["frame_idx"]
        image_path = os.path.join(self.main_path, label, frame_info["image"])

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label, part_number, frame_idx
