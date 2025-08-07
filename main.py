from torch.utils.data import DataLoader
from data import VideoDataset, FrameDataset
from utils import image_transform


main_path = "data/dataset/train"

print("Loading FrameDataset...")
frame_dataset = FrameDataset(main_path=main_path, transform=image_transform)
print("FrameDataset loaded with", len(frame_dataset), "frames.")


print("Creating VideoDataset...")
video_dataset = VideoDataset(frame_dataset)
print("VideoDataset created with", len(video_dataset), "videos.")


print("Creating DataLoader for VideoDataset...")
video_dataloader = DataLoader(video_dataset, batch_size=4, shuffle=True)
print("DataLoader created.")


for images, labels in video_dataloader:
    print(images.shape, labels)
    break 