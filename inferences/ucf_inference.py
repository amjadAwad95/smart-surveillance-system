import torch
import cv2 as cv
import numpy as np
from PIL import Image
from models import UCFModel
from utils import inference_transform
from .base_inference import BaseInference
from huggingface_hub import hf_hub_download


class UCFInferenceFromPath(BaseInference):
    """
    Performs video classification inference using a pretrained UCFModel
    loaded from the Hugging Face Hub.

    This class takes the path to a video file, extracts frames, applies
    preprocessing, and runs the model to obtain predictions.
    """

    def __init__(self, repo_id):
        """
        Initializes the UCFInferenceFromPath object.

        :param repo_id: Hugging Face Hub repository ID containing the
                        pretrained model checkpoint file `ucf_model.pth`.
        """
        self.repo_id = repo_id

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """
        Loads the pretrained UCFModel from the Hugging Face Hub if it is not already loaded.

        :return: The loaded UCFModel in evaluation mode.
        """
        if self.model:
            return self.model

        model_path = hf_hub_download(repo_id=self.repo_id, filename="ucf_model.pth")
        state_dict = torch.load(model_path)

        self.model = UCFModel(inference=True).to(device=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        return self.model

    def load_video(self, video_path):
        """
        Reads a video from the given file path and extracts its frames.

        :param video_path: Path to the video file.
        :return: List of frames in RGB format (each frame as a NumPy array).
        """
        frames_list = []
        cap = cv.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames_list.append(frame)

        cap.release()

        return frames_list

    def inference(self, video_path, max_frames=16):
        """
        Performs inference on a video file.

        Steps:
        1. Loads the pretrained model (if not already loaded).
        2. Reads the video and extracts all frames.
        3. Samples a fixed number of frames (max_frames) evenly spaced from the video.
        4. Applies the preprocessing transform to each frame.
        5. Stacks frames into a tensor and runs the model in evaluation mode.
        6. Returns the predicted class index.

        :param video_path: Path to the video file.
        :param max_frames: Number of frames to sample from the video (default: 16).
        :return: Predicted class index as a tensor (0 or 1).
        :raises ValueError: If the video contains no frames.
        """
        if not self.model:
            self.load_model()

        frames_list = self.load_video(video_path)

        num_frames = len(frames_list)
        if num_frames == 0:
            raise ValueError("No frames found in the video.")

        indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
        sampled_frames = [frames_list[i] for i in indices]

        video_tensor_list = []
        for frame in sampled_frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = inference_transform(frame_pil)
            video_tensor_list.append(frame_tensor)

        video_tensor = torch.stack(video_tensor_list)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0).float()

        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(video_tensor)

        return output.argmax(1)
