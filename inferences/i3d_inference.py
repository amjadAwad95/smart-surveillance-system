import json
import torch
import urllib
from PIL import Image
from inferences import BaseInference
from utils import inference_transform
from pytorchvideo.models.hub import i3d_r50


class I3DInferenceByFrames(BaseInference):
    """
    Performs video classification inference using a pretrained I3D (Inflated 3D ConvNet)
    model from PyTorchVideo.

    This class works with a list of frames instead of a video path. Frames are preprocessed,
    stacked into a tensor, and passed to the pretrained I3D model to obtain predictions.
    Class indices are mapped to human-readable labels using the Kinetics dataset class list.
    """

    def __init__(self):
        """
        Initializes the I3DInferenceByFrames object.

        - Loads the pretrained I3D model from PyTorchVideo.
        - Downloads the Kinetics-400 class names mapping file if not already present.
        - Builds a dictionary mapping class indices to human-readable labels.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_filename = "kinetics_classnames.json"
        try:
            urllib.URLopener().retrieve(json_url, json_filename)
        except:
            urllib.request.urlretrieve(json_url, json_filename)

        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)

        self.kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            self.kinetics_id_to_classname[v] = str(k).replace('"', "")

    def load_model(self):
        """
        Loads the pretrained I3D model from PyTorchVideo.

        :return: I3D model in evaluation mode on the selected device (CPU or GPU).
        """
        model = i3d_r50(pretrained=True).to(self.device)

        return model

    def inference(self, frames):
        """
        Performs inference on a list of video frames using the I3D model.

        Steps:
        1. Converts each frame to a PIL image.
        2. Applies preprocessing transform to each frame.
        3. Stacks frames into a 5D tensor (batch_size, channels, time, height, width).
        4. Runs inference using the pretrained I3D model.
        5. Maps the predicted class index to its human-readable label.

        :param frames: List of frames (each as a NumPy array in RGB format).
        :return: Predicted class name (string).
        """
        video_tensor_list = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = inference_transform(frame_pil)
            video_tensor_list.append(frame_tensor)

        video_tensor = torch.stack(video_tensor_list)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0).float()

        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(video_tensor)

        return self.kinetics_id_to_classname[int(output.argmax(1)[0])]
