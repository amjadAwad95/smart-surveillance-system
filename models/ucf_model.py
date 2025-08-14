import torch
import torch.nn as nn
from enum import Enum


class FineTueningStrategy(Enum):
    """
    Enumeration of fine-tuning strategies for the UCFModel.
    BLOCK: Unfreeze the last N blocks of the model.
    LAYER: Unfreeze the last N individual layers of the model.
    """
    BLOCK = "block"
    LAYER = "layer"


class UCFModel(nn.Module):
    """
    UCFModel is a wrapper around a pretrained PyTorchVideo model for binary video classification.
    It supports partial fine-tuning by unfreezing the last N blocks or layers based on the chosen strategy.
    """
    def __init__(self, model_name="i3d_r50", unfreeze_number=1, strategy: FineTueningStrategy=FineTueningStrategy.BLOCK, inference = False):
        """
        Initializes the UCFModel.

        :param model_name: Name of the pretrained PyTorchVideo model to load (default: 'i3d_r50').
        :param unfreeze_number: Number of blocks or layers to unfreeze for fine-tuning.
                                If less than 1, it defaults to 1.
        :param strategy: Fine-tuning strategy to use:
                        - FineTueningStrategy.BLOCK: Unfreeze last N blocks.
                        - FineTueningStrategy.LAYER: Unfreeze last N layers.
        :param inference: Boolean flag to indicate whether the model is used for inference only.
                        If True, parameters are not frozen or unfrozen — the model remains fully trainable.
        """
        super().__init__()
        self.model_name = model_name
        self.unfreeze_number = unfreeze_number
        self.strategy = strategy
        self.inference = inference

        self.model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)

        if self.unfreeze_number < 1:
            self.unfreeze_number = 1
            print("Number of unfreeze set to 1 as it was less than 1.")

        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, 2)

        if not self.inference:
            for param in self.model.parameters():
                param.requires_grad = False

            if self.strategy == FineTueningStrategy.LAYER:
                self.__unfreeze_layers()
            
            else:
                self.__unfreeze_blocks()


    def __unfreeze_blocks(self):
        """
        Unfreezes the last `unfreeze_number` blocks of the model for fine-tuning.
        If `unfreeze_number` exceeds the number of blocks, it is adjusted accordingly.
        """
        if self.unfreeze_number > len(self.model.blocks):
            self.unfreeze_number = len(self.model.blocks)
            print(f"Number of unfreeze blocks set to {self.unfreeze_number} as it exceeds the number of blocks in the model.")
        
        unfreeze_index_blocks = [-index for index in range(1, self.unfreeze_number + 1)]

        for index in unfreeze_index_blocks:
            for param in self.model.blocks[index].parameters():
                param.requires_grad = True
    

    def __unfreeze_layers(self):
        """
        Unfreezes the last `unfreeze_number` individual layers (parameters) of the model.
        If `unfreeze_number` exceeds the number of layers, it is adjusted accordingly.
        """
        params_list = list(self.model.parameters())

        if self.unfreeze_number > len(params_list):
            self.unfreeze_number = len(params_list)
            print(f"Number of unfreeze layers set to {self.unfreeze_number} as it exceeds the number of layers in the model.")

        layer_index = [-index for index in range(1, self.unfreeze_number + 1)]

        for index in layer_index:
            params_list[index].requires_grad = True


    def forward(self, frames):
        """
        Performs a forward pass of the input frames through the model.

        :param frames: A tensor of shape (batch_size, channels, time, height, width).
        :return: Output logits of shape (batch_size, num_classes).
        """
        return self.model(frames)
