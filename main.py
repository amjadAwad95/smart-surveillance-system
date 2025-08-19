from inferences import (
    HuggingfaceInferenceByFrames,
    I3DInferenceByFrames,
    UCFInferenceByFrames,
    YOLOInference,
)
from pipeline import RealTimeVideoProcessorWithTerminal

yolo_inference = YOLOInference()
ucf_inference = HuggingfaceInferenceByFrames(
    "Nikeytas/videomae-crime-detector-ultra-v1"
)
# ucf_inference = UCFInferenceByFrames("amjad-awad/ucf-i3d-model-by-block-lr-0.001")
normal_inference = I3DInferenceByFrames()


video_path = "videoplayback.mp4"

processor = RealTimeVideoProcessorWithTerminal(
    video_path,
    yolo_inference,
    ucf_inference,
    normal_inference,
    frame_skip=1,
    buffer_size=16,
    # beep=True,
)

processor.start_processing()
