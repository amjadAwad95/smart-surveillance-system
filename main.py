from inferences.ucf_inference import UCFInferenceFromPath

video_path = "Fighting013_x264A.mp4"
repo_id = "amjad-awad/ucf-i3d-model-by-block-lr-0.001"
max_frames = 16

ucf_inference = UCFInferenceFromPath(repo_id=repo_id)

label = ucf_inference.inference(video_path=video_path, max_frames=max_frames)

print("Crime" if label else "Not Crime")