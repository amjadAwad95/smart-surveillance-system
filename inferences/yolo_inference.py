import math
import logging
import cv2 as cv
from ultralytics import YOLO
from inferences import BaseInference
from deep_sort_realtime.deepsort_tracker import DeepSort


logging.getLogger("ultralytics").setLevel(logging.WARNING)


class YOLOInference(BaseInference):
    """
    Performs object detection and multi-object tracking using a YOLOv8 model
    combined with DeepSort for real-time tracking.

    This class loads a YOLO model, runs inference on input images or video frames,
    extracts bounding boxes and class labels, and then applies the DeepSort tracker
    to maintain consistent IDs across frames.
    """

    def __init__(self, model_name="yolov8l.pt", fuse=False):
        """
        Initializes the YOLOInference object.

        :param model_name: Name or path of the YOLOv8 model to load
                            (default: "yolov8l.pt").
        :param fuse: If True, fuses model layers for faster inference (default: False).
        """
        self.model_name = model_name
        self.fuse = fuse

        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = DeepSort(
            max_age=5,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            polygon=False,
            today=None,
        )

    def load_model(self):
        """
        Loads the YOLOv8 model for inference.

        :return: The YOLOv8 model instance.
        """
        model = YOLO(self.model_name)
        if self.fuse:
            model.fuse()

        return model

    def plot_boxes(self, results, img):
        """
        Processes YOLO detection results and extracts bounding boxes,
        confidence scores, and class names.

        :param results: YOLO detection results from the model.
        :param img: The input image on which detections were made.
        :return: A tuple (detections, img) where:
                    - detections: List of tuples in the format ([x, y, w, h], conf, class_name).
                    - img: The original image (unchanged).
        """
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cls = int(box.cls[0])
                currentClass = self.names[cls]

                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.5:
                    detections.append(([x1, y1, w, h], conf, currentClass))

        return detections, img

    def track_detection(self, detections, img, tracker: DeepSort):
        """
        Updates the DeepSort tracker with new detections and draws bounding
        boxes and IDs on the image.

        :param detections: List of detections in the format ([x, y, w, h], conf, class_name).
        :param img: The input image.
        :param tracker: DeepSort tracker instance.
        :return: Image with drawn bounding boxes and object IDs.
        """
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_name = track.det_class

            x1, y1, x2, y2 = map(int, ltrb)

            img = cv.rectangle(
                img, (x1, y1), (x2, y2), color=(255, 0, 255), thickness=1
            )

            img = cv.putText(
                img,
                f"{class_name}, {track_id}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_TRIPLEX,
                fontScale=0.35,
                color=(255, 0, 255),
                thickness=1,
            )

        return img

    def inference(self, image, stream=True):
        """
        Runs object detection and tracking on a single image or frame.

        Steps:
        1. Runs YOLOv8 detection on the image.
        2. Extracts bounding boxes, confidence scores, and class names.
        3. Updates the DeepSort tracker with current detections.
        4. Draws tracked bounding boxes and IDs on the image.
        5. Returns the annotated frame.

        :param image: Input image (NumPy array).
        :param stream: Whether to run YOLO in streaming mode (default: True).
        :return: Image annotated with bounding boxes and tracking IDs.
        """
        results = self.model(image, stream=stream)
        detections, frames = self.plot_boxes(results, image)
        detect_frame = self.track_detection(detections, frames, self.tracker)

        return detect_frame
