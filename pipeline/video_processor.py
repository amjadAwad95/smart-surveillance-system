import time
import queue
import logging
import winsound
import threading
import cv2 as cv
from collections import deque


class RealTimeVideoProcessorWithTerminal:
    """
    Real-time video processor for crime detection with YOLO object detection,
    UCF action recognition, and optional audio alerts.

    This class uses multithreading to handle:
    - Frame reading from a video file.
    - Object detection via YOLO.
    - Action recognition using UCF model (crime detection).
    - Real-time display with overlayed statistics.

    Queues and buffers are used to manage frames across threads. Statistics
    such as FPS, frame counts, and detection counts are tracked and displayed.
    """

    def __init__(
        self,
        video_path,
        yolo_inference,
        ucf_inference,
        normal_inference,
        frame_skip=3,
        buffer_size=8,
        beep=False,
    ):
        """
        Initializes the real-time video processor.

        :param video_path: Path to the video file to process.
        :param yolo_inference: Function or object handling YOLO inference on frames.
        :param ucf_inference: Function or object handling UCF action recognition.
        :param normal_inference: Function for handling normal action predictions.
        :param frame_skip: Number of frames to skip between processing steps.
        :param buffer_size: Number of frames to keep for action recognition buffer.
        :param beep: If True, play a beep sound when a crime is detected.
        """
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.buffer_size = buffer_size
        self.yolo_inference = yolo_inference
        self.ucf_inference = ucf_inference
        self.normal_inference = normal_inference
        self.beep = beep

        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=3)
        self.ucf_queue = queue.Queue(maxsize=8)

        self.ucf_buffer = deque(maxlen=self.buffer_size)

        self.running = True
        self.frame_count = 0
        self.crime_detected_count = 0
        self.total_processed_frames = 0

        self.start_time = time.time()
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        self.display_fps_counter = 0
        self.last_display_fps_time = time.time()
        self.display_fps = 0

        self.stats_lock = threading.Lock()

        self.skip_heavy_logging = True
        self.process_ucf_every_n = 2

        self.threads = []

        self.sound_frequency = 500
        self.sound_duration = 1000

    def frame_reader_thread(self):
        """
        Thread for reading frames from the video file.

        - Opens the video file using OpenCV.
        - Reads frames while respecting the frame skip value.
        - Maintains FPS statistics.
        - Pushes frames into the frame queue for downstream processing.
        """
        print("Starting frame reader thread...")

        cap = cv.VideoCapture(self.video_path)

        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            self.running = False
            return

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv.CAP_PROP_FPS)
        print(f"Video info: {total_frames} frames, {video_fps} FPS")

        target_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30
        last_frame_time = time.time()

        try:
            while self.running:
                ret, frame = cap.read()

                if not ret:
                    print("End of video reached")
                    self.running = False
                    break

                with self.stats_lock:
                    self.frame_count += 1

                if (self.frame_count % self.frame_skip) != 0:
                    continue

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                self.fps_counter += 1
                current_time = time.time()
                if (current_time - self.last_fps_time) >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time

                elapsed = current_time - last_frame_time
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                last_frame_time = time.time()

        except Exception as e:
            print(f"Error in frame reader: {e}")
        finally:
            cap.release()
            print("Frame reader thread finished")

    def detection_thread(self):
        """
        Processes frames for object detection.

        - Resizes frames to a manageable resolution.
        - Runs YOLO inference on each frame.
        - Sends frames to the display and UCF inference queues.
        - Handles queue overflow safely.
        """
        print("Starting detection thread...")

        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)

                h, w = frame.shape[:2]
                if h > 480:
                    scale = 480.0 / h
                    new_w = int(w * scale)
                    resized_frame = cv.resize(frame, (new_w, 480))
                else:
                    resized_frame = frame

                detect_image = cv.resize(resized_frame, (224, 224))
                detect_frame = self.yolo_inference(detect_image)

                if detect_frame is not None:
                    display_frame = detect_frame
                else:
                    display_frame = resized_frame

                try:
                    self.ucf_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.ucf_queue.get_nowait()
                        self.ucf_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                try:
                    self.display_queue.put_nowait(display_frame)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(display_frame)
                    except queue.Empty:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")

        print("Detection thread finished")

    def ucf_inference_thread(self):
        """
        Thread for performing UCF action recognition.

        - Collects frames into a buffer.
        - Runs the UCF model every N frames (configurable).
        - If crime is detected, increments the detection counter and optionally beeps.
        - Logs normal predictions at intervals if heavy logging is enabled.
        """
        print("Starting UCF inference thread...")

        ucf_frame_counter = 0

        while self.running or not self.ucf_queue.empty():
            try:
                frame = self.ucf_queue.get(timeout=0.5)
                ucf_frame_counter += 1

                if ucf_frame_counter % self.process_ucf_every_n != 0:
                    continue

                self.ucf_buffer.append(frame)

                if len(self.ucf_buffer) == self.buffer_size:
                    frame_list = list(self.ucf_buffer)

                    predict = self.ucf_inference(frame_list)

                    with self.stats_lock:
                        self.total_processed_frames += 1

                    if predict == 1:
                        with self.stats_lock:
                            self.crime_detected_count += 1

                        if self.beep:
                            winsound.Beep(self.sound_frequency, self.sound_duration)
                        print(
                            f"🚨 Crime detected at frame {self.frame_count}! (Total: {self.crime_detected_count})"
                        )

                    else:
                        if (
                            not self.skip_heavy_logging
                            or self.total_processed_frames % 20 == 0
                        ):
                            normal_predict = self.normal_inference(frame_list)
                            print(f"✅ Normal: {normal_predict}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in UCF inference thread: {e}")

        print("UCF inference thread finished")

    def display_thread(self):
        """
        Displays processed frames with overlays in a GUI window.

        - Shows FPS, frame count, and detected crimes.
        - Handles user input:
            - 'q' to quit
            - 's' to save the current frame
            - 'p' to print statistics
            - 'f' to toggle heavy logging
        - Calculates and updates display FPS.
        """
        print("Starting display thread...")

        while self.running:
            try:
                detect_frame = self.display_queue.get(timeout=0.1)

                self.display_fps_counter += 1
                current_time = time.time()
                if (current_time - self.last_display_fps_time) >= 1.0:
                    self.display_fps = self.display_fps_counter
                    self.display_fps_counter = 0
                    self.last_display_fps_time = current_time

                detect_frame = cv.resize(detect_frame, (600, 600))
                info_frame = self.add_info_overlay(detect_frame)

                cv.imshow("Real-time Crime Detection", info_frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("User requested quit...")
                    self.running = False
                    break
                elif key == ord("s"):
                    timestamp = int(time.time())
                    filename = f"detection_frame_{timestamp}.jpg"
                    cv.imwrite(filename, info_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord("p"):
                    self.print_statistics()
                elif key == ord("f"):
                    self.skip_heavy_logging = not self.skip_heavy_logging
                    print(
                        f"Heavy logging: {'OFF' if self.skip_heavy_logging else 'ON'}"
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display thread: {e}")

        cv.destroyAllWindows()
        print("Display thread finished")

    def add_info_overlay(self, frame):
        """
        Adds informative overlays to a frame.

        - Displays read FPS, display FPS, frame count, and crime count.
        - Adds key instructions for user controls at the bottom of the frame.

        return: np.ndarray: Frame with overlay text.
        """
        info_frame = frame.copy()

        font = cv.FONT_HERSHEY_TRIPLEX
        font_scale = 0.3
        color = (0, 255, 0)
        thickness = 1

        with self.stats_lock:
            fps_text = f"Read FPS: {self.current_fps}"
            display_fps_text = f"Display FPS: {self.display_fps}"
            frame_text = f"Frame: {self.frame_count}"
            crimes_text = f"Crimes: {self.crime_detected_count}"

        cv.putText(info_frame, fps_text, (10, 25), font, font_scale, color, thickness)
        cv.putText(
            info_frame,
            display_fps_text,
            (10, 50),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
        )
        cv.putText(info_frame, frame_text, (10, 75), font, font_scale, color, thickness)
        cv.putText(
            info_frame, crimes_text, (10, 100), font, font_scale, (0, 0, 255), thickness
        )

        cv.putText(
            info_frame,
            "q:quit s:save p:stats f:toggle-log",
            (10, info_frame.shape[0] - 10),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

        return info_frame

    def print_statistics(self):
        """
        Prints runtime statistics to the terminal.

        - Total frames read and processed.
        - Read FPS and display FPS.
        - Number of crime detections and crime detection rate.
        - Current queue sizes for frame, UCF, and display.
        """
        with self.stats_lock:
            runtime = time.time() - self.start_time
            print(f"\n=== Processing Statistics ===")
            print(f"Runtime: {runtime:.2f} seconds")
            print(f"Total frames read: {self.frame_count}")
            print(f"Read FPS: {self.current_fps}")
            print(f"Display FPS: {self.display_fps}")
            print(
                f"Frames processed for action recognition: {self.total_processed_frames}"
            )
            print(f"Crime detections: {self.crime_detected_count}")
            if self.total_processed_frames > 0:
                crime_rate = (
                    self.crime_detected_count / self.total_processed_frames
                ) * 100
                print(f"Crime detection rate: {crime_rate:.2f}%")
            print(
                f"Queue sizes - Frame: {self.frame_queue.qsize()}, UCF: {self.ucf_queue.qsize()}, Display: {self.display_queue.qsize()}"
            )
            print("========================\n")

    def start_processing(self):
        """
        Starts the video processing pipeline.

        - Spawns threads for reading frames, running detection, action recognition,
            and displaying results.
        - Joins threads and waits for completion.
        - Prints final statistics upon termination.
        """
        print(
            f"🚀 Starting OPTIMIZED real-time video processing for: {self.video_path}"
        )
        print(f"Frame skip: {self.frame_skip}, Buffer size: {self.buffer_size}")
        print(
            "Press 'q' to quit, 's' to save frame, 'p' for stats, 'f' to toggle logging"
        )

        self.threads = [
            threading.Thread(target=self.frame_reader_thread, name="FrameReader"),
            threading.Thread(target=self.detection_thread, name="Detection"),
            threading.Thread(target=self.ucf_inference_thread, name="UCFInference"),
            threading.Thread(target=self.display_thread, name="Display"),
        ]

        for thread in self.threads:
            thread.daemon = True
            thread.start()
            print(f"✓ Started {thread.name} thread")

        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            self.running = False
            print("\n🛑 Interrupted by user")

        self.print_statistics()
        print("🏁 Video processing completed!")
