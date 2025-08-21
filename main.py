import os
import time
import queue
import atexit
import logging
import winsound
import tempfile
import threading
import cv2 as cv
import streamlit as st
from collections import deque
from datetime import datetime
from inferences import HuggingfaceInferenceByFrames, YOLOInference, I3DInferenceByFrames


@st.cache_resource
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join("logs", f"video_processing_{timestamp}.log")

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Video processing session initialized. Log file: {log_filename}")
    return logger


if "logger" not in st.session_state:
    st.session_state.logger = setup_logging()


@st.cache_resource
def load_models():
    yolo = YOLOInference()
    ucf = HuggingfaceInferenceByFrames("Nikeytas/videomae-crime-detector-ultra-v1")
    normal = I3DInferenceByFrames()
    st.session_state.logger.info("Models loaded successfully")
    return yolo, ucf, normal


yolo_model, ucf_model, i3d_model = load_models()

if "frame_skip" not in st.session_state:
    st.session_state.frame_skip = 5
if "yolo_skip" not in st.session_state:
    st.session_state.yolo_skip = 2
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "threads_started" not in st.session_state:
    st.session_state.threads_started = False
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False
if "stop_detection" not in st.session_state:
    st.session_state.stop_detection = False
if "video_cap" not in st.session_state:
    st.session_state.video_cap = None
if "beep" not in st.session_state:
    st.session_state.beep = False


# Global queues and buffers
yolo_input_queue = queue.Queue(maxsize=2)
yolo_output_queue = queue.Queue(maxsize=2)
ucf_queue = queue.Queue(maxsize=1)
frame_buffer = deque(maxlen=16)

# Shared variables for frame display
current_display_frame = None
frame_lock = threading.Lock()


# --- Worker Functions ---
def yolo_worker(model, input_queue, output_queue, stop_event, logger):
    global current_display_frame
    logger.info("YOLO Worker started")

    while not stop_event.is_set():
        try:
            frame_data = input_queue.get(timeout=1)
            if frame_data is None:
                break

            frame, frame_id = frame_data
            result = model(frame)

            with frame_lock:
                current_display_frame = result

            if not output_queue.full():
                try:
                    output_queue.put((result, frame_id), timeout=0.1)
                    logger.debug(f"YOLO result queued for frame {frame_id}")
                except queue.Full:
                    logger.warning("YOLO output queue full, dropping result")

            input_queue.task_done()

        except queue.Empty:
            continue
        except Exception:
            logger.exception("YOLO Worker error")

    logger.info("YOLO Worker stopped")


def ucf_worker(ucf_model, normal_model, frame_buffer, output_queue, stop_event, logger, beep):
    logger.info("UCF Worker started")
    while not stop_event.is_set():
        try:
            if len(frame_buffer) >= 16:
                frames = list(frame_buffer)
                result = ucf_model(frames)

                while not output_queue.empty():
                    try:
                        output_queue.get_nowait()
                    except queue.Empty:
                        break

                if result == 1:
                    output_queue.put("CRIME DETECTED!")
                    logger.warning("CRIME DETECTED!")
                    if beep:
                        winsound.Beep(500, 1000)
                else:
                    normal_result = normal_model(frames)
                    output_queue.put(f"Activity: {normal_result}")
                    logger.info(f"Normal activity detected: {normal_result}")

            time.sleep(0.2)
        except Exception:
            logger.exception("UCF Worker error")
    logger.info("UCF Worker stopped")


def start_workers():
    if not st.session_state.threads_started:
        st.session_state.stop_event.clear()

        yolo_thread = threading.Thread(
            target=yolo_worker,
            args=(
                yolo_model,
                yolo_input_queue,
                yolo_output_queue,
                st.session_state.stop_event,
                st.session_state.logger
            ),
            daemon=True,
            name="YoloWorker",
        )

        ucf_thread = threading.Thread(
            target=ucf_worker,
            args=(
                ucf_model,
                i3d_model,
                frame_buffer,
                ucf_queue,
                st.session_state.stop_event,
                st.session_state.logger,
                st.session_state.beep
            ),
            daemon=True,
            name="UcfWorker",
        )

        st.session_state.threads = [yolo_thread, ucf_thread]

        for thread in st.session_state.threads:
            thread.start()
            st.session_state.logger.info(f"Thread {thread.name} started")

        st.session_state.threads_started = True


def stop_workers():
    if not st.session_state.threads_started:
        return

    st.session_state.logger.info("Stopping workers...")
    st.session_state.stop_event.set()

    try:
        yolo_input_queue.put(None, timeout=1)
    except queue.Full:
        pass

    if hasattr(st.session_state, "threads"):
        for thread in st.session_state.threads:
            if thread.is_alive():
                thread.join(timeout=2)
                if thread.is_alive():
                    st.session_state.logger.warning(f"{thread.name} did not stop gracefully")

    st.session_state.threads_started = False
    clear_queues()
    st.session_state.logger.info("All workers stopped")


def clear_queues():
    global current_display_frame
    queues_to_clear = [yolo_input_queue, yolo_output_queue, ucf_queue]

    for q in queues_to_clear:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
            except Exception:
                pass

    frame_buffer.clear()
    st.session_state.frame_count = 0

    with frame_lock:
        current_display_frame = None
    st.session_state.logger.debug("Queues and buffers cleared")


def release_camera():
    if st.session_state.video_cap is not None:
        st.session_state.video_cap.release()
        st.session_state.video_cap = None
        st.session_state.logger.info("Camera released")


def cleanup_resources(temp_file=None):
    release_camera()
    cv.destroyAllWindows()
    stop_workers()
    if temp_file and os.path.exists(temp_file):
        try:
            os.unlink(temp_file)
            st.session_state.logger.info(f"Temp file {temp_file} deleted")
        except Exception as e:
            st.session_state.logger.error(f"Error deleting temp file: {e}")


atexit.register(lambda: cleanup_resources())

# --- Streamlit UI ---
st.title("🎥 Real-time Crime Detection System 🚨")

source_type = st.radio("Choose Input Source", ["Use Webcam", "Upload Video File"])

file_name = None
temp_file = None

if source_type == "Upload Video File":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_file = tfile.name
        file_name = temp_file
        st.session_state.logger.info(f"Video uploaded: {file_name}")
else:
    file_name = 0
    st.session_state.logger.info("Using webcam as input source")

col_perf1, col_perf2 = st.columns(2)
with col_perf1:
    st.session_state.frame_skip = st.slider(
        "UCF Frame Skip", 1, 20, 5, help="Process UCF model every N frames"
    )
with col_perf2:
    st.session_state.yolo_skip = st.slider(
        "YOLO Frame Skip", 1, 10, 2, help="Send frame to YOLO every N frames"
    )

maintain_fps = st.checkbox(
    "Maintain Original Video FPS",
    value=True,
    help="Keep original video speed (recommended for uploaded videos)",
)

beep_state = st.checkbox(
    "Beep Sound When Crime detect",
    value=False,
    help="If You Want Add A Beep Sound When Crime detect",
)

st.session_state.beep = beep_state

resolution = st.selectbox(
    "Display Resolution (affects performance)", ["Full", "Half", "Quarter"], index=1
)

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🚀 Start Detection")
with col2:
    stop_button = st.button("⏹️ Stop Detection")

st_frame = st.empty()
status_placeholder = st.empty()
fps_placeholder = st.empty()

if stop_button:
    st.session_state.stop_detection = True
    st.session_state.detection_running = False
    release_camera()
    stop_workers()
    st.warning("🛑 Detection stopped - Camera released")
    st.session_state.logger.info("Stop button pressed")

if start_button and not st.session_state.detection_running:
    if source_type == "Upload Video File" and not file_name:
        st.warning("Please upload a video file.")
        st.session_state.logger.warning("Start pressed without uploading video")
        st.stop()

    st.session_state.frame_count = 0
    st.session_state.stop_detection = False
    st.session_state.detection_running = True
    clear_queues()
    stop_workers()
    start_workers()

    release_camera()
    st.session_state.video_cap = cv.VideoCapture(file_name)
    cap = st.session_state.video_cap

    if source_type == "Use Webcam":
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv.CAP_PROP_FPS, 30)
        target_fps = 30
        is_webcam = True
    else:
        target_fps = cap.get(cv.CAP_PROP_FPS)
        if target_fps <= 0 or target_fps > 120:
            target_fps = 25
        is_webcam = False

    if not cap.isOpened():
        st.error("Cannot open video source.")
        st.session_state.logger.error("Failed to open video source")
        st.session_state.detection_running = False
        cleanup_resources(temp_file)
        st.stop()

    st.success(f"✅ Detection started! Target FPS: {target_fps:.1f}")
    st.session_state.logger.info(f"Detection started, target FPS={target_fps}")

    prev_time = time.time()
    frame_times = deque(maxlen=10)
    yolo_frame_count = 0

    if maintain_fps and not is_webcam:
        frame_interval = 1.0 / target_fps
    else:
        frame_interval = 0

    last_frame_time = time.time()
    scale_factors = {"Full": 1.0, "Half": 0.5, "Quarter": 0.25}
    scale = scale_factors[resolution]

    try:
        while (
            st.session_state.detection_running and not st.session_state.stop_detection
        ):
            ret, frame = cap.read()
            if not ret:
                st.info("📹 End of video reached")
                st.session_state.logger.info("End of video reached")
                break

            st.session_state.frame_count += 1
            yolo_frame_count += 1
            st.session_state.logger.debug(f"Frame {st.session_state.frame_count} captured")

            if scale != 1.0:
                height, width = frame.shape[:2]
                new_width, new_height = int(width * scale), int(height * scale)
                display_frame = cv.resize(frame, (new_width, new_height))
            else:
                display_frame = frame

            if yolo_frame_count >= st.session_state.yolo_skip:
                if not yolo_input_queue.full():
                    try:
                        yolo_input_queue.put(
                            (display_frame.copy(), st.session_state.frame_count),
                            timeout=0.01,
                        )
                        st.session_state.logger.debug(
                            f"Frame {st.session_state.frame_count} sent to YOLO queue"
                        )
                        yolo_frame_count = 0
                    except queue.Full:
                        st.session_state.logger.warning("YOLO input queue full")

            display_this_frame = None
            with frame_lock:
                if current_display_frame is not None:
                    display_this_frame = current_display_frame.copy()

            if display_this_frame is not None:
                rgb_frame = cv.cvtColor(display_this_frame, cv.COLOR_BGR2RGB)
            else:
                rgb_frame = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)

            st_frame.image(rgb_frame, channels="RGB", use_container_width=True)

            if (st.session_state.frame_count % st.session_state.frame_skip) == 0:
                resized_frame = cv.resize(frame, (224, 224))
                frame_buffer.append(resized_frame)
                st.session_state.logger.debug(
                    f"Frame {st.session_state.frame_count} added to buffer for UCF"
                )

            try:
                if not ucf_queue.empty():
                    action = ucf_queue.get_nowait()
                    status_placeholder.markdown(f"### 🔍 Status: {action}")
                    st.session_state.logger.info(f"UCF result: {action}")
            except queue.Empty:
                pass

            curr_time = time.time()
            frame_time = curr_time - prev_time
            frame_times.append(frame_time)

            if len(frame_times) > 0:
                avg_frame_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                fps = 0

            prev_time = curr_time

            fps_placeholder.text(
                f"FPS: {fps:.1f} | Target: {target_fps:.1f} | Frame: {st.session_state.frame_count} | YOLO Queue: {yolo_input_queue.qsize()}"
            )
            st.session_state.logger.debug(f"FPS={fps:.1f}, Frame={st.session_state.frame_count}")

            if maintain_fps and not is_webcam and frame_interval > 0:
                elapsed = time.time() - last_frame_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
            else:
                time.sleep(0.001)

    except Exception:
        st.error("An error occurred")
        st.session_state.logger.exception("Error in main loop")
    finally:
        st.session_state.detection_running = False
        cleanup_resources(temp_file)
        st.success("🛑 Detection stopped and resources cleaned up")
        st.session_state.logger.info("Detection stopped and resources cleaned up")

if st.session_state.detection_running:
    st.info("🔄 Detection is running...")
elif st.session_state.threads_started:
    st.warning("⚠️ Workers are still active. Click Stop to terminate.")

with st.expander("📈 Performance Tips"):
    st.markdown(
        """
    **To maximize FPS:**
    - Use **Quarter** or **Half** resolution for display
    - Increase **YOLO Frame Skip** to 3-5 for faster processing  
    - Increase **UCF Frame Skip** to 10-15 if crime detection can be less frequent
    - For webcam: Uncheck "Maintain Original Video FPS" for maximum speed
    - Close other applications to free up system resources

    **Video Playback:**
    - **Keep "Maintain Original Video FPS" checked** for normal video speed
    - Uncheck it only if you want maximum processing speed (video will play very fast)
    - The app shows both actual FPS and target FPS for comparison

    **Camera Issues:**
    - Click "Stop Detection" to immediately release the camera
    - Camera is automatically released when switching between sources
    """
    )
