# Smart Surveillance System

This project focuses on developing an intelligent security system designed for environments such as malls, hospitals, schools, and offices. The system automatically monitors surroundings and detects unsafe or suspicious human activities, enhancing safety and security.

## Abstraction

The Smart Surveillance System is an AI-powered application developed to enhance safety in public spaces. It helps security teams by providing intelligent monitoring tools, enabling them to quickly identify and manage potential threats or suspicious activities.

## Introduction

The Smart Surveillance System is an AI-powered deep learning application designed to enhance security monitoring. It leverages an I3D model to detect and classify human actions and a YOLO model to identify and label objects in real time. The I3D model is fine-tuned using the UCF-Crime dataset to recognize suspicious or unsafe activities. To ensure accurate tracking, the system integrates the DeepSORT algorithm, which assigns unique IDs to each detected object. Additionally, threading techniques are implemented to optimize performance and allow multiple model instances to run efficiently.

## Dataset

The Smart Surveillance System uses the UCF-Crime dataset, a large-scale collection of 128 hours of real-world surveillance footage. The dataset includes 1,900 long and untrimmed videos featuring 13 categories of realistic anomalies: Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism. These categories were chosen for their relevance and potential impact on public safety, making the dataset ideal for training action recognition models.

The dataset link [UCF Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)

## Methodology

We followed a systematic process, beginning with thorough data analysis, followed by data preprocessing and model architecture design. We then moved to the training phase, including model evaluation, and finally developed the inference and system integration stages.

### Build Dataset 

The dataset architecture for this project is divided into three main parts:

1. **Frame Dataset**  Reads videos frame by frame from labeled directories, extracting both the frame label and index. Each frame is assigned a part number for organized retrieval. We apply several data augmentation techniques such as ```RandomHorizontalFlip```, ```RandomRotation```, ```RandomCro```, ```ColorJitter```, and ```Normalize``` to enhance model generalization.

2. **Video Dataset**  Combines frames to reconstruct full video sequences. During retrieval, videos are either padded or truncated to a fixed number of frames for consistent training. Truncation is handled using the ```linspace``` method to evenly sample frames across the video, minimizing information loss.

3. **Encoded Video Dataset**  Prepares the final dataset for training by converting labels into binary values: 0 (no crime) and 1 (crime). This encoding ensures the model can effectively distinguish between normal and abnormal activities.

### Build Model

In the model-building stage, we leveraged a pre-trained I3D model and fine-tuned it using two strategies:

1. **Block-level tuning**  Adjusting and retraining groups of layers (blocks) to adapt the model to the new dataset.

2. **Layer-level tuning**  Fine-tuning specific layers for more granular control over feature learning.

The final classification layer of the I3D model was removed and replaced with a custom output layer tailored to our binary classification task: predicting whether an activity represents a crime (1) or non-crime (0).

### Inference

The inference stage consists of three core components:

1. **Object Detection and Tracking**  Utilizes a YOLO model combined with the DeepSORT algorithm to analyze individual frames, draw bounding boxes around detected objects, and assign unique IDs for continuous tracking.

2. **Action Recognition (UCF Inference)**  Loads the fine-tuned I3D model (trained on the UCF-Crime dataset and hosted on Hugging Face) to analyze sequences of frames and identify suspicious or criminal actions.

3. **Normal Activity Detection (I3D Original)**  Uses the pre-trained original I3D model to classify and monitor normal activities, providing a baseline for comparison.

### Interface and Pipeline

In the final stage, we developed a user-friendly interface using the Streamlit library and incorporated threading to enhance performance. The interface enables users to:

- Select a live camera feed or a pre-recorded video from their device.

- Adjust video quality for optimal processing.

- Receive real-time alerts (such as a beep) when a suspicious activity or crime is detected.

- Automatically save all detection logs in the ```logs``` folder for monitoring and record-keeping.

## Results

We evaluated the Smart Surveillance System using popular metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. During the training stage, we observed stable performance across all experiments. We focused on varying the **learning rate** and training strategy (block-level vs. layer-level fine-tuning) to optimize the model.

### Evaluation Metrics

| Model Configuration                    | Accuracy | Precision | Recall  | F1-score |
|---------------------------------------|----------|-----------|---------|----------|
| UCF-I3D model (by 4 blocks, lr=0.0001) | 0.7931   | 0.79499   | 0.7931  | 0.7931   |
| UCF-I3D model (by block, lr=0.01)     | 0.67931  | 0.73241   | 0.67931 | 0.66416  |
| UCF-I3D model (by 3 blocks, lr=0.001) | 0.74138  | 0.74781   | 0.74138 | 0.74067  |
| UCF-I3D model (by layer, lr=0.001)    | 0.81724  | 0.82449   | 0.81724 | 0.81557  |
| UCF-I3D model (by block, lr=0.001)    | 0.81379  | 0.81697   | 0.81379 | 0.81286  |

### Insights
- Layer-level fine-tuning achieved the highest performance, with an F1-score of **0.81557**.  
- Block-level tuning with appropriate learning rates also performed well, demonstrating that partial fine-tuning can be effective.  
- Higher learning rates (e.g., 0.01) negatively impacted performance, indicating the model benefits from careful tuning.

## How to Run Locally

1. Clone the repository:

```
git clone https://github.com/amjadAwad95/smart-surveillance-system.git
cd smart-surveillance-system
```

2. Create a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate
```

3. Install the dependencies:

```
pip install -r requirements.txt
```

4. Run the app

```
streamlit run main.py
```

## Conclusion

The Smart Surveillance System demonstrates the effective integration of AI-powered models for real-time security monitoring. By combining I3D-based action recognition with YOLO object detection and DeepSORT tracking, the system can accurately identify both normal and suspicious human activities in public spaces such as malls, hospitals, schools, and offices.  

Our experiments showed that fine-tuning strategies, particularly layer-level tuning, significantly improve model performance, achieving high accuracy and F1-scores. The system also provides a user-friendly interface through Streamlit, enabling live camera or video input, adjustable video quality, real-time alerts, and comprehensive logging.  

Overall, this project highlights the potential of deep learning in enhancing public safety, providing security personnel with reliable tools for monitoring and responding to suspicious or unsafe activities efficiently. Future work can focus on expanding the dataset, improving detection of complex multi-person interactions, and optimizing inference speed for large-scale deployments.

