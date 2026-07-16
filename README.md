# Facial Emotion Detection App

A real-time facial emotion recognition system that detects faces from a webcam feed or uploaded image and classifies the emotion expressed (happy, sad, angry, surprised, neutral, etc.). Built as part of the Route AI/ML final projects, the system combines a face detection pipeline with a deep learning classifier and is deployed through an interactive Streamlit interface.

## Features

- Real-time face detection from webcam feed
- Emotion classification into multiple categories (happy, sad, angry, surprised, neutral, fear, disgust)
- Support for both webcam input and image upload
- Displays predicted emotion with confidence score
- Bounding box overlay on detected faces

## Dataset

- **FER2013** (Facial Expression Recognition 2013) — grayscale 48x48 pixel face images labeled with 7 emotion categories.
- Source: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Tech Stack

- **Face Detection:** OpenCV Haar Cascades / MediaPipe Face Detection
- **Emotion Classification:** CNN model trained with TensorFlow/Keras
- **Interface:** Streamlit
- **Other libraries:** NumPy, Pandas, Matplotlib, scikit-learn

## Project Structure

```
project_root/
├── app.py
├── model/
│   ├── __init__.py
│   ├── predict.py
│   └── preprocess.py
├── notebooks/
│   └── emotion_detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/esraahussie/Emotion-Detection-Face-.git
   cd Emotion-Detection-Face-
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open the app**
   The app will open automatically in your browser at `http://localhost:8501`. Allow camera access when prompted to use the real-time detection feature.

## Model Training

The full training pipeline — data loading, preprocessing, model building, training, and evaluation — is available in `notebooks/emotion_detection.ipynb`. The exported model is saved in the `model/` directory and loaded by the Streamlit app for inference.

## Evaluation

The model was evaluated using accuracy, F1-score, and a confusion matrix to account for class imbalance in the dataset (see notebook for full results and discussion).

## Challenges

- Handling class imbalance in FER2013 (e.g., "happy" and "neutral" being overrepresented)
- Real-time performance under varying lighting conditions and head poses
- Robust face detection across different face sizes and angles
