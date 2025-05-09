# 😊 Facial Expression Recognition (FER)

A real-time facial expression recognition system built using **TensorFlow**, **OpenCV**, and **Streamlit**.  
This project detects human emotions (angry, disgust, fear, happy, neutral, sad, surprise) from facial images or webcam video input.

---

## 🚀 Features

- 🎥 Real-time emotion recognition via webcam
- 🔍 Uses Haar cascades for face detection
- 🧠 TensorFlow CNN model trained on FER-2013
- 📈 Evaluation with precision, recall, F1-score, confusion matrix
- 💻 Streamlit web app interface for instant use
- ✅ Modular project structure (data, models, notebooks, app)

---

## 🧠 Emotions Detected

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprise 😲

---

## 🗂️ Project Structure

Facial-Expression-Recognition/
├── models/
│ └── expression_model_*.h5 # Trained model
├── data/
│ ├── test # Test data
│ ├── train # Train data
│ ├── class_weights.pkl # Class Weights
│ ├── class_names.pkl # Class label mappings
│ └── *.npy # Preprocessed data
├── notebooks/
│ ├── 1_data_preprocessing.ipynb
│ ├── 2_train_model.ipynb
│ ├── 3_evaluate_model.ipynb
│ ├── streamlit_fer_app.ipynb
├── streamlit_fer_app.py # Real-time app using Streamlit
├── README.md

---

## 📦 Installation

 **Clone the repo**
   ```bash
   git clone https://github.com/your-username/Facial-Expression-Recognition.git
   cd Facial-Expression-Recognition

## 🔧 Running the Real-Time App

streamlit run streamlit_fer_webcam.py

📊 Model Evaluation

    Model: Custom CNN (Conv2D + BatchNorm + Dropout)

    Dataset: FER-2013 Facial Expression Dataset

    Test Accuracy: ~58%

    Val Accuracy: ~55%

    Best Precision: Happy, Surprise

    Needs Improvement: Fear, Disgust

🛠️ Future Improvements

    Transfer learning with MobileNetV2 or EfficientNet

    Better augmentation for low-recall emotions

    Web app version with Flask/React

    Streamlit Cloud or Hugging Face Spaces deployment

🧑‍💻 Author

Rohit
Data Science & AI Enthusiast
