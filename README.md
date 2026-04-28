🧠 Stress Detection using Thermal Imaging & Physiological Signals


🚀 Overview

This project implements a contactless stress detection system using facial thermal imaging and physiological signals. It analyzes how the human body reacts during baseline, stress, and recovery phases and predicts a stress risk score using deep learning.

✨ Key Features

✔️ Contactless detection (no wearable devices)


✔️ Multimodal analysis (thermal + physiological signals)


✔️ CNN-LSTM model for spatial + temporal learning


✔️ Explainable AI (SHAP & Attention) for interpretability


✔️ Generates a stress risk score (0–1)


🛠️ Tech Stack
1.Python


2.OpenCV – Image processing


3.NumPy – Data handling


4.TensorFlow / PyTorch – Deep learning


5.Matplotlib – Visualization


✨ How It Works
1.Capture facial thermal video


2.Detect and segment facial regions


3.Extract:
   Temperature features
   Heart rate
   Respiration patterns


4.Combine all features


5.Pass data to CNN-LSTM model


6.Apply Explainable AI


7.Generate final stress score

🧪 Experimental Phases


1.Baseline Phase → Normal state


2.Stress Phase → Stress induced


3.Recovery Phase → Return to normal



📈 Applications


Healthcare monitoring 🏥


Mental stress analysis 🧘


Clinical screening 🧑‍⚕️


Research & AI applications 🤖



