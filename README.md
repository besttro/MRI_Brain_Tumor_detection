# 🧠 Brain Tumor Detection (YOLOv8 + Streamlit)

A deep-learning web application for brain tumor detection using MRI scans.  
Built with **YOLOv8**, **Streamlit**, and **Python**, featuring a modern neon-dark UI and real-time detection.

---

## 🚀 Features
- Brain tumor detection using **YOLOv8**
- Upload MRI images (JPG, JPEG, PNG)
- Real-time inference with bounding boxes
- Detailed confidence metrics & detection report
- Custom dark theme with glowing neural effects
- Fully supports **Python venv**

---

## 📦 Installation Guide (Using venv)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Create & activate virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

---

## ▶️ Run the Application
Make sure your venv is active, then run:

```bash
streamlit run app.py
```

Open in your browser:

```
http://localhost:8501
```

---

## 📁 Project Structure
```
📦 project/
 ┣ 📂 YOLOv8_Results/
 │   ┗ 📂 mri_tumor/
 │        ┗ 📂 weights/
 │             ┗ best.pt
 ┣ app.py
 ┣ requirements.txt
 ┗ README.md
```

Ensure the YOLO model path is correct:
```python
MODEL_PATH = 'YOLOv8_Results/mri_tumor/weights/best.pt'
```

---

## 🧠 How It Works
1. User uploads an MRI scan  
2. Streamlit loads the image and stores it temporarily  
3. YOLOv8 performs inference  
4. The result image with bounding boxes is returned  
5. Confidence scores + class names + bounding boxes displayed  
6. Data table of detections generated using Pandas  

---

## 🔧 Requirements
Core dependencies used in this project:

```
ultralytics
streamlit
pillow
pandas
opencv-python
numpy
```

---

## 📸 Tech Stack
- Python 3.9+
- Streamlit
- Ultralytics YOLOv8
- Pillow / OpenCV
- Pandas

---
