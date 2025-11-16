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
git clone https://github.com/besttro/MRI_Brain_Tumor_detection.git
cd MRI_Brain_Tumor_detection
```

### 2️⃣ Create and activate virtual environment

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

### 3️⃣ Install required packages  
(Extracted directly from the project code)

```bash
pip install streamlit ultralytics pillow pandas
```

*Note:*  
`opencv-python` will be installed automatically by ultralytics when needed.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

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
 │             ┗ best.pt       # Your YOLOv8 trained model
 ┣ app.py                      # Main Streamlit UI + YOLO code
 ┗ README.md
```

Ensure the model path in `app.py` matches:

```python
MODEL_PATH = 'YOLOv8_Results/mri_tumor/weights/best.pt'
```

---

## 🧠 How It Works

1. User uploads an MRI image  
2. Streamlit loads and stores it temporarily  
3. YOLOv8 performs tumor detection  
4. Bounding boxes + confidence scores are drawn  
5. A detailed detection table is generated  
6. Custom CSS provides a neon-dark interface  

---

## 🔧 Dependencies Used in This Project  
(Directly from code in `app.py`)

```
streamlit
ultralytics
pillow
pandas
```

Optional (auto-installed by ultralytics):

```
opencv-python
numpy
```

---

## 📸 Tech Stack

- Python 3.9+  
- Streamlit  
- Ultralytics YOLOv8  
- Pillow  
- Pandas  

---
