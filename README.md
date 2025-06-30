# AI-Powered_Heritage_Site_Health_Monitoring

# 🚧 Automated Crack Detection and Depth Estimation

An AI-powered system for real-time crack detection, segmentation, and depth analysis using computer vision and deep learning techniques. Ideal for infrastructure monitoring through webcams, CCTV, or drone feeds.

---

## 📖 Overview

Traditional infrastructure inspections are manual, time-consuming, and prone to error. This project automates the process using deep learning (YOLOv8) and image processing techniques like Canny Edge Detection. By integrating these into an interactive UI (Streamlit or OpenCV), this system enables:

- Early crack detection
- Depth severity estimation
- Accurate edge localization
- Real-time monitoring

---

## 🚀 Features

- 📷 Real-time crack detection via webcam
- 🧠 Crack segmentation using **YOLOv8**
- ✂️ Edge localization with **Canny Edge Detection**
- 📏 Depth estimation for severity
- 🌐 Web app interface using **Streamlit**
- 🖥️ Local OpenCV UI for low-resource environments
- 🔋 Lightweight, scalable, and easily integrable

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries                       |
|----------------|------------------------------------------|
| Language        | Python                                   |
| Interface       | Streamlit, OpenCV                        |
| Deep Learning   | YOLOv8 (via Ultralytics)                 |
| Image Processing| Canny Edge Detection, Pillow             |
| Visualization   | Plotly, OpenCV                           |
| Models          | YOLOv8n for detection and segmentation   |

### 📦 Libraries Used

- `torch`
- `opencv-python`
- `ultralytics`
- `numpy`
- `Pillow`
- `streamlit`
- `plotly`
- `scikit-learn`

---

## 📁 Project Structure

```
project-root/
├── finalwebapp.py                # Main Streamlit web app
├── segmentation_model/          # YOLOv8 segmentation weights
├── runs/detect/train3/weights/  # Crack detection YOLOv8 weights
├── pdf_report.py                # (Optional) PDF export helper
├── assets/                      # Icons or visuals
├── README.md                    # You are here
```

---

## 🧪 How to Run

### 🔸 Option 1: Run with Streamlit (Web Interface)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run finalwebapp.py
```

### 🔸 Option 2: Run Real-Time OpenCV Interface (if available)

Can be integrated using `cv2.VideoCapture` in Python.

---

## 🧠 YOLOv8 Models

| Model           | Purpose             | Path                               |
|----------------|---------------------|------------------------------------|
| `best.pt`       | Crack detection      | `runs/detect/train3/weights/best.pt` |
| `best.pt` (seg) | Crack segmentation   | `segmentation_model/weights/best.pt` |
| `yolov8n.pt`    | Default fallback     | via `ultralytics` if custom model is missing |

---

## 📊 Functional Highlights

| Module              | Capability |
|---------------------|-----------|
| Crack Detection     | YOLOv8 object detection |
| Segmentation        | YOLOv8 segmentation or placeholder Canny edge |
| Edge Detection      | Canny edge refinement |
| Depth Estimation    | Image brightness & shadow-based heuristic |
| Biological Growth   | HSV + contour detection (if enabled) |
| Material Analysis   | MobileNetV2 + Rule-based fallback |
| Visualizations      | Crack severity pie chart, growth area bar chart, depth heatmaps, and more |

---

## 🔍 Sample Output


![WhatsApp Image 2025-06-30 at 15 20 41_64fe063f](https://github.com/user-attachments/assets/d2a614e6-d8e4-44d4-b369-e8e8234f36a1)
![WhatsApp Image 2025-06-30 at 15 20 41_e24551c2](https://github.com/user-attachments/assets/a7696767-3aa8-43d1-97c7-8386d5c85921)
![WhatsApp Image 2025-06-30 at 15 20 41_e0366bca](https://github.com/user-attachments/assets/413d270c-2007-4b9b-a5fe-bc095039118f)
![WhatsApp Image 2025-06-30 at 15 20 41_31c6a637](https://github.com/user-attachments/assets/0b6c8861-dd1b-4896-8fce-4023aee89140)
![WhatsApp Image 2025-06-30 at 15 20 42_98ad61ea](https://github.com/user-attachments/assets/4f293f1b-7bb0-4e78-88f8-fa6c67eb2c06)

---

## 📦 Dependencies

Ensure Python 3.8+ is installed. Then install:

```bash
pip install torch torchvision
pip install opencv-python ultralytics numpy Pillow streamlit scikit-learn plotly
```

---

## 👥 Contributors

- **Rijja H**
- **Rohith Varshighan S**
- **Nikhil S**

---

## 📜 License

This project is licensed under the **MIT License**. Free to use and modify.

---

## 🌐 Future Improvements

- Drone/CCTV live integration
- Multi-class damage analysis (spalling, corrosion, etc.)
- Export to PDF (via `pdf_report.py`)
- Alert system for critical cracks
