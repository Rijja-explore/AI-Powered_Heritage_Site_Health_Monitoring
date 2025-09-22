# 🏛️ AI-Powered Heritage Site Health Monitoring

🚧 **Automated Crack Detection and Depth Estimation**  
An AI-powered system for real-time crack detection, segmentation, and depth analysis using computer vision and deep learning techniques. Designed for infrastructure monitoring via webcam, CCTV, or drone feeds.

---

## 📖 Overview

Traditional infrastructure inspections are manual, time-consuming, and prone to human error. This project automates the process using:

- Deep Learning (YOLOv8)
- Image Processing (e.g., Canny Edge Detection)
- Interactive UI (Streamlit or OpenCV)

🔍 Key Goals:
- Early crack detection  
- Depth severity estimation  
- Accurate edge localization  
- Real-time monitoring  

---

## 🚀 Features

- 📷 Real-time crack detection via webcam
- 🧠 Crack segmentation using YOLOv8
- ✂️ Edge localization with Canny Edge Detection
- 📏 Depth estimation based on image heuristics
- 🌐 Streamlit web interface
- 🖥️ Local OpenCV UI for resource-constrained setups
- 🔋 Lightweight, scalable, and modular

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries                       |
|----------------|------------------------------------------|
| **Language**    | Python                                  |
| **Interface**   | Streamlit, OpenCV                       |
| **Deep Learning** | YOLOv8 (via Ultralytics)               |
| **Image Processing** | Canny Edge Detection, Pillow        |
| **Visualization** | Plotly, OpenCV                        |
| **Models**      | YOLOv8n (Detection + Segmentation)      |

---

## 📦 Libraries Used

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

```text
project-root/
├── finalwebapp.py # Main Streamlit web app
├── segmentation_model/ # YOLOv8 segmentation weights
├── runs/detect/train3/weights/ # Crack detection YOLOv8 weights
├── pdf_report.py # (Optional) PDF export helper
├── assets/ # Icons or visuals
├── README.md # You are here
```


---

## 🧪 How to Run

### 🔸 Option 1: Web Interface (Streamlit)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the web app
streamlit run finalwebapp.py
```

🔸 Option 2: OpenCV Interface (Optional)
For direct webcam feed using OpenCV:
```bash
python camera_capture.py
```

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

## Screenshots
![Screenshot 2025-06-28 114730](https://github.com/user-attachments/assets/95acf337-b9bf-4648-8c08-cb636d907071)
![Screenshot 2025-06-28 114938](https://github.com/user-attachments/assets/0abe1ca2-b246-40f7-804f-15becdd8bacf)
![Screenshot 2025-06-28 115027](https://github.com/user-attachments/assets/21a8b12e-e853-42dc-bede-2704e7105070)
![Screenshot 2025-06-28 115054](https://github.com/user-attachments/assets/474d94fe-3d87-44ea-b451-f3262530df01)
![Screenshot 2025-06-28 115115](https://github.com/user-attachments/assets/c330d3d5-8903-46d7-8db6-ffe04006fe18)
---

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

