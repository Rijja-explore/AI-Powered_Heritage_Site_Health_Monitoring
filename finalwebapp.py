# # import streamlit as st
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import pandas as pd
# # from ultralytics import YOLO
# # from sklearn.linear_model import LinearRegression
# # import os
# # import torch
# # import torch.nn as nn
# # import torchvision.models as models
# # import torchvision.transforms as transforms
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from datetime import datetime, timedelta
# # import warnings
# # try:
# #     from scipy import ndimage
# #     from skimage import measure
# #     SCIPY_SKIMAGE_AVAILABLE = True
# # except ImportError:
# #     SCIPY_SKIMAGE_AVAILABLE = False
# #     st.warning("SciPy or scikit-image not installed. Some advanced features may be limited.")
# # warnings.filterwarnings('ignore')

# # # Set page config as the FIRST Streamlit command
# # st.set_page_config(page_title="Heritage Sites Health Monitoring", layout="wide")

# # # Initialize session state for storing analysis results
# # if 'analysis_results' not in st.session_state:
# #     st.session_state.analysis_results = None
# # if 'image_name' not in st.session_state:
# #     st.session_state.image_name = None
# # if 'image_np' not in st.session_state:
# #     st.session_state.image_np = None

# # # Model loading with better error handling and fallbacks
# # @st.cache_resource
# # def load_models():
# #     models_status = {}
# #     try:
# #         yolo_path = "runs/detect/train3/weights/best.pt"
# #         if os.path.exists(yolo_path):
# #             yolo_model = YOLO(yolo_path)
# #             models_status['yolo'] = f"Custom model loaded from {yolo_path}"
# #         else:
# #             yolo_model = YOLO("yolov8n.pt")
# #             models_status['yolo'] = "Using default YOLOv8n model"

# #         seg_path = "./segmentation_model/weights/best.pt"
# #         if os.path.exists(seg_path):
# #             segmentation_model = YOLO(seg_path)
# #             models_status['segmentation'] = f"Custom segmentation model loaded from {seg_path}"
# #         else:
# #             segmentation_model = YOLO("yolov8n-seg.pt")
# #             models_status['segmentation'] = "Using default YOLOv8n-seg model"

# #         material_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
# #         material_model.classifier = nn.Sequential(
# #             nn.Dropout(0.2),
# #             nn.Linear(material_model.last_channel, 8)
# #         )
# #         material_model.eval()
# #         models_status['material'] = "MobileNetV2 model loaded with custom classifier for 8 material types"

# #         st.success("✅ All models loaded successfully!")
# #         with st.expander("Model Loading Details"):
# #             for model_type, status in models_status.items():
# #                 st.info(f"{model_type.capitalize()}: {status}")

# #         return yolo_model, segmentation_model, material_model
# #     except Exception as e:
# #         st.error(f"❌ Model loading failed: {str(e)}")
# #         st.warning("⚠ Some features may not work properly without models.")
# #         return None, None, None

# # yolo_model, segmentation_model, material_model = load_models()

# # # Image processing functions
# # def load_and_preprocess_image(uploaded_file):
# #     try:
# #         image = Image.open(uploaded_file).convert('RGB')
# #         image_np = np.array(image)
# #         if image_np.size == 0:
# #             raise ValueError("Invalid image file: The uploaded image appears to be empty.")
# #         return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
# #     except Exception as e:
# #         st.error(f"❌ Error loading or preprocessing the image: {str(e)}")
# #         return None

# # def calculate_severity(width_cm, length_cm, label):
# #     try:
# #         if 'crack' not in label.lower():
# #             return None
# #         area = width_cm * length_cm
# #         max_dimension = max(width_cm, length_cm)
# #         if max_dimension < 0.5 and area < 0.25:
# #             return 'Minor'
# #         elif max_dimension < 1.5 and area < 2.0:
# #             return 'Moderate'
# #         elif max_dimension < 3.0 and area < 6.0:
# #             return 'Severe'
# #         else:
# #             return 'Critical'
# #     except Exception as e:
# #         st.error(f"❌ Severity calculation error: {str(e)}")
# #         return 'Unknown'

# # def detect_with_yolo(image_np, px_to_cm_ratio=0.1):
# #     try:
# #         if yolo_model is None:
# #             st.warning("⚠ YOLO model is not loaded. Using placeholder detection.")
# #             height, width = image_np.shape[:2]
# #             placeholder_detection = {
# #                 'width_cm': 2.5,
# #                 'length_cm': 3.0,
# #                 'severity': 'Moderate',
# #                 'confidence': 0.85,
# #                 'label': 'crack',
# #                 'bbox': (width//4, height//4, 3*width//4, 3*height//4)
# #             }
# #             annotated_image = image_np.copy()
# #             x1, y1, x2, y2 = placeholder_detection['bbox']
# #             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(annotated_image, f"Placeholder: crack (2.5cm x 3.0cm) - Moderate",
# #                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
# #             return annotated_image, [placeholder_detection]

# #         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
# #         results = yolo_model.predict(image_rgb, conf=0.3)
# #         crack_details = []
# #         annotated_image = image_np.copy()

# #         for result in results:
# #             if result.boxes is not None and len(result.boxes) > 0:
# #                 for box in result.boxes:
# #                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
# #                     width_px = x2 - x1
# #                     length_px = y2 - y1
# #                     width_cm = width_px * px_to_cm_ratio
# #                     length_cm = length_px * px_to_cm_ratio

# #                     class_id = int(box.cls[0].cpu().numpy())
# #                     label = yolo_model.names.get(class_id, "unknown")
# #                     confidence = float(box.conf[0].cpu().numpy())
# #                     severity = calculate_severity(width_cm, length_cm, label)

# #                     crack_details.append({
# #                         'width_cm': width_cm,
# #                         'length_cm': length_cm,
# #                         'severity': severity,
# #                         'confidence': confidence,
# #                         'label': label,
# #                         'bbox': (x1, y1, x2, y2)
# #                     })

# #                     color = {
# #                         'Minor': (0, 255, 0),
# #                         'Moderate': (0, 255, 255),
# #                         'Severe': (0, 165, 255),
# #                         'Critical': (255, 0, 0)
# #                     }.get(severity, (128, 128, 128))

# #                     cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
# #                     severity_text = f" - {severity}" if severity else ""
# #                     display_text = f"{label}: {width_cm:.2f}cm x {length_cm:.2f}cm{severity_text} ({confidence:.2f})"
# #                     text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
# #                     cv2.rectangle(annotated_image, (x1, y1-25), (x1 + text_size[0], y1), (0, 0, 0), -1)
# #                     cv2.putText(annotated_image, display_text, (x1, y1 - 10),
# #                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# #         if not crack_details:
# #             st.info("ℹ No objects detected by YOLO.")
# #         return annotated_image, crack_details
# #     except Exception as e:
# #         st.error(f"❌ YOLO detection failed: {str(e)}")
# #         return image_np, []

# # def detect_biological_growth_advanced(image_np):
# #     try:
# #         growth_image = image_np.copy()
# #         hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
# #         lower_green1 = np.array([35, 40, 40])
# #         upper_green1 = np.array([85, 255, 255])
# #         lower_green2 = np.array([25, 30, 20])
# #         upper_green2 = np.array([95, 200, 150])
# #         mask_green1 = cv2.inRange(hsv, lower_green1, upper_green1)
# #         mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
# #         combined_mask = cv2.bitwise_or(mask_green1, mask_green2)
# #         kernel = np.ones((5, 5), np.uint8)
# #         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
# #         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
# #         contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         growth_detected = False
# #         total_growth_area = 0
# #         for contour in contours:
# #             area = cv2.contourArea(contour)
# #             if area > 100:
# #                 growth_detected = True
# #                 cv2.drawContours(growth_image, [contour], -1, (0, 0, 255), 2)
# #                 x, y, w, h = cv2.boundingRect(contour)
# #                 cv2.putText(growth_image, f"Growth: {area:.0f}px",
# #                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
# #                 total_growth_area += area
# #         if not growth_detected:
# #             cv2.putText(growth_image, "No biological growth detected",
# #                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #         else:
# #             cv2.putText(growth_image, f"Total growth area: {total_growth_area:.0f} pixels",
# #                        (50, image_np.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# #         return growth_image, growth_detected, total_growth_area
# #     except Exception as e:
# #         st.error(f"❌ Biological growth detection failed: {str(e)}")
# #         return image_np, False, 0

# # def detect_biological_growth(image_np, crack_details):
# #     try:
# #         growth_image, growth_detected_advanced, growth_area_px = detect_biological_growth_advanced(image_np)
# #         yolo_growth_detected = False
# #         for crack in crack_details:
# #             if any(keyword in crack['label'].lower() for keyword in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
# #                 x1, y1, x2, y2 = crack['bbox']
# #                 width_cm = crack['width_cm']
# #                 length_cm = crack['length_cm']
# #                 confidence = crack['confidence']
# #                 cv2.rectangle(growth_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
# #                 cv2.putText(growth_image, f"YOLO Growth: {width_cm:.2f}cm x {length_cm:.2f}cm ({confidence:.2f})",
# #                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
# #                 yolo_growth_detected = True
# #         overall_growth_detected = growth_detected_advanced or yolo_growth_detected
# #         if overall_growth_detected:
# #             cv2.putText(growth_image, "BIOLOGICAL GROWTH DETECTED",
# #                        (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
# #         return growth_image
# #     except Exception as e:
# #         st.error(f"❌ Enhanced biological growth detection failed: {str(e)}")
# #         return image_np

# # def calculate_biological_growth_area(crack_details, seg_results, image_np, px_to_cm_ratio):
# #     try:
# #         total_area_cm2 = 0
# #         areas_counted = set()
# #         if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
# #             masks = seg_results[0].masks.data.cpu().numpy()
# #             image_height, image_width = image_np.shape[:2]
# #             for mask in masks:
# #                 resized_mask = cv2.resize(mask.astype(np.uint8), (image_width, image_height),
# #                                         interpolation=cv2.INTER_NEAREST)
# #                 mask_area_px = np.sum(resized_mask)
# #                 mask_area_cm2 = mask_area_px * (px_to_cm_ratio ** 2)
# #                 if mask_area_cm2 not in areas_counted:
# #                     total_area_cm2 += mask_area_cm2
# #                     areas_counted.add(mask_area_cm2)
# #         else:
# #             for crack in crack_details:
# #                 if any(keyword in crack['label'].lower() for keyword in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
# #                     area = crack['width_cm'] * crack['length_cm']
# #                     if area not in areas_counted:
# #                         total_area_cm2 += area
# #                         areas_counted.add(area)
# #             _, growth_detected, growth_area_px = detect_biological_growth_advanced(image_np)
# #             if growth_detected and growth_area_px > 0:
# #                 growth_area_cm2 = growth_area_px * (px_to_cm_ratio ** 2)
# #                 if growth_area_cm2 not in areas_counted:
# #                     total_area_cm2 += growth_area_cm2
# #                     areas_counted.add(growth_area_cm2)
# #         return total_area_cm2
# #     except Exception as e:
# #         st.error(f"❌ Biological growth area calculation failed: {str(e)}")
# #         return 0

# # def segment_image(image_np):
# #     try:
# #         if segmentation_model is None:
# #             st.warning("⚠ Segmentation model is not loaded. Creating placeholder segmentation.")
# #             gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
# #             edges = cv2.Canny(gray, 100, 200)
# #             segmented_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# #             cv2.putText(segmented_image, "Placeholder Segmentation",
# #                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
# #             return segmented_image, None
# #         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
# #         results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
# #         if results and len(results) > 0:
# #             segmented_image = results[0].plot()
# #             return segmented_image, results
# #         else:
# #             st.info("ℹ No segments detected in the image.")
# #             return image_np, None
# #     except Exception as e:
# #         st.error(f"❌ Segmentation failed: {str(e)}")
# #         return image_np, None

# # def preprocess_image_for_depth_estimation(image_np):
# #     try:
# #         gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
# #         blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# #         return cv2.equalizeHist(blurred_image)
# #     except Exception as e:
# #         st.error(f"❌ Depth preprocessing failed: {str(e)}")
# #         return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# # def create_depth_estimation_heatmap(equalized_image):
# #     try:
# #         _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
# #         shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)
# #         depth_estimation = 255 - shadow_region
# #         depth_estimation_normalized = cv2.normalize(depth_estimation, None, 0, 255, cv2.NORM_MINMAX)
# #         return cv2.applyColorMap(depth_estimation_normalized.astype(np.uint8), cv2.COLORMAP_JET)
# #     except Exception as e:
# #         st.error(f"❌ Depth heatmap creation failed: {str(e)}")
# #         return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# # def apply_canny_edge_detection(image_np):
# #     try:
# #         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
# #         edges = cv2.Canny(gray, 100, 200)
# #         return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# #     except Exception as e:
# #         st.error(f"❌ Edge detection failed: {str(e)}")
# #         return image_np

# # # def classify_material(image_np):
# # #     try:
# # #         if material_model is None:
# # #             st.warning("⚠ Material classification model not loaded. Using texture-based classification.")
# # #             return classify_material_fallback(image_np)
# # #         transform = transforms.Compose([
# # #             transforms.ToPILImage(),
# # #             transforms.Resize((224, 224)),
# # #             transforms.ToTensor(),
# # #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # #         ])
# # #         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
# # #         image_tensor = transform(image_rgb).unsqueeze(0)
# # #         with torch.no_grad():
# # #             output = material_model(image_tensor)
# # #             probabilities = torch.softmax(output, dim=1)
# # #             _, predicted = torch.max(output, 1)
# # #         material_classes = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']
# # #         predicted_material = material_classes[predicted.item()]
# # #         probs = probabilities[0].cpu().numpy()
# # #         return predicted_material, probs
# # #     except Exception as e:
# # #         st.error(f"❌ Material classification failed: {str(e)}")
# # #         return classify_material_fallback(image_np)

# # # def classify_material_fallback(image_np):
# # #     try:
# # #         hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
# # #         mean_hue = np.mean(hsv[:, :, 0])
# # #         mean_saturation = np.mean(hsv[:, :, 1])
# # #         mean_value = np.mean(hsv[:, :, 2])
# # #         std_value = np.std(hsv[:, :, 2])
# # #         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
# # #         texture_measure = np.std(gray)
# # #         mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))
# # #         if mean_r > mean_g > mean_b and mean_saturation > 80:
# # #             material = 'Brick'
# # #             probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
# # #         elif texture_measure > 60 and mean_value < 120:
# # #             if mean_value < 80:
# # #                 material = 'Stone'
# # #                 probs = np.array([0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])
# # #             else:
# # #                 material = 'Sandstone'
# # #                 probs = np.array([0.2, 0.05, 0.05, 0.05, 0.02, 0.01, 0.1, 0.6])
# # #         elif mean_value > 180 and std_value < 30:
# # #             if mean_saturation < 20:
# # #                 if texture_measure < 20:
# # #                     material = 'Marble'
# # #                     probs = np.array([0.05, 0.05, 0.1, 0.05, 0.02, 0.01, 0.7, 0.02])
# # #                 else:
# # #                     material = 'Plaster'
# # #                     probs = np.array([0.05, 0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.01])
# # #             else:
# # #                 material = 'Concrete'
# # #                 probs = np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
# # #         elif mean_hue > 10 and mean_hue < 30 and mean_saturation > 50:
# # #             material = 'Wood'
# # #             probs = np.array([0.05, 0.1, 0.05, 0.05, 0.7, 0.02, 0.02, 0.01])
# # #         elif mean_value > 150 and texture_measure > 40:
# # #             if mean_saturation < 30:
# # #                 material = 'Metal'
# # #                 probs = np.array([0.02, 0.05, 0.05, 0.1, 0.05, 0.7, 0.02, 0.01])
# # #             else:
# # #                 material = 'Concrete'
# # #                 probs = np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
# # #         else:
# # #             material = 'Stone'
# # #             probs = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
# # #         return material, probs
# # #     except Exception as e:
# # #         st.error(f"❌ Fallback material classification failed: {str(e)}")
# # #         return 'Unknown', np.array([0.125] * 8)

# # # def visualize_material_classification(material, probabilities):
# # #     try:
# # #         fig = go.Figure(data=[
# # #             go.Bar(
# # #                 x=['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone'],
# # #                 y=probabilities,
# # #                 marker_color=['#8B4513', '#FF4500', '#FFD700', '#808080', '#DEB887', '#C0C0C0', '#F5F5DC', '#F4A460'],
# # #                 text=[f'{p:.3f}' for p in probabilities],
# # #                 textposition='auto'
# # #             )
# # #         ])
# # #         fig.update_layout(
# # #             title=f'Material Classification: {material}',
# # #             yaxis_title='Confidence Score',
# # #             yaxis_range=[0, 1],
# # #             xaxis_tickangle=45,
# # #             plot_bgcolor='rgba(0,0,0,0)',
# # #             paper_bgcolor='rgba(0,0,0,0)',
# # #             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
# # #         )
# # #         return fig
# # #     except Exception as e:
# # #         st.error(f"❌ Material visualization failed: {str(e)}")
# # #         return None

# # # Define material classes globally
# # material_classes = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']

# # # === 1. CLASSIFICATION FUNCTION ===
# # def classify_material(image_np):
# #     try:
# #         if material_model is None:
# #             st.warning("⚠ Material classification model not loaded. Using texture-based fallback.")
# #             return classify_material_fallback(image_np)

# #         transform = transforms.Compose([
# #             transforms.ToPILImage(),
# #             transforms.Resize((224, 224)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                  std=[0.229, 0.224, 0.225])
# #         ])

# #         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
# #         image_tensor = transform(image_rgb).unsqueeze(0)

# #         with torch.no_grad():
# #             output = material_model(image_tensor)
# #             probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

# #         predicted_index = np.argmax(probabilities)
# #         predicted_material = material_classes[predicted_index]

# #         # If confidence is too low, fallback
# #         if probabilities[predicted_index] < 0.5:
# #             return classify_material_fallback(image_np)

# #         return predicted_material, probabilities

# #     except Exception as e:
# #         st.error(f"❌ Model-based classification failed: {e}")
# #         return classify_material_fallback(image_np)


# # # === 2. FALLBACK CLASSIFIER ===
# # def classify_material_fallback(image_np):
# #     try:
# #         hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
# #         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# #         mean_hue = np.mean(hsv[:, :, 0])
# #         mean_sat = np.mean(hsv[:, :, 1])
# #         mean_val = np.mean(hsv[:, :, 2])
# #         std_val = np.std(hsv[:, :, 2])
# #         texture = np.std(gray)

# #         mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))

# #         # Debug
# #         st.write({
# #             "Mean Hue": mean_hue,
# #             "Saturation": mean_sat,
# #             "Value": mean_val,
# #             "Value STD": std_val,
# #             "Texture": texture,
# #             "R": mean_r, "G": mean_g, "B": mean_b
# #         })

# #         # Rule-based logic
# #         if mean_r > mean_g > mean_b and mean_sat > 80:
# #             return 'Brick', np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
# #         elif texture > 60 and mean_val < 120:
# #             if mean_val < 80:
# #                 return 'Stone', np.array([0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])
# #             else:
# #                 return 'Sandstone', np.array([0.2, 0.05, 0.05, 0.05, 0.02, 0.01, 0.1, 0.6])
# #         elif mean_val > 180 and std_val < 30:
# #             if mean_sat < 20:
# #                 if texture < 20:
# #                     return 'Marble', np.array([0.05, 0.05, 0.1, 0.05, 0.02, 0.01, 0.7, 0.02])
# #                 else:
# #                     return 'Plaster', np.array([0.05, 0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.01])
# #             else:
# #                 return 'Concrete', np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
# #         elif 10 < mean_hue < 30 and mean_sat > 50:
# #             return 'Wood', np.array([0.05, 0.1, 0.05, 0.05, 0.7, 0.02, 0.02, 0.01])
# #         elif mean_val > 150 and texture > 40:
# #             if mean_sat < 30:
# #                 return 'Metal', np.array([0.02, 0.05, 0.05, 0.1, 0.05, 0.7, 0.02, 0.01])
# #             else:
# #                 return 'Concrete', np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
# #         else:
# #             return 'Stone', np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

# #     except Exception as e:
# #         st.error(f"❌ Fallback classification failed: {e}")
# #         return 'Unknown', np.array([0.125] * 8)


# # # === 3. PLOTLY VISUALIZATION ===
# # def visualize_material_classification(material, probabilities):
# #     try:
# #         fig = go.Figure(data=[
# #             go.Bar(
# #                 x=material_classes,
# #                 y=probabilities,
# #                 marker_color=['#8B4513', '#FF4500', '#FFD700', '#808080',
# #                               '#DEB887', '#C0C0C0', '#F5F5DC', '#F4A460'],
# #                 text=[f'{p:.2f}' for p in probabilities],
# #                 textposition='auto'
# #             )
# #         ])
# #         fig.update_layout(
# #             title=f'Material Classification: {material}',
# #             yaxis_title='Confidence Score',
# #             yaxis_range=[0, 1],
# #             xaxis_tickangle=45,
# #             plot_bgcolor='rgba(0,0,0,0)',
# #             paper_bgcolor='rgba(0,0,0,0)',
# #             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
# #         )
# #         return fig
# #     except Exception as e:
# #         st.error(f"❌ Visualization failed: {e}")
# #         return None
    
# # def plot_crack_severity(crack_details):
# #     try:
# #         severities = [crack['severity'] for crack in crack_details if crack['severity']]
# #         if not severities:
# #             return None
# #         severity_counts = pd.Series(severities).value_counts()
# #         fig = px.pie(
# #             names=severity_counts.index,
# #             values=severity_counts.values,
# #             title='Crack Severity Distribution',
# #             color=severity_counts.index,
# #             color_discrete_map={
# #                 'Minor': '#00FF00',
# #                 'Moderate': '#FFFF00',
# #                 'Severe': '#FFA500',
# #                 'Critical': '#FF0000'
# #             }
# #         )
# #         fig.update_traces(textinfo='percent+label')
# #         fig.update_layout(
# #             plot_bgcolor='rgba(0,0,0,0)',
# #             paper_bgcolor='rgba(0,0,0,0)',
# #             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
# #         )
# #         return fig
# #     except Exception as e:
# #         st.error(f"❌ Crack severity visualization failed: {str(e)}")
# #         return None

# # def plot_biological_growth_area(growth_area_cm2, total_image_area_cm2):
# #     try:
# #         if growth_area_cm2 == 0:
# #             return None
# #         fig = go.Figure(data=[
# #             go.Bar(
# #                 x=['Biological Growth', 'Non-Growth Area'],
# #                 y=[growth_area_cm2, total_image_area_cm2 - growth_area_cm2],
# #                 marker_color=['#FF0000', '#00FF00'],
# #                 text=[f'{growth_area_cm2:.2f} cm²', f'{(total_image_area_cm2 - growth_area_cm2):.2f} cm²'],
# #                 textposition='auto'
# #             )
# #         ])
# #         fig.update_layout(
# #             title='Biological Growth Area vs. Total Area',
# #             yaxis_title='Area (cm²)',
# #             plot_bgcolor='rgba(0,0,0,0)',
# #             paper_bgcolor='rgba(0,0,0,0)',
# #             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
# #         )
# #         return fig
# #     except Exception as e:
# #         st.error(f"❌ Biological growth area visualization failed: {str(e)}")
# #         return None

# # def plot_environmental_footprints(carbon_footprint, water_footprint):
# #     try:
# #         fig = go.Figure(data=[
# #             go.Bar(
# #                 x=['Carbon Footprint', 'Water Footprint'],
# #                 y=[carbon_footprint, water_footprint],
# #                 marker_color=['#FF4500', '#00B7EB'],
# #                 text=[f'{carbon_footprint:.2f} kg CO2e', f'{water_footprint:.2f} liters'],
# #                 textposition='auto'
# #             )
# #         ])
# #         fig.update_layout(
# #             title='Environmental Footprints',
# #             yaxis_title='Impact',
# #             plot_bgcolor='rgba(0,0,0,0)',
# #             paper_bgcolor='rgba(0,0,0,0)',
# #             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
# #         )
# #         return fig
# #     except Exception as e:
# #         st.error(f"❌ Environmental footprints visualization failed: {str(e)}")
# #         return None

# # def estimate_material_quantity(crack_details, growth_area_cm2, material):
# #     try:
# #         density = {
# #             'Concrete': 0.0024,
# #             'Brick': 0.0019,
# #             'Steel': 0.0078,
# #             'Wood': 0.0007,
# #             'Stone': 0.0027,
# #             'Plaster': 0.0012,
# #             'Marble': 0.0027,
# #             'Sandstone': 0.0023,
# #             'Metal': 0.0078,
# #             'Glass': 0.0025
# #         }.get(material, 0.002)
# #         crack_area_cm2 = sum(c['width_cm'] * c['length_cm'] for c in crack_details if 'crack' in c['label'].lower())
# #         crack_volume_cm3 = crack_area_cm2 * 1.0
# #         growth_volume_cm3 = growth_area_cm2 * 0.1
# #         total_volume_cm3 = crack_volume_cm3 + growth_volume_cm3
# #         total_mass_kg = total_volume_cm3 * density
# #         return max(total_mass_kg, 0.1)
# #     except Exception as e:
# #         st.error(f"❌ Material quantity estimation failed: {str(e)}")
# #         return 0.1

# # def predict_crack_progression(crack_details):
# #     try:
# #         if not crack_details:
# #             return "No cracks detected for progression analysis."
# #         predictions = []
# #         for i, crack in enumerate(crack_details):
# #             current_area = crack['width_cm'] * crack['length_cm']
# #             time_points = np.array([0, 3, 6, 9, 12]).reshape(-1, 1)
# #             severity_factor = {
# #                 'Minor': 1.05,
# #                 'Moderate': 1.15,
# #                 'Severe': 1.25,
# #                 'Critical': 1.35
# #             }.get(crack['severity'], 1.1)
# #             areas = [current_area * (severity_factor ** (t/12)) for t in [0, 3, 6, 9, 12]]
# #             areas = np.array(areas).reshape(-1, 1)
# #             model = LinearRegression()
# #             model.fit(time_points, areas)
# #             future_months = np.array([15, 18, 21, 24]).reshape(-1, 1)
# #             future_areas = model.predict(future_months)
# #             prediction_text = f"Crack {i+1} ({crack['label']}): Current area {current_area:.2f} cm²\n"
# #             prediction_text += f"Predicted progression: 15 months: {future_areas[0][0]:.2f} cm², "
# #             prediction_text += f"18 months: {future_areas[1][0]:.2f} cm², "
# #             prediction_text += f"24 months: {future_areas[3][0]:.2f} cm²"
# #             predictions.append(prediction_text)
# #         return "\n\n".join(predictions)
# #     except Exception as e:
# #         st.error(f"❌ Crack progression prediction failed: {str(e)}")
# #         return "Unable to predict crack progression."

# # def calculate_carbon_footprint(material: str, quantity_kg: float) -> float:
# #     emission_factors = {
# #         'Concrete': 0.13,
# #         'Stone': 0.07,
# #         'Brick': 0.22,
# #         'Steel': 1.85,
# #         'Wood': 0.04,
# #         'Plaster': 0.12,
# #         'Marble': 0.15,
# #         'Sandstone': 0.09,
# #         'Glass': 1.0,
# #         'Metal': 1.85
# #     }
# #     factor = emission_factors.get(material, 0.1)
# #     return quantity_kg * factor

# # def calculate_water_footprint(material: str, quantity_kg: float) -> float:
# #     water_factors = {
# #         'Concrete': 150,
# #         'Brick': 120,
# #         'Steel': 200,
# #         'Wood': 50,
# #         'Stone': 30,
# #         'Plaster': 80,
# #         'Marble': 100,
# #         'Sandstone': 60,
# #         'Glass': 300,
# #         'Metal': 200
# #     }
# #     factor = water_factors.get(material, 100)
# #     return quantity_kg * factor

# # def main():
# #     st.title("🏛 Heritage Sites Health Monitoring System")
# #     st.markdown("""
# #     *Advanced AI-powered monitoring system for heritage site conservation*
    
# #     This system provides comprehensive analysis including:
# #     - 🔍 *Crack Detection*: AI-powered structural damage identification
# #     - 🌿 *Biological Growth Detection*: Moss, algae, and vegetation analysis
# #     - 🧱 *Material Classification*: Automated building material identification
# #     - 📊 *Depth Analysis*: 3D structural assessment
# #     - 📈 *Predictive Analytics*: Future deterioration forecasting
# #     - 🌍 *Environmental Impact*: Automatic carbon and water footprint analysis
# #     - 📊 *Data Visualization*: Interactive charts for analysis insights
# #     """)

# #     st.sidebar.title("🛠 Analysis Settings")
# #     px_to_cm_ratio = st.sidebar.slider(
# #         "Pixel to CM Ratio",
# #         min_value=0.01,
# #         max_value=1.0,
# #         value=0.1,
# #         step=0.01
# #     )
# #     confidence_threshold = st.sidebar.slider(
# #         "Detection Confidence Threshold",
# #         min_value=0.1,
# #         max_value=0.9,
# #         value=0.3,
# #         step=0.05
# #     )

# #     tab1, tab2, tab3 = st.tabs(["🔬 Image Analysis", "🌍 Environmental Footprints", "ℹ About"])

# #     with tab1:
# #         st.header("Upload and Analyze Heritage Site Images")
# #         uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
# #         if uploaded_file is not None:
# #             st.subheader("📸 Original Image")
# #             image_np = load_and_preprocess_image(uploaded_file)
# #             if image_np is not None:
# #                 st.session_state.image_np = image_np
# #                 st.session_state.image_name = uploaded_file.name
# #                 st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
# #                 if st.button("🚀 Start Analysis", type="primary"):
# #                     with st.spinner("🔄 Performing comprehensive analysis..."):
# #                         results = {}
# #                         progress_bar = st.progress(0)
# #                         status_text = st.empty()

# #                         status_text.text("🔍 Detecting structural damage...")
# #                         progress_bar.progress(10)
# #                         annotated_image, crack_details = detect_with_yolo(image_np, px_to_cm_ratio)
# #                         results['crack_detection'] = (annotated_image, crack_details)

# #                         status_text.text("🧱 Analyzing building materials...")
# #                         progress_bar.progress(30)
# #                         material, probabilities = classify_material(image_np)
# #                         results['material_analysis'] = (material, probabilities)

# #                         status_text.text("🌿 Detecting biological growth...")
# #                         progress_bar.progress(50)
# #                         growth_image = detect_biological_growth(image_np, crack_details)
# #                         results['biological_growth'] = growth_image

# #                         status_text.text("📐 Performing segmentation...")
# #                         progress_bar.progress(70)
# #                         segmented_image, seg_results = segment_image(image_np)
# #                         results['segmentation'] = (segmented_image, seg_results)

# #                         status_text.text("📊 Generating depth and edge analysis...")
# #                         progress_bar.progress(80)
# #                         preprocessed = preprocess_image_for_depth_estimation(image_np)
# #                         depth_heatmap = create_depth_estimation_heatmap(preprocessed)
# #                         results['depth_analysis'] = depth_heatmap
# #                         edges = apply_canny_edge_detection(image_np)
# #                         results['edge_detection'] = edges

# #                         status_text.text("🌍 Calculating environmental impact...")
# #                         progress_bar.progress(90)
# #                         bio_growth_area = calculate_biological_growth_area(
# #                             crack_details, seg_results, image_np, px_to_cm_ratio
# #                         )
# #                         quantity_kg = estimate_material_quantity(crack_details, bio_growth_area, material)
# #                         carbon_footprint = calculate_carbon_footprint(material, quantity_kg)
# #                         water_footprint = calculate_water_footprint(material, quantity_kg)
# #                         results['environmental'] = (carbon_footprint, water_footprint, quantity_kg, bio_growth_area)

# #                         status_text.text("✅ Analysis complete!")
# #                         progress_bar.progress(100)

# #                         st.session_state.analysis_results = {
# #                             'crack_details': crack_details,
# #                             'material': material,
# #                             'probabilities': probabilities,
# #                             'bio_growth_area': bio_growth_area,
# #                             'carbon_footprint': carbon_footprint,
# #                             'water_footprint': water_footprint,
# #                             'quantity_kg': quantity_kg,
# #                             'seg_results': seg_results
# #                         }

# #                         st.success("🎉 Analysis completed successfully!")

# #                         # Display results in a grid layout
# #                         st.subheader("🔍 Analysis Results and Visualizations")

# #                         # Image-based results (Crack Detection, Biological Growth, Segmentation, Depth, Edge)
# #                         st.markdown("### Image Analysis Results")
# #                         col1, col2, col3 = st.columns(3)
# #                         with col1:
# #                             st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
# #                                      caption="Crack Detection", use_container_width=True)
# #                         with col2:
# #                             st.image(cv2.cvtColor(growth_image, cv2.COLOR_BGR2RGB),
# #                                      caption="Biological Growth Detection", use_container_width=True)
# #                         with col3:
# #                             st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
# #                                      caption="Image Segmentation", use_container_width=True)

# #                         col4, col5 = st.columns(2)
# #                         with col4:
# #                             st.image(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB),
# #                                      caption="Depth Estimation", use_container_width=True)
# #                         with col5:
# #                             st.image(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB),
# #                                      caption="Edge Detection", use_container_width=True)

# #                         # Chart-based visualizations
# #                         st.markdown("### Data Visualizations")
# #                         total_area_cm2 = (image_np.shape[1] * px_to_cm_ratio) * (image_np.shape[0] * px_to_cm_ratio)
# #                         col6, col7 = st.columns(2)
# #                         with col6:
# #                             severity_fig = plot_crack_severity(crack_details)
# #                             if severity_fig:
# #                                 st.plotly_chart(severity_fig, use_container_width=True)
# #                             else:
# #                                 st.info("No crack severity data to visualize.")
# #                         with col7:
# #                             material_fig = visualize_material_classification(material, probabilities)
# #                             if material_fig:
# #                                 st.plotly_chart(material_fig, use_container_width=True)

# #                         col8, col9 = st.columns(2)
# #                         with col8:
# #                             growth_fig = plot_biological_growth_area(bio_growth_area, total_area_cm2)
# #                             if growth_fig:
# #                                 st.plotly_chart(growth_fig, use_container_width=True)
# #                             else:
# #                                 st.info("No biological growth data to visualize.")
# #                         with col9:
# #                             footprint_fig = plot_environmental_footprints(carbon_footprint, water_footprint)
# #                             if footprint_fig:
# #                                 st.plotly_chart(footprint_fig, use_container_width=True)

# #                         # Summary metrics
# #                         st.markdown("### Analysis Summary")
# #                         col10, col11, col12 = st.columns(3)
# #                         with col10:
# #                             st.metric("Dominant Material", material)
# #                             if crack_details:
# #                                 st.write("*Crack Details:*")
# #                                 for i, crack in enumerate(crack_details, 1):
# #                                     severity_color = {
# #                                         'Minor': '🟢',
# #                                         'Moderate': '🟡',
# #                                         'Severe': '🟠',
# #                                         'Critical': '🔴'
# #                                     }.get(crack['severity'], '⚪')
# #                                     st.write(f"{severity_color} Crack {i}: {crack['width_cm']:.2f} × {crack['length_cm']:.2f} cm - {crack['severity']}")
# #                             else:
# #                                 st.info("✅ No structural damage detected")
# #                         with col11:
# #                             st.metric("Biological Growth Area", f"{bio_growth_area:.2f} cm²")
# #                             st.metric("Material Quantity", f"{quantity_kg:.2f} kg")
# #                         with col12:
# #                             st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
# #                             st.metric("Water Footprint", f"{water_footprint:.2f} liters")

# #                         # Predictive analysis
# #                         st.subheader("📈 Predictive Analysis")
# #                         with st.expander("Crack Progression Forecast"):
# #                             prediction = predict_crack_progression(crack_details)
# #                             st.text(prediction)

# #                         progress_bar.empty()
# #                         status_text.empty()

# #     with tab2:
# #         st.header("🌍 Environmental Footprints")
# #         st.markdown("Automatically calculated carbon and water footprints based on the latest image analysis.")
# #         if st.session_state.analysis_results is None:
# #             st.info("ℹ No analysis results available. Please perform an analysis in the Image Analysis tab.")
# #         else:
# #             results = st.session_state.analysis_results
# #             quantity_kg = results.get('quantity_kg', 0)
# #             carbon_footprint = results.get('carbon_footprint', 0)
# #             water_footprint = results.get('water_footprint', 0)
# #             material = results.get('material', 'Unknown')
# #             st.subheader("Footprint Results")
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 st.metric("Material", material)
# #                 st.metric("Estimated Quantity", f"{quantity_kg:.2f} kg")
# #             with col2:
# #                 st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
# #                 st.metric("Water Footprint", f"{water_footprint:.2f} liters")

# #     with tab3:
# #         st.header("ℹ About Heritage Sites Health Monitoring")
# #         st.markdown("""
# #         ### 🎯 Purpose
# #         This application aids heritage conservators in monitoring the structural health of historical buildings using AI.

# #         ### 🔧 Technologies Used
# #         - *YOLOv8*: Object detection and segmentation
# #         - *Computer Vision*: Advanced image processing
# #         - *Machine Learning*: Material classification
# #         - *Plotly*: Interactive visualizations
        
# #             ### 📋 Features
# #             - *Automated Detection*: Identifies structural damage
# #             - *Material Analysis*: Recognizes building materials
# #             - *Biological Growth*: Detects moss and algae
# #             - *Depth Analysis*: 3D structural assessment
# #             - *Predictive Modeling*: Forecasts deterioration
# #             - *Environmental Impact*: Automatic carbon and water footprint analysis
# #             - *Visualization*: Interactive charts for analysis insights
        
# #         ### 🚀 How to Use
# #         1. Upload images
# #         2. Adjust settings
# #         3. Analyze
# #         4. Review results and visualizations
        

# #         """)

# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import pandas as pd
# from ultralytics import YOLO
# from sklearn.linear_model import LinearRegression
# import os
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import warnings
# try:
#     from scipy import ndimage
#     from skimage import measure
#     SCIPY_SKIMAGE_AVAILABLE = True
# except ImportError:
#     SCIPY_SKIMAGE_AVAILABLE = False
#     st.warning("SciPy or scikit-image not installed. Some advanced features may be limited.")
# from pdf_report import save_image_to_temp, generate_pdf_report  # Import PDF functions

# warnings.filterwarnings('ignore')

# # Set page config as the FIRST Streamlit command
# st.set_page_config(page_title="Heritage Sites Health Monitoring", layout="wide")

# # Initialize session state
# if 'analysis_results' not in st.session_state:
#     st.session_state.analysis_results = None
# if 'image_name' not in st.session_state:
#     st.session_state.image_name = None
# if 'image_np' not in st.session_state:
#     st.session_state.image_np = None
# if 'analysis_completed' not in st.session_state:
#     st.session_state.analysis_completed = False
# if 'pdf_buffer' not in st.session_state:
#     st.session_state.pdf_buffer = None

# # Model loading
# @st.cache_resource
# def load_models():
#     models_status = {}
#     try:
#         yolo_path = "runs/detect/train3/weights/best.pt"
#         if os.path.exists(yolo_path):
#             yolo_model = YOLO(yolo_path)
#             models_status['yolo'] = f"Custom model loaded from {yolo_path}"
#         else:
#             yolo_model = YOLO("yolov8n.pt")
#             models_status['yolo'] = "Using default YOLOv8n model"

#         seg_path = "./segmentation_model/weights/best.pt"
#         if os.path.exists(seg_path):
#             segmentation_model = YOLO(seg_path)
#             models_status['segmentation'] = f"Custom segmentation model loaded from {seg_path}"
#         else:
#             segmentation_model = YOLO("yolov8n-seg.pt")
#             models_status['segmentation'] = "Using default YOLOv8n-seg model"

#         material_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
#         material_model.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(material_model.last_channel, 8)
#         )
#         material_model.eval()
#         models_status['material'] = "MobileNetV2 model loaded with custom classifier for 8 material types"

#         st.success("✅ All models loaded successfully!")
#         with st.expander("Model Loading Details"):
#             for model_type, status in models_status.items():
#                 st.info(f"{model_type.capitalize()}: {status}")

#         return yolo_model, segmentation_model, material_model
#     except Exception as e:
#         st.error(f"❌ Model loading failed: {str(e)}")
#         st.warning("⚠ Some features may not work properly without models.")
#         return None, None, None

# yolo_model, segmentation_model, material_model = load_models()

# # Image processing functions
# def load_and_preprocess_image(uploaded_file):
#     try:
#         image = Image.open(uploaded_file).convert('RGB')
#         image_np = np.array(image)
#         if image_np.size == 0:
#             raise ValueError("Invalid image file: The uploaded image appears to be empty.")
#         return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     except Exception as e:
#         st.error(f"❌ Error loading or preprocessing the image: {str(e)}")
#         return None

# def calculate_severity(width_cm, length_cm, label):
#     try:
#         if 'crack' not in label.lower():
#             return None
#         area = width_cm * length_cm
#         max_dimension = max(width_cm, length_cm)
#         if max_dimension < 0.5 and area < 0.25:
#             return 'Minor'
#         elif max_dimension < 1.5 and area < 2.0:
#             return 'Moderate'
#         elif max_dimension < 3.0 and area < 6.0:
#             return 'Severe'
#         else:
#             return 'Critical'
#     except Exception as e:
#         st.error(f"❌ Severity calculation error: {str(e)}")
#         return 'Unknown'

# def detect_with_yolo(image_np, px_to_cm_ratio=0.1):
#     try:
#         if yolo_model is None:
#             st.warning("⚠ YOLO model is not loaded. Using placeholder detection.")
#             height, width = image_np.shape[:2]
#             placeholder_detection = {
#                 'width_cm': 2.5,
#                 'length_cm': 3.0,
#                 'severity': 'Moderate',
#                 'confidence': 0.85,
#                 'label': 'crack',
#                 'bbox': (width//4, height//4, 3*width//4, 3*height//4)
#             }
#             annotated_image = image_np.copy()
#             x1, y1, x2, y2 = placeholder_detection['bbox']
#             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(annotated_image, f"Placeholder: crack (2.5cm x 3.0cm) - Moderate",
#                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             return annotated_image, [placeholder_detection]

#         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#         results = yolo_model.predict(image_rgb, conf=0.3)
#         crack_details = []
#         annotated_image = image_np.copy()

#         for result in results:
#             if result.boxes is not None and len(result.boxes) > 0:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                     width_px = x2 - x1
#                     length_px = y2 - y1
#                     width_cm = width_px * px_to_cm_ratio
#                     length_cm = length_px * px_to_cm_ratio

#                     class_id = int(box.cls[0].cpu().numpy())
#                     label = yolo_model.names.get(class_id, "unknown")
#                     confidence = float(box.conf[0].cpu().numpy())
#                     severity = calculate_severity(width_cm, length_cm, label)

#                     crack_details.append({
#                         'width_cm': width_cm,
#                         'length_cm': length_cm,
#                         'severity': severity,
#                         'confidence': confidence,
#                         'label': label,
#                         'bbox': (x1, y1, x2, y2)
#                     })

#                     color = {
#                         'Minor': (0, 255, 0),
#                         'Moderate': (0, 255, 255),
#                         'Severe': (0, 165, 255),
#                         'Critical': (255, 0, 0)
#                     }.get(severity, (128, 128, 128))

#                     cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
#                     severity_text = f" - {severity}" if severity else ""
#                     display_text = f"{label}: {width_cm:.2f}cm x {length_cm:.2f}cm{severity_text} ({confidence:.2f})"
#                     text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#                     cv2.rectangle(annotated_image, (x1, y1-25), (x1 + text_size[0], y1), (0, 0, 0), -1)
#                     cv2.putText(annotated_image, display_text, (x1, y1 - 10),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#         if not crack_details:
#             st.info("ℹ No objects detected by YOLO.")
#         return annotated_image, crack_details
#     except Exception as e:
#         st.error(f"❌ YOLO detection failed: {str(e)}")
#         return image_np, []

# def detect_biological_growth_advanced(image_np):
#     try:
#         growth_image = image_np.copy()
#         hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
#         lower_green1 = np.array([35, 40, 40])
#         upper_green1 = np.array([85, 255, 255])
#         lower_green2 = np.array([25, 30, 20])
#         upper_green2 = np.array([95, 200, 150])
#         mask_green1 = cv2.inRange(hsv, lower_green1, upper_green1)
#         mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
#         combined_mask = cv2.bitwise_or(mask_green1, mask_green2)
#         kernel = np.ones((5, 5), np.uint8)
#         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
#         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
#         contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         growth_detected = False
#         total_growth_area = 0
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > 100:
#                 growth_detected = True
#                 cv2.drawContours(growth_image, [contour], -1, (0, 0, 255), 2)
#                 x, y, w, h = cv2.boundingRect(contour)
#                 cv2.putText(growth_image, f"Growth: {area:.0f}px",
#                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 total_growth_area += area
#         if not growth_detected:
#             cv2.putText(growth_image, "No biological growth detected",
#                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(growth_image, f"Total growth area: {total_growth_area:.0f} pixels",
#                        (50, image_np.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         return growth_image, growth_detected, total_growth_area
#     except Exception as e:
#         st.error(f"❌ Biological growth detection failed: {str(e)}")
#         return image_np, False, 0

# def detect_biological_growth(image_np, crack_details):
#     try:
#         growth_image, growth_detected_advanced, growth_area_px = detect_biological_growth_advanced(image_np)
#         yolo_growth_detected = False
#         for crack in crack_details:
#             if any(keyword in crack['label'].lower() for keyword in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
#                 x1, y1, x2, y2 = crack['bbox']
#                 width_cm = crack['width_cm']
#                 length_cm = crack['length_cm']
#                 confidence = crack['confidence']
#                 cv2.rectangle(growth_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
#                 cv2.putText(growth_image, f"YOLO Growth: {width_cm:.2f}cm x {length_cm:.2f}cm ({confidence:.2f})",
#                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#                 yolo_growth_detected = True
#         overall_growth_detected = growth_detected_advanced or yolo_growth_detected
#         if overall_growth_detected:
#             cv2.putText(growth_image, "BIOLOGICAL GROWTH DETECTED",
#                        (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#         return growth_image
#     except Exception as e:
#         st.error(f"❌ Enhanced biological growth detection failed: {str(e)}")
#         return image_np

# def calculate_biological_growth_area(crack_details, seg_results, image_np, px_to_cm_ratio):
#     try:
#         total_area_cm2 = 0
#         areas_counted = set()
#         if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
#             masks = seg_results[0].masks.data.cpu().numpy()
#             image_height, image_width = image_np.shape[:2]
#             for mask in masks:
#                 resized_mask = cv2.resize(mask.astype(np.uint8), (image_width, image_height),
#                                         interpolation=cv2.INTER_NEAREST)
#                 mask_area_px = np.sum(resized_mask)
#                 mask_area_cm2 = mask_area_px * (px_to_cm_ratio ** 2)
#                 if mask_area_cm2 not in areas_counted:
#                     total_area_cm2 += mask_area_cm2
#                     areas_counted.add(mask_area_cm2)
#         else:
#             for crack in crack_details:
#                 if any(keyword in crack['label'].lower() for keyword in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
#                     area = crack['width_cm'] * crack['length_cm']
#                     if area not in areas_counted:
#                         total_area_cm2 += area
#                         areas_counted.add(area)
#             _, growth_detected, growth_area_px = detect_biological_growth_advanced(image_np)
#             if growth_detected and growth_area_px > 0:
#                 growth_area_cm2 = growth_area_px * (px_to_cm_ratio ** 2)
#                 if growth_area_cm2 not in areas_counted:
#                     total_area_cm2 += growth_area_cm2
#                     areas_counted.add(growth_area_cm2)
#         return total_area_cm2
#     except Exception as e:
#         st.error(f"❌ Biological growth area calculation failed: {str(e)}")
#         return 0

# def segment_image(image_np):
#     try:
#         if segmentation_model is None:
#             st.warning("⚠ Segmentation model is not loaded. Creating placeholder segmentation.")
#             gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#             edges = cv2.Canny(gray, 100, 200)
#             segmented_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#             cv2.putText(segmented_image, "Placeholder Segmentation",
#                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             return segmented_image, None
#         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#         results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
#         if results and len(results) > 0:
#             segmented_image = results[0].plot()
#             return segmented_image, results
#         else:
#             st.info("ℹ No segments detected in the image.")
#             return image_np, None
#     except Exception as e:
#         st.error(f"❌ Segmentation failed: {str(e)}")
#         return image_np, None

# def preprocess_image_for_depth_estimation(image_np):
#     try:
#         gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#         blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#         return cv2.equalizeHist(blurred_image)
#     except Exception as e:
#         st.error(f"❌ Depth preprocessing failed: {str(e)}")
#         return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# def create_depth_estimation_heatmap(equalized_image):
#     try:
#         _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
#         shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)
#         depth_estimation = 255 - shadow_region
#         depth_estimation_normalized = cv2.normalize(depth_estimation, None, 0, 255, cv2.NORM_MINMAX)
#         return cv2.applyColorMap(depth_estimation_normalized.astype(np.uint8), cv2.COLORMAP_JET)
#     except Exception as e:
#         st.error(f"❌ Depth heatmap creation failed: {str(e)}")
#         return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# def apply_canny_edge_detection(image_np):
#     try:
#         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 100, 200)
#         return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     except Exception as e:
#         st.error(f"❌ Edge detection failed: {str(e)}")
#         return image_np

# # Define material classes globally
# material_classes = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']

# def classify_material(image_np):
#     try:
#         if material_model is None:
#             st.warning("⚠ Material classification model not loaded. Using texture-based fallback.")
#             return classify_material_fallback(image_np)

#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#         image_tensor = transform(image_rgb).unsqueeze(0)

#         with torch.no_grad():
#             output = material_model(image_tensor)
#             probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

#         predicted_index = np.argmax(probabilities)
#         predicted_material = material_classes[predicted_index]

#         if probabilities[predicted_index] < 0.5:
#             return classify_material_fallback(image_np)

#         return predicted_material, probabilities

#     except Exception as e:
#         st.error(f"❌ Model-based classification failed: {e}")
#         return classify_material_fallback(image_np)

# def classify_material_fallback(image_np):
#     try:
#         hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
#         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

#         mean_hue = np.mean(hsv[:, :, 0])
#         mean_sat = np.mean(hsv[:, :, 1])
#         mean_val = np.mean(hsv[:, :, 2])
#         std_val = np.std(hsv[:, :, 2])
#         texture = np.std(gray)

#         mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))

#         if mean_r > mean_g > mean_b and mean_sat > 80:
#             return 'Brick', np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
#         elif texture > 60 and mean_val < 120:
#             if mean_val < 80:
#                 return 'Stone', np.array([0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])
#             else:
#                 return 'Sandstone', np.array([0.2, 0.05, 0.05, 0.05, 0.02, 0.01, 0.1, 0.6])
#         elif mean_val > 180 and std_val < 30:
#             if mean_sat < 20:
#                 if texture < 20:
#                     return 'Marble', np.array([0.05, 0.05, 0.1, 0.05, 0.02, 0.01, 0.7, 0.02])
#                 else:
#                     return 'Plaster', np.array([0.05, 0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.01])
#             else:
#                 return 'Concrete', np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
#         elif 10 < mean_hue < 30 and mean_sat > 50:
#             return 'Wood', np.array([0.05, 0.1, 0.05, 0.05, 0.7, 0.02, 0.02, 0.01])
#         elif mean_val > 150 and texture > 40:
#             if mean_sat < 30:
#                 return 'Metal', np.array([0.02, 0.05, 0.05, 0.1, 0.05, 0.7, 0.02, 0.01])
#             else:
#                 return 'Concrete', np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
#         else:
#             return 'Stone', np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

#     except Exception as e:
#         st.error(f"❌ Fallback classification failed: {e}")
#         return 'Unknown', np.array([0.125] * 8)

# def visualize_material_classification(material, probabilities):
#     try:
#         fig = go.Figure(data=[
#             go.Bar(
#                 x=material_classes,
#                 y=probabilities,
#                 marker_color=['#8B4513', '#FF4500', '#FFD700', '#808080',
#                               '#DEB887', '#C0C0C0', '#F5F5DC', '#F4A460'],
#                 text=[f'{p:.2f}' for p in probabilities],
#                 textposition='auto'
#             )
#         ])
#         fig.update_layout(
#             title=f'Material Classification: {material}',
#             yaxis_title='Confidence Score',
#             yaxis_range=[0, 1],
#             xaxis_tickangle=45,
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
#         )
#         return fig
#     except Exception as e:
#         st.error(f"❌ Visualization failed: {e}")
#         return None

# def plot_crack_severity(crack_details):
#     try:
#         severities = [crack['severity'] for crack in crack_details if crack['severity']]
#         if not severities:
#             return None
#         severity_counts = pd.Series(severities).value_counts()
#         fig = px.pie(
#             names=severity_counts.index,
#             values=severity_counts.values,
#             title='Crack Severity Distribution',
#             color=severity_counts.index,
#             color_discrete_map={
#                 'Minor': '#00FF00',
#                 'Moderate': '#FFFF00',
#                 'Severe': '#FFA500',
#                 'Critical': '#FF0000'
#             }
#         )
#         fig.update_traces(textinfo='percent+label')
#         fig.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
#         )
#         return fig
#     except Exception as e:
#         st.error(f"❌ Crack severity visualization failed: {str(e)}")
#         return None

# def plot_biological_growth_area(growth_area_cm2, total_image_area_cm2):
#     try:
#         if growth_area_cm2 == 0:
#             return None
#         fig = go.Figure(data=[
#             go.Bar(
#                 x=['Biological Growth', 'Non-Growth Area'],
#                 y=[growth_area_cm2, total_image_area_cm2 - growth_area_cm2],
#                 marker_color=['#FF0000', '#00FF00'],
#                 text=[f'{growth_area_cm2:.2f} cm²', f'{(total_image_area_cm2 - growth_area_cm2):.2f} cm²'],
#                 textposition='auto'
#             )
#         ])
#         fig.update_layout(
#             title='Biological Growth Area vs. Total Area',
#             yaxis_title='Area (cm²)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
#         )
#         return fig
#     except Exception as e:
#         st.error(f"❌ Biological growth area visualization failed: {str(e)}")
#         return None

# def plot_environmental_footprints(carbon_footprint, water_footprint):
#     try:
#         fig = go.Figure(data=[
#             go.Bar(
#                 x=['Carbon Footprint', 'Water Footprint'],
#                 y=[carbon_footprint, water_footprint],
#                 marker_color=['#FF4500', '#00B7EB'],
#                 text=[f'{carbon_footprint:.2f} kg CO2e', f'{water_footprint:.2f} liters'],
#                 textposition='auto'
#             )
#         ])
#         fig.update_layout(
#             title='Environmental Footprints',
#             yaxis_title='Impact',
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
#         )
#         return fig
#     except Exception as e:
#         st.error(f"❌ Environmental footprints visualization failed: {str(e)}")
#         return None

# def estimate_material_quantity(crack_details, growth_area_cm2, material):
#     try:
#         density = {
#             'Concrete': 0.0024,
#             'Brick': 0.0019,
#             'Steel': 0.0078,
#             'Wood': 0.0007,
#             'Stone': 0.0027,
#             'Plaster': 0.0012,
#             'Marble': 0.0027,
#             'Sandstone': 0.0023,
#             'Metal': 0.0078,
#             'Glass': 0.0025
#         }.get(material, 0.002)
#         crack_area_cm2 = sum(c['width_cm'] * c['length_cm'] for c in crack_details if 'crack' in c['label'].lower())
#         crack_volume_cm3 = crack_area_cm2 * 1.0
#         growth_volume_cm3 = growth_area_cm2 * 0.1
#         total_volume_cm3 = crack_volume_cm3 + growth_volume_cm3
#         total_mass_kg = total_volume_cm3 * density
#         return max(total_mass_kg, 0.1)
#     except Exception as e:
#         st.error(f"❌ Material quantity estimation failed: {str(e)}")
#         return 0.1

# def predict_crack_progression(crack_details):
#     try:
#         if not crack_details:
#             return "No cracks detected for progression analysis."
#         predictions = []
#         for i, crack in enumerate(crack_details):
#             current_area = crack['width_cm'] * crack['length_cm']
#             time_points = np.array([0, 3, 6, 9, 12]).reshape(-1, 1)
#             severity_factor = {
#                 'Minor': 1.05,
#                 'Moderate': 1.15,
#                 'Severe': 1.25,
#                 'Critical': 1.35
#             }.get(crack['severity'], 1.1)
#             areas = [current_area * (severity_factor ** (t/12)) for t in [0, 3, 6, 9, 12]]
#             areas = np.array(areas).reshape(-1, 1)
#             model = LinearRegression()
#             model.fit(time_points, areas)
#             future_months = np.array([15, 18, 21, 24]).reshape(-1, 1)
#             future_areas = model.predict(future_months)
#             prediction_text = f"Crack {i+1} ({crack['label']}): Current area {current_area:.2f} cm²\n"
#             prediction_text += f"Predicted progression: 15 months: {future_areas[0][0]:.2f} cm², "
#             prediction_text += f"18 months: {future_areas[1][0]:.2f} cm², "
#             prediction_text += f"24 months: {future_areas[3][0]:.2f} cm²"
#             predictions.append(prediction_text)
#         return "\n\n".join(predictions)
#     except Exception as e:
#         st.error(f"❌ Crack progression prediction failed: {str(e)}")
#         return "Unable to predict crack progression."

# def calculate_carbon_footprint(material: str, quantity_kg: float) -> float:
#     emission_factors = {
#         'Concrete': 0.13,
#         'Stone': 0.07,
#         'Brick': 0.22,
#         'Steel': 1.85,
#         'Wood': 0.04,
#         'Plaster': 0.12,
#         'Marble': 0.15,
#         'Sandstone': 0.09,
#         'Glass': 1.0,
#         'Metal': 1.85
#     }
#     factor = emission_factors.get(material, 0.1)
#     return quantity_kg * factor

# def calculate_water_footprint(material: str, quantity_kg: float) -> float:
#     water_factors = {
#         'Concrete': 150,
#         'Brick': 120,
#         'Steel': 200,
#         'Wood': 50,
#         'Stone': 30,
#         'Plaster': 80,
#         'Marble': 100,
#         'Sandstone': 60,
#         'Glass': 300,
#         'Metal': 200
#     }
#     factor = water_factors.get(material, 100)
#     return quantity_kg * factor

# def main():
#     st.title("🏛 Heritage Sites Health Monitoring System")
#     st.markdown("""
#     *Advanced AI-powered monitoring system for heritage site conservation*
    
#     This system provides comprehensive analysis including:
#     - 🔍 *Crack Detection*: AI-powered structural damage identification
#     - 🌿 *Biological Growth Detection*: Moss, algae, and vegetation analysis
#     - 🧱 *Material Classification*: Automated building material identification
#     - 📊 *Depth Analysis*: 3D structural assessment
#     - 📈 *Predictive Analytics*: Future deterioration forecasting
#     - 🌍 *Environmental Impact*: Automatic carbon and water footprint analysis
#     - 📊 *Data Visualization*: Interactive charts for analysis insights
#     - 📄 *PDF Reports*: Downloadable analysis reports with images
#     """)

#     st.sidebar.title("🛠 Analysis Settings")
#     px_to_cm_ratio = st.sidebar.slider(
#         "Pixel to CM Ratio",
#         min_value=0.01,
#         max_value=1.0,
#         value=0.1,
#         step=0.01
#     )
#     confidence_threshold = st.sidebar.slider(
#         "Detection Confidence Threshold",
#         min_value=0.1,
#         max_value=0.9,
#         value=0.3,
#         step=0.05
#     )

#     tab1, tab2, tab3 = st.tabs(["🔬 Image Analysis", "🌍 Environmental Footprints", "ℹ About"])

#     with tab1:
#         st.header("Upload and Analyze Heritage Site Images")
#         uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
#         if uploaded_file is not None:
#             st.subheader("📸 Original Image")
#             image_np = load_and_preprocess_image(uploaded_file)
#             if image_np is not None:
#                     st.session_state.image_np = image_np
#                     st.session_state.image_name = uploaded_file.name
#                     st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
#                     if st.button("🚀 Start Analysis", type="primary"):
#                         with st.spinner("🔄 Performing comprehensive analysis..."):
#                             results = {}
#                             progress_bar = st.progress(0)
#                             status_text = st.empty()

#                             status_text.text("🔍 Detecting structural damage...")
#                             progress_bar.progress(10)
#                             annotated_image, crack_details = detect_with_yolo(image_np, px_to_cm_ratio)
#                             results['crack_detection'] = (annotated_image, crack_details)

#                             status_text.text("🧱 Analyzing building materials...")
#                             progress_bar.progress(30)
#                             material, probabilities = classify_material(image_np)
#                             results['material_analysis'] = (material, probabilities)

#                             status_text.text("🌿 Detecting biological growth...")
#                             progress_bar.progress(50)
#                             growth_image = detect_biological_growth(image_np, crack_details)
#                             results['biological_growth'] = growth_image

#                             status_text.text("📐 Performing segmentation...")
#                             progress_bar.progress(70)
#                             segmented_image, seg_results = segment_image(image_np)
#                             results['segmentation'] = (segmented_image, seg_results)

#                             status_text.text("📊 Generating depth and edge analysis...")
#                             progress_bar.progress(80)
#                             preprocessed = preprocess_image_for_depth_estimation(image_np)
#                             depth_heatmap = create_depth_estimation_heatmap(preprocessed)
#                             results['depth_analysis'] = depth_heatmap
#                             edges = apply_canny_edge_detection(image_np)
#                             results['edge_detection'] = edges

#                             status_text.text("🌍 Calculating environmental impact...")
#                             progress_bar.progress(90)
#                             bio_growth_area = calculate_biological_growth_area(
#                                 crack_details, seg_results, image_np, px_to_cm_ratio
#                             )
#                             quantity_kg = estimate_material_quantity(crack_details, bio_growth_area, material)
#                             carbon_footprint = calculate_carbon_footprint(material, quantity_kg)
#                             water_footprint = calculate_water_footprint(material, quantity_kg)
#                             results['environmental'] = (carbon_footprint, water_footprint, quantity_kg, bio_growth_area)

#                             status_text.text("✅ Analysis complete!")
#                             progress_bar.progress(100)

#                             st.session_state.analysis_results = {
#                                 'crack_details': crack_details,
#                                 'material': material,
#                                 'probabilities': probabilities,
#                                 'bio_growth_area': bio_growth_area,
#                                 'carbon_footprint': carbon_footprint,
#                                 'water_footprint': water_footprint,
#                                 'quantity_kg': quantity_kg,
#                                 'seg_results': seg_results,
#                                 'annotated_image': annotated_image,
#                                 'growth_image': growth_image,
#                                 'segmented_image': segmented_image,
#                                 'depth_heatmap': depth_heatmap,
#                                 'edges': edges
#                             }
#                             st.session_state.analysis_completed = True
#                             st.session_state.pdf_buffer = None  # Reset PDF buffer

#                             st.success("🎉 Analysis completed successfully!")

#                             # Display results
#                             st.subheader("🔍 Analysis Results and Visualizations")

#                             # Image-based results
#                             st.markdown("### Image Analysis Results")
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
#                                          caption="Crack Detection", use_container_width=True)
#                             with col2:
#                                 st.image(cv2.cvtColor(growth_image, cv2.COLOR_BGR2RGB),
#                                          caption="Biological Growth Detection", use_container_width=True)
#                             with col3:
#                                 st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
#                                          caption="Image Segmentation", use_container_width=True)

#                             col4, col5 = st.columns(2)
#                             with col4:
#                                 st.image(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB),
#                                          caption="Depth Estimation", use_container_width=True)
#                             with col5:
#                                 st.image(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB),
#                                          caption="Edge Detection", use_container_width=True)

#                             # Chart-based visualizations
#                             st.markdown("### Data Visualizations")
#                             total_area_cm2 = (image_np.shape[1] * px_to_cm_ratio) * (image_np.shape[0] * px_to_cm_ratio)
                            
#                             col6, col7 = st.columns(2)
#                             with col6:
#                                 severity_fig = plot_crack_severity(crack_details)
#                                 if severity_fig:
#                                     st.plotly_chart(severity_fig, use_container_width=True)
#                                 else:
#                                     st.info("No crack severity data to visualize.")
#                                 with col7:
#                                     material_fig = visualize_material_classification(material, probabilities)
#                                     if material_fig:
#                                         st.plotly_chart(material_fig)

#                             col8, col9 = st.columns(2)
#                             with col8:
#                                 growth_fig = plot_biological_growth_area(bio_growth_area, total_area_cm2)
#                                 if growth_fig:
#                                     st.plotly_chart(growth_fig, use_container_width=True)
#                                 else:
#                                     st.info("No biological growth data to visualize.")
#                                 with col9:
#                                     footprint_fig = plot_environmental_footprints(carbon_footprint, water_footprint)
#                                     if footprint_fig:
#                                         st.plotly_chart(footprint_fig, use_container_width=True)

#                             # Summary metrics
#                             st.markdown("### Analysis Summary")
#                             col10,col11,col12 = st.columns(3)
#                             with col10:
#                                 st.metric("Dominant Material", material)
#                                 if crack_details:
#                                     st.write("*Crack Details:*")
#                                     for i, crack in enumerate(crack_details, 1):
#                                         severity_color = {
#                                             'Minor': '🟢',
#                                             'Moderate': '🟡',
#                                             'Severe': '🟠',
#                                             'Critical': '🔴'
#                                         }.get(crack['severity'], '⚪')
#                                         st.write(f"{severity_color} Crack {i}: {crack['width_cm']:.2f} × {crack['length_cm']:.2f} cm - {crack['severity']}")
#                                 else:
#                                     st.info("✅ No structural damage detected")
#                             with col11:
#                                 st.metric("Biological Growth Area", f"{bio_growth_area:.2f} cm²")
#                                 st.metric("Material Quantity", f"{quantity_kg:.2f} kg")
#                             with col12:
#                                 st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
#                                 st.metric("Water Footprint", f"{water_footprint:.2f} liters")

#                             # Predictive analysis
#                             st.subheader("📈 Predictive Analysis")
#                             with st.expander("Crack Progression Forecast"):
#                                 prediction = predict_crack_progression(crack_details)
#                                 st.text(prediction)

#                             progress_bar.empty()
#                             status_text.empty()

#         # PDF Download Section
#         if st.session_state.analysis_completed and st.session_state.analysis_results:
#             st.subheader("📄 Download Analysis Report")
#             with st.form(key="pdf_generate_form"):
#                 submit_button = st.form_submit_button("Generate PDF Report")
#                 if submit_button:
#                     with st.spinner("📝 Generating PDF report..."):
#                         results = st.session_state.analysis_results
#                         pdf_buffer = generate_pdf_report(
#                             st.session_state.image_np,
#                             results['annotated_image'],
#                             results['growth_image'],
#                             results['segmented_image'],
#                             results['depth_heatmap'],
#                             results['edges'],
#                             results['crack_details'],
#                             results['material'],
#                             results['probabilities'],
#                             results['bio_growth_area'],
#                             results['quantity_kg'],
#                             results['carbon_footprint'],
#                             results['water_footprint'],
#                             predict_crack_progression(results['crack_details'])
#                         )
#                         if pdf_buffer:
#                             st.session_state.pdf_buffer = pdf_buffer
#                             st.success("✅ PDF report generated successfully!")
#                         else:
#                             st.error("❌ Failed to generate PDF report.")

#             if st.session_state.pdf_buffer:
#                 st.download_button(
#                     label="📥 Download PDF",
#                     data=st.session_state.pdf_buffer,
#                     file_name=f"Heritage_Analysis_Report_{st.session_state.image_name or 'report'}.pdf",
#                     mime="application/pdf",
#                     key="pdf_download_button"
#                 )

#     with tab2:
#         st.header("🌍 Environmental Footprints")
#         st.markdown("Automatically calculated carbon and water footprints based on the latest image analysis.")
#         if st.session_state.analysis_results is None:
#             st.info("ℹ No analysis results available. Please perform an analysis in the Image Analysis tab.")
#         else:
#             results = st.session_state.analysis_results
#             quantity_kg = results.get('quantity_kg', 0)
#             carbon_footprint = results.get('carbon_footprint', 0)
#             water_footprint = results.get('water_footprint', 0)
#             material = results.get('material', 'Unknown')
#             st.subheader("Footprint Results")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Material", material)
#                 st.metric("Estimated Quantity", f"{quantity_kg:.2f} kg")
#             with col2:
#                 st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
#                 st.metric("Water Footprint", f"{water_footprint:.2f} liters")

#     with tab3:
#         st.header("ℹ About Heritage Sites Health Monitoring")
#         st.markdown("""
#         ### 🎯 Purpose
#         This application aids heritage conservators in monitoring the structural health of historical buildings using AI.

#         ### 🔧 Technologies Used
#         - *YOLOv8*: Object detection and segmentation
#         - *Computer Vision*: Advanced image processing
#         - *Machine Learning*: API
#         - *Plotly*: Interactive visualizations
#         - *ReportLab*: PDF report generation
        
#         ### 📋 Features
#         - *Automated Detection*: Identifies structural damage
#         - *Material Analysis*: Recognizes building materials
#         - *Biological Growth*: Detects moss and algae
#         - *Depth Analysis*: 3D structural assessment
#         - *Predictive Modeling*: Forecasts deterioration
#         - *Environmental Impact*: Automatic carbon and water footprint analysis
#         - *Visualization*: Interactive charts for analysis insights
#         - *PDF Reports*: Downloadable analysis reports with images
        
#         ### 🚀 How to Use
#         1. Upload images
#         2. Adjust settings
#         3. Analyze
#         4. Review results and visualizations
#         5. Generate and download PDF report
#         """)

# if __name__ == "__main__":
#     main()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import warnings
try:
    from scipy import ndimage
    from skimage import measure
    SCIPY_SKIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_SKIMAGE_AVAILABLE = False
    st.warning("SciPy or scikit-image not installed. Some advanced features may be limited.")
from pdf_report import save_image_to_temp, generate_pdf_report

# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="Heritage Sites Health Monitoring", layout="wide")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'video_frame_results' not in st.session_state:
    st.session_state.video_frame_results = {}
if 'image_name' not in st.session_state:
    st.session_state.image_name = None
if 'image_np' not in st.session_state:
    st.session_state.image_np = None
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'pdf_buffer' not in st.session_state:
    st.session_state.pdf_buffer = None
if 'video_pdf_buffers' not in st.session_state:
    st.session_state.video_pdf_buffers = {}

# Model loading
@st.cache_resource
def load_models():
    models_status = {}
    try:
        yolo_path = "runs/detect/train3/weights/best.pt"
        if os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
            models_status['yolo'] = f"Custom model loaded from {yolo_path}"
        else:
            yolo_model = YOLO("yolov8n.pt")
            models_status['yolo'] = "Using default YOLOv8n model"

        seg_path = "./segmentation_model/weights/best.pt"
        if os.path.exists(seg_path):
            segmentation_model = YOLO(seg_path)
            models_status['segmentation'] = f"Custom segmentation model loaded from {seg_path}"
        else:
            segmentation_model = YOLO("yolov8n-seg.pt")
            models_status['segmentation'] = "Using default YOLOv8n-seg model"

        material_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        material_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(material_model.last_channel, 8)
        )
        material_model.eval()
        models_status['material'] = "MobileNetV2 model loaded with custom classifier for 8 material types"

        st.success("✅ All models loaded successfully!")
        with st.expander("Model Loading Details"):
            for model_type, status in models_status.items():
                st.info(f"{model_type.capitalize()}: {status}")

        return yolo_model, segmentation_model, material_model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.warning("⚠ Some features may not work properly without models.")
        return None, None, None

yolo_model, segmentation_model, material_model = load_models()

# Image processing functions
def load_and_preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        if image_np.size == 0:
            raise ValueError("Invalid image file: The uploaded image appears to be empty.")
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"❌ Error loading or preprocessing the image: {str(e)}")
        return None

def calculate_severity(width_cm, length_cm, label):
    try:
        if 'crack' not in label.lower():
            return None
        area = width_cm * length_cm
        max_dimension = max(width_cm, length_cm)
        if max_dimension < 0.5 and area < 0.25:
            return 'Minor'
        elif max_dimension < 1.5 and area < 2.0:
            return 'Moderate'
        elif max_dimension < 3.0 and area < 6.0:
            return 'Severe'
        else:
            return 'Critical'
    except Exception as e:
        st.error(f"❌ Severity calculation error: {str(e)}")
        return 'Unknown'

def detect_with_yolo(image_np, px_to_cm_ratio=0.1):
    try:
        if yolo_model is None:
            st.warning("⚠ YOLO model is not loaded. Using placeholder detection.")
            height, width = image_np.shape[:2]
            placeholder_detection = {
                'width_cm': 2.5,
                'length_cm': 3.0,
                'severity': 'Moderate',
                'confidence': 0.85,
                'label': 'crack',
                'bbox': (width//4, height//4, 3*width//4, 3*height//4)
            }
            annotated_image = image_np.copy()
            x1, y1, x2, y2 = placeholder_detection['bbox']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Placeholder: crack (2.5cm x 3.0cm) - Moderate",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return annotated_image, [placeholder_detection]

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(image_rgb, conf=0.3)
        crack_details = []
        annotated_image = image_np.copy()

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    width_px = x2 - x1
                    length_px = y2 - y1
                    width_cm = width_px * px_to_cm_ratio
                    length_cm = length_px * px_to_cm_ratio

                    class_id = int(box.cls[0].cpu().numpy())
                    label = yolo_model.names.get(class_id, "unknown")
                    confidence = float(box.conf[0].cpu().numpy())
                    severity = calculate_severity(width_cm, length_cm, label)

                    crack_details.append({
                        'width_cm': width_cm,
                        'length_cm': length_cm,
                        'severity': severity,
                        'confidence': confidence,
                        'label': label,
                        'bbox': (x1, y1, x2, y2)
                    })

                    color = {
                        'Minor': (0, 255, 0),
                        'Moderate': (0, 255, 255),
                        'Severe': (0, 165, 255),
                        'Critical': (255, 0, 0)
                    }.get(severity, (128, 128, 128))

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                    severity_text = f" - {severity}" if severity else ""
                    display_text = f"{label}: {width_cm:.2f}cm x {length_cm:.2f}cm{severity_text} ({confidence:.2f})"
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1-25), (x1 + text_size[0], y1), (0, 0, 0), -1)
                    cv2.putText(annotated_image, display_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if not crack_details:
            st.info("ℹ No objects detected by YOLO.")
        return annotated_image, crack_details
    except Exception as e:
        st.error(f"❌ YOLO detection failed: {str(e)}")
        return image_np, []

def detect_biological_growth_advanced(image_np):
    try:
        growth_image = image_np.copy()
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        lower_green2 = np.array([25, 30, 20])
        upper_green2 = np.array([95, 200, 150])
        mask_green1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
        combined_mask = cv2.bitwise_or(mask_green1, mask_green2)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        growth_detected = False
        total_growth_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                growth_detected = True
                cv2.drawContours(growth_image, [contour], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(growth_image, f"Growth: {area:.0f}px",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                total_growth_area += area
        if not growth_detected:
            cv2.putText(growth_image, "No biological growth detected",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(growth_image, f"Total growth area: {total_growth_area:.0f} pixels",
                       (50, image_np.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return growth_image, growth_detected, total_growth_area
    except Exception as e:
        st.error(f"❌ Biological growth detection failed: {str(e)}")
        return image_np, False, 0

def detect_biological_growth(image_np, crack_details):
    try:
        growth_image, growth_detected_advanced, growth_area_px = detect_biological_growth_advanced(image_np)
        yolo_growth_detected = False
        for crack in crack_details:
            if any(keyword in crack['label'].lower() for keyword in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
                x1, y1, x2, y2 = crack['bbox']
                width_cm = crack['width_cm']
                length_cm = crack['length_cm']
                confidence = crack['confidence']
                cv2.rectangle(growth_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(growth_image, f"YOLO Growth: {width_cm:.2f}cm x {length_cm:.2f}cm ({confidence:.2f})",
                           (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                yolo_growth_detected = True
        overall_growth_detected = growth_detected_advanced or yolo_growth_detected
        if overall_growth_detected:
            cv2.putText(growth_image, "BIOLOGICAL GROWTH DETECTED",
                       (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return growth_image
    except Exception as e:
        st.error(f"❌ Enhanced biological growth detection failed: {str(e)}")
        return image_np

def calculate_biological_growth_area(crack_details, seg_results, image_np, px_to_cm_ratio):
    try:
        total_area_cm2 = 0
        areas_counted = set()
        if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
            masks = seg_results[0].masks.data.cpu().numpy()
            image_height, image_width = image_np.shape[:2]
            for mask in masks:
                resized_mask = cv2.resize(mask.astype(np.uint8), (image_width, image_height),
                                        interpolation=cv2.INTER_NEAREST)
                mask_area_px = np.sum(resized_mask)
                mask_area_cm2 = mask_area_px * (px_to_cm_ratio ** 2)
                if mask_area_cm2 not in areas_counted:
                    total_area_cm2 += mask_area_cm2
                    areas_counted.add(mask_area_cm2)
        else:
            for crack in crack_details:
                if any(keyword in crack['label'].lower() for keyword in ['moss', 'growth', 'algae', 'lichen', 'vegetation']):
                    area = crack['width_cm'] * crack['length_cm']
                    if area not in areas_counted:
                        total_area_cm2 += area
                        areas_counted.add(area)
            _, growth_detected, growth_area_px = detect_biological_growth_advanced(image_np)
            if growth_detected and growth_area_px > 0:
                growth_area_cm2 = growth_area_px * (px_to_cm_ratio ** 2)
                if growth_area_cm2 not in areas_counted:
                    total_area_cm2 += growth_area_cm2
                    areas_counted.add(growth_area_cm2)
        return total_area_cm2
    except Exception as e:
        st.error(f"❌ Biological growth area calculation failed: {str(e)}")
        return 0

def segment_image(image_np):
    try:
        if segmentation_model is None:
            st.warning("⚠ Segmentation model is not loaded. Creating placeholder segmentation.")
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            segmented_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(segmented_image, "Placeholder Segmentation",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return segmented_image, None
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
        if results and len(results) > 0:
            segmented_image = results[0].plot()
            return segmented_image, results
        else:
            st.info("ℹ No segments detected in the image.")
            return image_np, None
    except Exception as e:
        st.error(f"❌ Segmentation failed: {str(e)}")
        return image_np, None

def preprocess_image_for_depth_estimation(image_np):
    try:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        return cv2.equalizeHist(blurred_image)
    except Exception as e:
        st.error(f"❌ Depth preprocessing failed: {str(e)}")
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

def create_depth_estimation_heatmap(equalized_image):
    try:
        _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
        shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)
        depth_estimation = 255 - shadow_region
        depth_estimation_normalized = cv2.normalize(depth_estimation, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(depth_estimation_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    except Exception as e:
        st.error(f"❌ Depth heatmap creation failed: {str(e)}")
        return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

def apply_canny_edge_detection(image_np):
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        st.error(f"❌ Edge detection failed: {str(e)}")
        return image_np

# Define material classes globally
material_classes = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']

def classify_material(image_np):
    try:
        if material_model is None:
            st.warning("⚠ Material classification model not loaded. Using texture-based fallback.")
            return classify_material_fallback(image_np)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            output = material_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

        predicted_index = np.argmax(probabilities)
        predicted_material = material_classes[predicted_index]

        if probabilities[predicted_index] < 0.5:
            return classify_material_fallback(image_np)

        return predicted_material, probabilities
    except Exception as e:
        st.error(f"❌ Model-based classification failed: {e}")
        return classify_material_fallback(image_np)

def classify_material_fallback(image_np):
    try:
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])
        std_val = np.std(hsv[:, :, 2])
        texture = np.std(gray)

        mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))

        st.write({
            "Mean Hue": mean_hue,
            "Saturation": mean_sat,
            "Value": mean_val,
            "Value STD": std_val,
            "Texture": texture,
            "R": mean_r, "G": mean_g, "B": mean_b
        })

        if mean_r > mean_g > mean_b and mean_sat > 80:
            return 'Brick', np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
        elif texture > 60 and mean_val < 120:
            if mean_val < 80:
                return 'Stone', np.array([0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])
            else:
                return 'Sandstone', np.array([0.2, 0.05, 0.05, 0.05, 0.02, 0.01, 0.1, 0.6])
        elif mean_val > 180 and std_val < 30:
            if mean_sat < 20:
                if texture < 20:
                    return 'Marble', np.array([0.05, 0.05, 0.1, 0.05, 0.02, 0.01, 0.7, 0.02])
                else:
                    return 'Plaster', np.array([0.05, 0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.01])
            else:
                return 'Concrete', np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
        elif 10 < mean_hue < 30 and mean_sat > 50:
            return 'Wood', np.array([0.05, 0.1, 0.05, 0.05, 0.7, 0.02, 0.02, 0.01])
        elif mean_val > 150 and texture > 40:
            if mean_sat < 30:
                return 'Metal', np.array([0.02, 0.05, 0.05, 0.1, 0.05, 0.7, 0.02, 0.01])
            else:
                return 'Concrete', np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.03, 0.02])
        else:
            return 'Stone', np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    except Exception as e:
        st.error(f"❌ Fallback classification failed: {e}")
        return 'Unknown', np.array([0.125] * 8)

def visualize_material_classification(material, probabilities):
    try:
        fig = go.Figure(data=[
            go.Bar(
                x=material_classes,
                y=probabilities,
                marker_color=['#8B4513', '#FF4500', '#FFD700', '#808080',
                              '#DEB887', '#C0C0C0', '#F5F5DC', '#F4A460'],
                text=[f'{p:.2f}' for p in probabilities],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title=f'Material Classification: {material}',
            yaxis_title='Confidence Score',
            yaxis_range=[0, 1],
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
        )
        return fig
    except Exception as e:
        st.error(f"❌ Visualization failed: {e}")
        return None

def plot_crack_severity(crack_details):
    try:
        severities = [crack['severity'] for crack in crack_details if crack['severity']]
        if not severities:
            return None
        severity_counts = pd.Series(severities).value_counts()
        fig = px.pie(
            names=severity_counts.index,
            values=severity_counts.values,
            title='Crack Severity Distribution',
            color=severity_counts.index,
            color_discrete_map={
                'Minor': '#00FF00',
                'Moderate': '#FFFF00',
                'Severe': '#FFA500',
                'Critical': '#FF0000'
            }
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
        )
        return fig
    except Exception as e:
        st.error(f"❌ Crack severity visualization failed: {str(e)}")
        return None

def plot_biological_growth_area(growth_area_cm2, total_image_area_cm2):
    try:
        if growth_area_cm2 == 0:
            return None
        fig = go.Figure(data=[
            go.Bar(
                x=['Biological Growth', 'Non-Growth Area'],
                y=[growth_area_cm2, total_image_area_cm2 - growth_area_cm2],
                marker_color=['#FF0000', '#00FF00'],
                text=[f'{growth_area_cm2:.2f} cm²', f'{(total_image_area_cm2 - growth_area_cm2):.2f} cm²'],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='Biological Growth Area vs. Total Area',
            yaxis_title='Area (cm²)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
        )
        return fig
    except Exception as e:
        st.error(f"❌ Biological growth area visualization failed: {str(e)}")
        return None

def plot_environmental_footprints(carbon_footprint, water_footprint):
    try:
        fig = go.Figure(data=[
            go.Bar(
                x=['Carbon Footprint', 'Water Footprint'],
                y=[carbon_footprint, water_footprint],
                marker_color=['#FF4500', '#00B7EB'],
                text=[f'{carbon_footprint:.2f} kg CO2e', f'{water_footprint:.2f} liters'],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='Environmental Footprints',
            yaxis_title='Impact',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000')
        )
        return fig
    except Exception as e:
        st.error(f"❌ Environmental footprints visualization failed: {str(e)}")
        return None

def estimate_material_quantity(crack_details, growth_area_cm2, material):
    try:
        density = {
            'Concrete': 0.0024,
            'Brick': 0.0019,
            'Steel': 0.0078,
            'Wood': 0.0007,
            'Stone': 0.0027,
            'Plaster': 0.0012,
            'Marble': 0.0027,
            'Sandstone': 0.0023,
            'Metal': 0.0078,
            'Glass': 0.0025
        }.get(material, 0.002)
        crack_area_cm2 = sum(c['width_cm'] * c['length_cm'] for c in crack_details if 'crack' in c['label'].lower())
        crack_volume_cm3 = crack_area_cm2 * 1.0
        growth_volume_cm3 = growth_area_cm2 * 0.1
        total_volume_cm3 = crack_volume_cm3 + growth_volume_cm3
        total_mass_kg = total_volume_cm3 * density
        return max(total_mass_kg, 0.1)
    except Exception as e:
        st.error(f"❌ Material quantity estimation failed: {str(e)}")
        return 0.1

def predict_crack_progression(crack_details):
    try:
        if not crack_details:
            return "No cracks detected for progression analysis."
        predictions = []
        for i, crack in enumerate(crack_details):
            current_area = crack['width_cm'] * crack['length_cm']
            time_points = np.array([0, 3, 6, 9, 12]).reshape(-1, 1)
            severity_factor = {
                'Minor': 1.05,
                'Moderate': 1.15,
                'Severe': 1.25,
                'Critical': 1.35
            }.get(crack['severity'], 1.1)
            areas = [current_area * (severity_factor ** (t/12)) for t in [0, 3, 6, 9, 12]]
            areas = np.array(areas).reshape(-1, 1)
            model = LinearRegression()
            model.fit(time_points, areas)
            future_months = np.array([15, 18, 21, 24]).reshape(-1, 1)
            future_areas = model.predict(future_months)
            prediction_text = f"Crack {i+1} ({crack['label']}): Current area {current_area:.2f} cm²\n"
            prediction_text += f"Predicted progression: 15 months: {future_areas[0][0]:.2f} cm², "
            prediction_text += f"18 months: {future_areas[1][0]:.2f} cm², "
            prediction_text += f"24 months: {future_areas[3][0]:.2f} cm²"
            predictions.append(prediction_text)
        return "\n\n".join(predictions)
    except Exception as e:
        st.error(f"❌ Crack progression prediction failed: {str(e)}")
        return "Unable to predict crack progression."

def calculate_carbon_footprint(material: str, quantity_kg: float) -> float:
    emission_factors = {
        'Concrete': 0.13,
        'Stone': 0.07,
        'Brick': 0.22,
        'Steel': 1.85,
        'Wood': 0.04,
        'Plaster': 0.12,
        'Marble': 0.15,
        'Sandstone': 0.09,
        'Glass': 1.0,
        'Metal': 1.85
    }
    factor = emission_factors.get(material, 0.1)
    return quantity_kg * factor

def calculate_water_footprint(material: str, quantity_kg: float) -> float:
    water_factors = {
        'Concrete': 150,
        'Brick': 120,
        'Steel': 200,
        'Wood': 50,
        'Stone': 30,
        'Plaster': 80,
        'Marble': 100,
        'Sandstone': 60,
        'Glass': 300,
        'Metal': 200
    }
    factor = water_factors.get(material, 100)
    return quantity_kg * factor

def main():
    st.title("🏛 Heritage Sites Health Monitoring System")
    st.markdown("""
    *Advanced AI-powered monitoring system for heritage site conservation*
    
    This system provides comprehensive analysis including:
    - 🔍 *Crack Detection*: AI-powered structural damage identification
    - 🌿 *Biological Growth Detection*: Moss, algae, and vegetation analysis
    - 🧱 *Material Classification*: Automated building material identification
    - 📊 *Depth Analysis*: 3D structural assessment
    - 📈 *Predictive Analytics*: Future deterioration forecasting
    - 🌍 *Environmental Impact*: Automatic carbon and water footprint analysis
    - 📊 *Data Visualization*: Interactive charts for analysis insights
    - 📄 *PDF Reports*: Downloadable analysis reports with images
    """)

    st.sidebar.title("🛠 Analysis Settings")
    px_to_cm_ratio = st.sidebar.slider(
        "Pixel to CM Ratio",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01
    )
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05
    )

    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Image Analysis", "📽 Video Analysis", "🌍 Environmental Footprints", "ℹ About"])

    with tab1:
        st.header("Upload and Analyze Heritage Site Images")
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.subheader("📸 Original Image")
            image_np = load_and_preprocess_image(uploaded_file)
            if image_np is not None:
                st.session_state.image_np = image_np
                st.session_state.image_name = uploaded_file.name
                st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
                if st.button("🚀 Start Analysis", type="primary"):
                    with st.spinner("🔄 Performing comprehensive analysis..."):
                        results = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("🔍 Detecting structural damage...")
                        progress_bar.progress(10)
                        annotated_image, crack_details = detect_with_yolo(image_np, px_to_cm_ratio)
                        results['crack_detection'] = (annotated_image, crack_details)

                        status_text.text("🧱 Analyzing building materials...")
                        progress_bar.progress(30)
                        material, probabilities = classify_material(image_np)
                        results['material_analysis'] = (material, probabilities)

                        status_text.text("🌿 Detecting biological growth...")
                        progress_bar.progress(50)
                        growth_image = detect_biological_growth(image_np, crack_details)
                        results['biological_growth'] = growth_image

                        status_text.text("📐 Performing segmentation...")
                        progress_bar.progress(70)
                        segmented_image, seg_results = segment_image(image_np)
                        results['segmentation'] = (segmented_image, seg_results)

                        status_text.text("📊 Generating depth and edge analysis...")
                        progress_bar.progress(80)
                        preprocessed = preprocess_image_for_depth_estimation(image_np)
                        depth_heatmap = create_depth_estimation_heatmap(preprocessed)
                        results['depth_analysis'] = depth_heatmap
                        edges = apply_canny_edge_detection(image_np)
                        results['edge_detection'] = edges

                        status_text.text("🌍 Calculating environmental impact...")
                        progress_bar.progress(90)
                        bio_growth_area = calculate_biological_growth_area(
                            crack_details, seg_results, image_np, px_to_cm_ratio
                        )
                        quantity_kg = estimate_material_quantity(crack_details, bio_growth_area, material)
                        carbon_footprint = calculate_carbon_footprint(material, quantity_kg)
                        water_footprint = calculate_water_footprint(material, quantity_kg)
                        results['environmental'] = (carbon_footprint, water_footprint, quantity_kg, bio_growth_area)

                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)

                        st.session_state.analysis_results = {
                            'crack_details': crack_details,
                            'material': material,
                            'probabilities': probabilities,
                            'bio_growth_area': bio_growth_area,
                            'carbon_footprint': carbon_footprint,
                            'water_footprint': water_footprint,
                            'quantity_kg': quantity_kg,
                            'seg_results': seg_results,
                            'annotated_image': annotated_image,
                            'growth_image': growth_image,
                            'segmented_image': segmented_image,
                            'depth_heatmap': depth_heatmap,
                            'edges': edges
                        }
                        st.session_state.analysis_completed = True
                        st.session_state.pdf_buffer = None

                        st.success("🎉 Analysis completed successfully!")

                        # Display results
                        st.subheader("🔍 Analysis Results and Visualizations")

                        # Image-based results
                        st.markdown("### Image Analysis Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                                     caption="Crack Detection", use_container_width=True)
                        with col2:
                            st.image(cv2.cvtColor(growth_image, cv2.COLOR_BGR2RGB),
                                     caption="Biological Growth Detection", use_container_width=True)
                        with col3:
                            st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
                                     caption="Image Segmentation", use_container_width=True)

                        col4, col5 = st.columns(2)
                        with col4:
                            st.image(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB),
                                     caption="Depth Estimation", use_container_width=True)
                        with col5:
                            st.image(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB),
                                     caption="Edge Detection", use_container_width=True)

                        # Chart-based visualizations
                        st.markdown("### Data Visualizations")
                        total_area_cm2 = (image_np.shape[1] * px_to_cm_ratio) * (image_np.shape[0] * px_to_cm_ratio)
                        col6, col7 = st.columns(2)
                        with col6:
                            severity_fig = plot_crack_severity(crack_details)
                            if severity_fig:
                                st.plotly_chart(severity_fig, use_container_width=True)
                            else:
                                st.info("No crack severity data to visualize.")
                        with col7:
                            material_fig = visualize_material_classification(material, probabilities)
                            if material_fig:
                                st.plotly_chart(material_fig, use_container_width=True)

                        col8, col9 = st.columns(2)
                        with col8:
                            growth_fig = plot_biological_growth_area(bio_growth_area, total_area_cm2)
                            if growth_fig:
                                st.plotly_chart(growth_fig, use_container_width=True)
                            else:
                                st.info("No biological growth data to visualize.")
                        with col9:
                            footprint_fig = plot_environmental_footprints(carbon_footprint, water_footprint)
                            if footprint_fig:
                                st.plotly_chart(footprint_fig, use_container_width=True)

                        # Summary metrics
                        st.markdown("### Analysis Summary")
                        col10, col11, col12 = st.columns(3)
                        with col10:
                            st.metric("Dominant Material", material)
                            if crack_details:
                                st.write("*Crack Details:*")
                                for i, crack in enumerate(crack_details, 1):
                                    severity_color = {
                                        'Minor': '🟢',
                                        'Moderate': '🟡',
                                        'Severe': '🟠',
                                        'Critical': '🔴'
                                    }.get(crack['severity'], '⚪')
                                    st.write(f"{severity_color} Crack {i}: {crack['width_cm']:.2f} × {crack['length_cm']:.2f} cm - {crack['severity']}")
                            else:
                                st.info("✅ No structural damage detected")
                        with col11:
                            st.metric("Biological Growth Area", f"{bio_growth_area:.2f} cm²")
                            st.metric("Material Quantity", f"{quantity_kg:.2f} kg")
                        with col12:
                            st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
                            st.metric("Water Footprint", f"{water_footprint:.2f} liters")

                        # Predictive analysis
                        st.subheader("📈 Predictive Analysis")
                        with st.expander("Crack Progression Forecast"):
                            prediction = predict_crack_progression(crack_details)
                            st.text(prediction)

                        progress_bar.empty()
                        status_text.empty()

        # PDF Download Section
        if st.session_state.analysis_completed and st.session_state.analysis_results:
            st.subheader("📄 Download Analysis Report")
            with st.form(key="pdf_generate_form"):
                submit_button = st.form_submit_button("Generate PDF Report")
                if submit_button:
                    with st.spinner("📝 Generating PDF report..."):
                        results = st.session_state.analysis_results
                        pdf_buffer = generate_pdf_report(
                            st.session_state.image_np,
                            results['annotated_image'],
                            results['growth_image'],
                            results['segmented_image'],
                            results['depth_heatmap'],
                            results['edges'],
                            results['crack_details'],
                            results['material'],
                            results['probabilities'],
                            results['bio_growth_area'],
                            results['quantity_kg'],
                            results['carbon_footprint'],
                            results['water_footprint'],
                            predict_crack_progression(results['crack_details'])
                        )
                        if pdf_buffer:
                            st.session_state.pdf_buffer = pdf_buffer
                            st.success("✅ PDF report generated successfully!")
                        else:
                            st.error("❌ Failed to generate PDF report.")

            if st.session_state.pdf_buffer:
                st.download_button(
                    label="📥 Download PDF",
                    data=st.session_state.pdf_buffer,
                    file_name=f"Heritage_Analysis_Report_{st.session_state.image_name or 'report'}.pdf",
                    mime="application/pdf",
                    key="pdf_download_button"
                )

    with tab2:
        st.header("Upload and Analyze Heritage Site Videos")
        uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps)  # Process 1 frame per second
            frame_idx = 0
            frame_number = 0

            st.subheader("📽 Frame-by-Frame Analysis")
            st.session_state.video_frame_results = {}
            st.session_state.video_pdf_buffers = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_number += 1
                    image_np = frame.copy()
                    st.markdown(f"### 📸 Frame {frame_number}")
                    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_number}", use_container_width=True)

                    with st.spinner("🔄 Performing comprehensive analysis..."):
                        results = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("🔍 Detecting structural damage...")
                        progress_bar.progress(10)
                        annotated_image, crack_details = detect_with_yolo(image_np, px_to_cm_ratio)
                        results['crack_detection'] = (annotated_image, crack_details)

                        status_text.text("🧱 Analyzing building materials...")
                        progress_bar.progress(30)
                        material, probabilities = classify_material(image_np)
                        results['material_analysis'] = (material, probabilities)

                        status_text.text("🌿 Detecting biological growth...")
                        progress_bar.progress(50)
                        growth_image = detect_biological_growth(image_np, crack_details)
                        results['biological_growth'] = growth_image

                        status_text.text("📐 Performing segmentation...")
                        progress_bar.progress(70)
                        segmented_image, seg_results = segment_image(image_np)
                        results['segmentation'] = (segmented_image, seg_results)

                        status_text.text("📊 Generating depth and edge analysis...")
                        progress_bar.progress(80)
                        preprocessed = preprocess_image_for_depth_estimation(image_np)
                        depth_heatmap = create_depth_estimation_heatmap(preprocessed)
                        results['depth_analysis'] = depth_heatmap
                        edges = apply_canny_edge_detection(image_np)
                        results['edge_detection'] = edges

                        status_text.text("🌍 Calculating environmental impact...")
                        progress_bar.progress(90)
                        bio_growth_area = calculate_biological_growth_area(
                            crack_details, seg_results, image_np, px_to_cm_ratio
                        )
                        quantity_kg = estimate_material_quantity(crack_details, bio_growth_area, material)
                        carbon_footprint = calculate_carbon_footprint(material, quantity_kg)
                        water_footprint = calculate_water_footprint(material, quantity_kg)
                        results['environmental'] = (carbon_footprint, water_footprint, quantity_kg, bio_growth_area)

                        status_text.text("✅ Frame Analysis Complete")
                        progress_bar.progress(100)

                        # Store results in session state
                        st.session_state.video_frame_results[frame_number] = {
                            'image_np': image_np,
                            'crack_details': crack_details,
                            'material': material,
                            'probabilities': probabilities,
                            'bio_growth_area': bio_growth_area,
                            'carbon_footprint': carbon_footprint,
                            'water_footprint': water_footprint,
                            'quantity_kg': quantity_kg,
                            'seg_results': seg_results,
                            'annotated_image': annotated_image,
                            'growth_image': growth_image,
                            'segmented_image': segmented_image,
                            'depth_heatmap': depth_heatmap,
                            'edges': edges
                        }

                        # Display results
                        st.markdown("### 🔍 Frame Analysis Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                                     caption="Crack Detection", use_container_width=True)
                        with col2:
                            st.image(cv2.cvtColor(growth_image, cv2.COLOR_BGR2RGB),
                                     caption="Biological Growth", use_container_width=True)
                        with col3:
                            st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
                                     caption="Segmentation", use_container_width=True)

                        col4, col5 = st.columns(2)
                        with col4:
                            st.image(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB),
                                     caption="Depth Estimation", use_container_width=True)
                        with col5:
                            st.image(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB),
                                     caption="Edge Detection", use_container_width=True)

                        # Visualizations
                        st.markdown("### Data Visualizations")
                        total_area_cm2 = (image_np.shape[1] * px_to_cm_ratio) * (image_np.shape[0] * px_to_cm_ratio)
                        col6, col7 = st.columns(2)
                        with col6:
                            fig = plot_crack_severity(crack_details)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"crack_severity_chart_{frame_number}")
                        with col7:
                            fig = visualize_material_classification(material, probabilities)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"material_chart_{frame_number}")

                        col8, col9 = st.columns(2)
                        with col8:
                            fig = plot_biological_growth_area(bio_growth_area, total_area_cm2)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"growth_chart_{frame_number}")
                        with col9:
                            fig = plot_environmental_footprints(carbon_footprint, water_footprint)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"footprint_chart_{frame_number}")

                        # Summary
                        st.markdown("### Summary")
                        col10, col11, col12 = st.columns(3)
                        with col10:
                            st.metric("Dominant Material", material)
                            if crack_details:
                                st.write("*Crack Details:*")
                                for i, crack in enumerate(crack_details, 1):
                                    severity_color = {
                                        'Minor': '🟢',
                                        'Moderate': '🟡',
                                        'Severe': '🟠',
                                        'Critical': '🔴'
                                    }.get(crack['severity'], '⚪')
                                    st.write(f"{severity_color} Crack {i}: {crack['width_cm']:.2f} × {crack['length_cm']:.2f} cm - {crack['severity']}")
                            else:
                                st.info("✅ No structural damage detected")
                        with col11:
                            st.metric("Biological Growth Area", f"{bio_growth_area:.2f} cm²")
                            st.metric("Material Quantity", f"{quantity_kg:.2f} kg")
                        with col12:
                            st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
                            st.metric("Water Footprint", f"{water_footprint:.2f} liters")

                        # Predictive Analysis
                        st.subheader("📈 Predictive Analysis")
                        with st.expander("Crack Progression Forecast"):
                            prediction = predict_crack_progression(crack_details)
                            st.text(prediction)

                        # PDF Generation for Frame
                        st.subheader("📄 Download Frame Analysis Report")
                        with st.form(key=f"pdf_generate_form_frame_{frame_number}"):
                            submit_button = st.form_submit_button("Generate PDF Report")
                            if submit_button:
                                with st.spinner("📝 Generating PDF report..."):
                                    frame_results = st.session_state.video_frame_results[frame_number]
                                    pdf_buffer = generate_pdf_report(
                                        frame_results['image_np'],
                                        frame_results['annotated_image'],
                                        frame_results['growth_image'],
                                        frame_results['segmented_image'],
                                        frame_results['depth_heatmap'],
                                        frame_results['edges'],
                                        frame_results['crack_details'],
                                        frame_results['material'],
                                        frame_results['probabilities'],
                                        frame_results['bio_growth_area'],
                                        frame_results['quantity_kg'],
                                        frame_results['carbon_footprint'],
                                        frame_results['water_footprint'],
                                        predict_crack_progression(frame_results['crack_details'])
                                    )
                                    if pdf_buffer:
                                        st.session_state.video_pdf_buffers[frame_number] = pdf_buffer
                                        st.success("✅ PDF report generated successfully!")
                                    else:
                                        st.error("❌ Failed to generate PDF report.")

                        if frame_number in st.session_state.video_pdf_buffers:
                            st.download_button(
                                label="📥 Download PDF",
                                data=st.session_state.video_pdf_buffers[frame_number],
                                file_name=f"Heritage_Analysis_Frame_{frame_number}.pdf",
                                mime="application/pdf",
                                key=f"pdf_download_button_frame_{frame_number}"
                            )

                        progress_bar.empty()
                        status_text.empty()

                frame_idx += 1

            cap.release()
            os.unlink(tfile.name)

    with tab3:
        st.header("🌍 Environmental Footprints")
        st.markdown("Automatically calculated carbon and water footprints based on the latest image analysis.")
        if st.session_state.analysis_results is None:
            st.info("ℹ No analysis results available. Please perform an analysis in the Image Analysis tab.")
        else:
            results = st.session_state.analysis_results
            quantity_kg = results.get('quantity_kg', 0)
            carbon_footprint = results.get('carbon_footprint', 0)
            water_footprint = results.get('water_footprint', 0)
            material = results.get('material', 'Unknown')
            st.subheader("Footprint Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Material", material)
                st.metric("Estimated Quantity", f"{quantity_kg:.2f} kg")
            with col2:
                st.metric("Carbon Footprint", f"{carbon_footprint:.2f} kg CO2e")
                st.metric("Water Footprint", f"{water_footprint:.2f} liters")

    with tab4:
        st.header("ℹ About Heritage Sites Health Monitoring")
        st.markdown("""
        ### 🎯 Purpose
        This application aids heritage conservators in monitoring the structural health of historical buildings using AI.

        ### 🔧 Technologies Used
        - *YOLOv8*: Object detection and segmentation
        - *Computer Vision*: Advanced image processing
        - *Machine Learning*: Material classification
        - *Plotly*: Interactive visualizations
        - *ReportLab*: PDF report generation
        
        ### 📋 Features
        - *Automated Detection*: Identifies structural damage
        - *Material Analysis*: Recognizes building materials
        - *Biological Growth*: Detects moss and algae
        - *Depth Analysis*: 3D structural assessment
        - *Predictive Modeling*: Forecasts deterioration
        - *Environmental Impact*: Automatic carbon and water footprint analysis
        - *Visualization*: Interactive charts for analysis insights
        - *PDF Reports*: Downloadable analysis reports with images
        
        ### 🚀 How to Use
        1. Upload images or videos
        2. Adjust settings
        3. Analyze
        4. Review results and visualizations
        5. Generate and download PDF report
        
      
        """)

if __name__ == "__main__":
    main()