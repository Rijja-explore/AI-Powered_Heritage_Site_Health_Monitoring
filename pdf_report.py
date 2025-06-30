from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

def save_image_to_temp(image_np, prefix="image"):
    """Save numpy image to a temporary PNG file and return the path."""
    try:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=prefix) as temp_file:
            pil_image.save(temp_file.name, format='PNG')
            return temp_file.name
    except Exception as e:
        st.error(f"❌ Failed to save image to temporary file: {str(e)}")
        return None

def generate_pdf_report(image_np, annotated_image, growth_image, segmented_image, depth_heatmap, edges, crack_details, material, probabilities, bio_growth_area, quantity_kg, carbon_footprint, water_footprint, prediction):
    """Generate a PDF report using reportlab with only altered images included."""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch, bottomMargin=inch/2)
        elements = []
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        caption_style = ParagraphStyle(name='Caption', parent=normal_style, fontSize=8, alignment=1)
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])

        temp_files = []

        elements.append(Paragraph("Heritage Sites Health Monitoring Report", title_style))
        elements.append(Spacer(1, 0.5*inch))
        image_name = st.session_state.get('image_name', 'Unknown')
        temp_path = save_image_to_temp(image_np, "original")
        if temp_path:
            img = ReportLabImage(temp_path, width=4*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Paragraph("Original Image", caption_style))
            temp_files.append(temp_path)
        else:
            elements.append(Paragraph("Failed to include the original image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        elements.append(Paragraph(f"Image: {image_name}", normal_style))
        elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Spacer(1, 2*inch))

        elements.append(Paragraph("Introduction", heading_style))
        elements.append(Paragraph("This report provides a comprehensive analysis of the structural health of a heritage site based on the provided image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Image Analysis", heading_style))

        elements.append(Paragraph("Crack Detection", heading_style))
        crack_data = [['Crack', 'Width (cm)', 'Length (cm)', 'Severity', 'Confidence']]
        if crack_details:
            for i, crack in enumerate(crack_details, 1):
                severity = crack['severity'] if crack['severity'] else 'N/A'
                crack_data.append([f"Crack {i}", f"{crack['width_cm']:.2f}", f"{crack['length_cm']:.2f}", severity, f"{crack['confidence']:.2f}"])
        else:
            crack_data.append(['No cracks detected', '-', '-', '-', '-'])
        crack_table = Table(crack_data)
        crack_table.setStyle(table_style)
        elements.append(crack_table)
        temp_path = save_image_to_temp(annotated_image, "annotated")
        if temp_path:
            img = ReportLabImage(temp_path, width=4*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Paragraph("Crack Detection Results", caption_style))
            temp_files.append(temp_path)
        else:
            elements.append(Paragraph("Failed to include crack detection image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Material Classification", heading_style))
        elements.append(Paragraph(f"Dominant material: {material}", normal_style))
        material_data = [['Material', 'Confidence Score']]
        material_classes = ['Stone', 'Brick', 'Plaster', 'Concrete', 'Wood', 'Metal', 'Marble', 'Sandstone']
        for m, p in zip(material_classes, probabilities):
            material_data.append([m, f"{p:.2f}"])
        material_table = Table(material_data)
        material_table.setStyle(table_style)
        elements.append(material_table)
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Biological Growth Detection", heading_style))
        elements.append(Paragraph(f"Total biological growth area: {bio_growth_area:.2f} cm²", normal_style))
        temp_path = save_image_to_temp(growth_image, "growth")
        if temp_path:
            img = ReportLabImage(temp_path, width=4*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Paragraph("Biological Growth Detection", caption_style))
            temp_files.append(temp_path)
        else:
            elements.append(Paragraph("Failed to include biological growth image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Image Segmentation", heading_style))
        temp_path = save_image_to_temp(segmented_image, "segmented")
        if temp_path:
            img = ReportLabImage(temp_path, width=4*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Paragraph("Image Segmentation Results", caption_style))
            temp_files.append(temp_path)
        else:
            elements.append(Paragraph("Failed to include segmentation image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Depth Analysis", heading_style))
        temp_path = save_image_to_temp(depth_heatmap, "depth")
        if temp_path:
            img = ReportLabImage(temp_path, width=4*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Paragraph("Depth Estimation Heatmap", caption_style))
            temp_files.append(temp_path)
        else:
            elements.append(Paragraph("Failed to include depth estimation image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Edge Detection", heading_style))
        temp_path = save_image_to_temp(edges, "edges")
        if temp_path:
            img = ReportLabImage(temp_path, width=4*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Paragraph("Edge Detection Results", caption_style))
            temp_files.append(temp_path)
        else:
            elements.append(Paragraph("Failed to include edge detection image.", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Environmental Footprints", heading_style))
        footprint_data = [
            ['Metric', 'Value'],
            ['Material Quantity', f"{quantity_kg:.2f} kg"],
            ['Carbon Footprint', f"{carbon_footprint:.2f} kg CO2e"],
            ['Water Footprint', f"{water_footprint:.2f} liters"]
        ]
        footprint_table = Table(footprint_data)
        footprint_table.setStyle(table_style)
        elements.append(footprint_table)
        elements.append(Spacer(1, 0.25*inch))

        elements.append(Paragraph("Predictive Analysis", heading_style))
        elements.append(Paragraph("Crack Progression Forecast", heading_style))
        prediction_lines = prediction.split('\n')
        for line in prediction_lines:
            elements.append(Paragraph(line, normal_style))
        elements.append(Spacer(1, 0.25*inch))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"❌ PDF generation failed: {str(e)}")
        return None
    finally:
        for temp_path in temp_files:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass