"""
PV Module Visual Defects Detection System
A comprehensive Streamlit application for detecting and analyzing defects in solar panels
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import tempfile
import os
import json
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="PV Module Defect Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .quality-pass {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .quality-fail {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

class ImageQualityChecker:
    """Handles image quality assessment"""
    
    @staticmethod
    def check_image_size(image: Image.Image, min_size: int = 640) -> Tuple[bool, str]:
        """Check if image meets minimum size requirements"""
        width, height = image.size
        if width < min_size or height < min_size:
            return False, f"Image size ({width}x{height}) is below minimum required ({min_size}x{min_size})"
        return True, f"Image size ({width}x{height}) meets requirements"
    
    @staticmethod
    def check_brightness(image: Image.Image) -> Tuple[bool, float, str]:
        """Check image brightness levels"""
        gray = image.convert('L')
        np_img = np.array(gray)
        brightness = np.mean(np_img)
        
        if brightness < 40:
            return False, brightness, "Image is too dark (brightness: {:.1f}/255)".format(brightness)
        elif brightness > 220:
            return False, brightness, "Image is overexposed (brightness: {:.1f}/255)".format(brightness)
        else:
            return True, brightness, "Brightness level is optimal ({:.1f}/255)".format(brightness)
    
    @staticmethod
    def check_contrast(image: Image.Image) -> Tuple[bool, float, str]:
        """Check image contrast"""
        gray = image.convert('L')
        np_img = np.array(gray)
        contrast = np.std(np_img)
        
        if contrast < 20:
            return False, contrast, "Image has low contrast ({:.1f})".format(contrast)
        return True, contrast, "Contrast level is good ({:.1f})".format(contrast)
    
    @staticmethod
    def check_sharpness(image: Image.Image) -> Tuple[bool, float, str]:
        """Check image sharpness using Laplacian variance"""
        gray = np.array(image.convert('L'))
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return False, laplacian_var, "Image is blurry (sharpness: {:.1f})".format(laplacian_var)
        return True, laplacian_var, "Image sharpness is good ({:.1f})".format(laplacian_var)
    
    @staticmethod
    def check_file_format(file_extension: str) -> Tuple[bool, str]:
        """Validate file format"""
        allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if file_extension.lower() in allowed_formats:
            return True, f"File format ({file_extension}) is supported"
        return False, f"File format ({file_extension}) is not supported. Use: {', '.join(allowed_formats)}"
    
    @staticmethod
    def perform_quality_check(image: Image.Image, filename: str) -> Dict:
        """Perform comprehensive quality check"""
        results = {
            'overall_status': True,
            'checks': {},
            'recommendations': []
        }
        
        # File format check
        file_ext = os.path.splitext(filename)[1]
        format_pass, format_msg = ImageQualityChecker.check_file_format(file_ext)
        results['checks']['file_format'] = {'passed': format_pass, 'message': format_msg}
        if not format_pass:
            results['overall_status'] = False
            results['recommendations'].append("Please upload an image in supported format")
        
        # Size check
        size_pass, size_msg = ImageQualityChecker.check_image_size(image)
        results['checks']['size'] = {'passed': size_pass, 'message': size_msg}
        if not size_pass:
            results['overall_status'] = False
            results['recommendations'].append("Upload a higher resolution image (min 640x640)")
        
        # Brightness check
        bright_pass, brightness_val, bright_msg = ImageQualityChecker.check_brightness(image)
        results['checks']['brightness'] = {'passed': bright_pass, 'value': brightness_val, 'message': bright_msg}
        if not bright_pass:
            results['overall_status'] = False
            if brightness_val < 40:
                results['recommendations'].append("Increase image brightness or retake in better lighting")
            else:
                results['recommendations'].append("Reduce exposure or retake in less bright conditions")
        
        # Contrast check
        contrast_pass, contrast_val, contrast_msg = ImageQualityChecker.check_contrast(image)
        results['checks']['contrast'] = {'passed': contrast_pass, 'value': contrast_val, 'message': contrast_msg}
        if not contrast_pass:
            results['overall_status'] = False
            results['recommendations'].append("Enhance image contrast or retake with better lighting angles")
        
        # Sharpness check
        sharp_pass, sharp_val, sharp_msg = ImageQualityChecker.check_sharpness(image)
        results['checks']['sharpness'] = {'passed': sharp_pass, 'value': sharp_val, 'message': sharp_msg}
        if not sharp_pass:
            results['overall_status'] = False
            results['recommendations'].append("Ensure camera focus is sharp and stable when capturing")
        
        return results

class RoboflowDetector:
    """Handles Roboflow API integration for defect detection"""
    
    def __init__(self, api_key: str, model_endpoint: str):
        self.api_key = api_key
        self.model_endpoint = model_endpoint
    
    def detect_defects(self, image: Image.Image, confidence: float = 0.4) -> Dict:
        """Send image to Roboflow API and get detection results"""
        try:
            # Convert image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            
            # Prepare API request
            response = requests.post(
                self.model_endpoint,
                params={
                    "api_key": self.api_key,
                    "confidence": confidence
                },
                files={"file": ("image.jpg", img_bytes, "image/jpeg")}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Detection failed: {str(e)}")
            return None
    
    def draw_detections(self, image: Image.Image, predictions: List[Dict]) -> Image.Image:
        """Draw bounding boxes and labels on image"""
        draw = ImageDraw.Draw(image)
        
        # Define colors for different defect classes
        class_colors = {
            'crack': '#FF0000',
            'hotspot': '#FF6600',
            'shading': '#FFD700',
            'soiling': '#8B4513',
            'bird_drop': '#808080',
            'delamination': '#FF1493',
            'snail_trail': '#9370DB',
            'default': '#00FF00'
        }
        
        for pred in predictions:
            # Get bounding box coordinates
            x = pred['x'] - pred['width'] / 2
            y = pred['y'] - pred['height'] / 2
            x2 = x + pred['width']
            y2 = y + pred['height']
            
            # Get color for class
            class_name = pred['class'].lower()
            color = class_colors.get(class_name, class_colors['default'])
            
            # Draw bounding box
            draw.rectangle([x, y, x2, y2], outline=color, width=3)
            
            # Draw label with confidence
            label = f"{pred['class']} ({pred['confidence']:.2f})"
            
            # Create label background
            text_bbox = draw.textbbox((x, y - 20), label)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, 
                          text_bbox[2] + 2, text_bbox[3] + 2], 
                          fill=color)
            draw.text((x, y - 20), label, fill='white')
        
        return image

class DataAnalyzer:
    """Handles data analysis and visualization"""
    
    @staticmethod
    def create_defect_summary(predictions: List[Dict]) -> pd.DataFrame:
        """Create summary statistics of detected defects"""
        if not predictions:
            return pd.DataFrame(columns=['Defect Class', 'Count', 'Avg Confidence', 'Min Confidence', 'Max Confidence'])
        
        df = pd.DataFrame(predictions)
        summary = df.groupby('class').agg({
            'confidence': ['count', 'mean', 'min', 'max']
        }).round(3)
        
        summary.columns = ['Count', 'Avg Confidence', 'Min Confidence', 'Max Confidence']
        summary = summary.reset_index()
        summary.rename(columns={'class': 'Defect Class'}, inplace=True)
        summary = summary.sort_values('Count', ascending=False)
        
        return summary
    
    @staticmethod
    def create_visualizations(predictions: List[Dict]) -> Dict:
        """Create various visualizations for defect data"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        figs = {}
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # 1. Bar chart of defect counts
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        defect_counts = df['class'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(defect_counts)))
        bars = ax1.bar(defect_counts.index, defect_counts.values, color=colors)
        ax1.set_xlabel('Defect Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Defect Distribution by Type', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        figs['bar_chart'] = fig1
        
        # 2. Pie chart of defect distribution
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(defect_counts)))
        wedges, texts, autotexts = ax2.pie(defect_counts.values, 
                                            labels=defect_counts.index,
                                            colors=colors,
                                            autopct='%1.1f%%',
                                            startangle=90,
                                            textprops={'fontweight': 'bold'})
        ax2.set_title('Defect Type Distribution', fontsize=14, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        figs['pie_chart'] = fig2
        
        # 3. Confidence score histogram
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.hist(df['confidence'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(df['confidence'].mean(), color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean: {df["confidence"].mean():.3f}')
        ax3.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution of Detection Confidence Scores', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        figs['confidence_hist'] = fig3
        
        # 4. Box plot of confidence by defect type
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        df_sorted = df.sort_values('class')
        box_data = [df_sorted[df_sorted['class'] == cls]['confidence'].values 
                   for cls in df_sorted['class'].unique()]
        bp = ax4.boxplot(box_data, labels=df_sorted['class'].unique(), 
                         patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.set_xlabel('Defect Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax4.set_title('Confidence Score Distribution by Defect Type', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        figs['box_plot'] = fig4
        
        return figs

class ReportGenerator:
    """Handles Excel and PDF report generation"""
    
    @staticmethod
    def export_to_excel(summary_df: pd.DataFrame, predictions: List[Dict], 
                       image_metadata: Dict, filename: str = "defect_report.xlsx") -> bytes:
        """Generate Excel report with defect data"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Summary Statistics
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed Predictions
            if predictions:
                detailed_df = pd.DataFrame(predictions)
                detailed_df.to_excel(writer, sheet_name='Detailed Predictions', index=False)
            
            # Sheet 3: Metadata
            metadata_df = pd.DataFrame([image_metadata])
            metadata_df.to_excel(writer, sheet_name='Image Metadata', index=False)
            
            # Format the Excel file
            workbook = writer.book
            
            # Format Summary sheet
            summary_sheet = workbook['Summary']
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            
            for cell in summary_sheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
            
            # Auto-adjust column widths
            for sheet in workbook.worksheets:
                for column in sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    sheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def export_to_pdf(original_image: Image.Image, processed_image: Image.Image,
                     summary_df: pd.DataFrame, charts: Dict, 
                     metadata: Dict, filename: str = "defect_report.pdf") -> bytes:
        """Generate comprehensive PDF report"""
        output = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(output, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title Page
        elements.append(Paragraph("PV Module Defect Detection Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Metadata section
        elements.append(Paragraph("Report Information", heading_style))
        
        metadata_data = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Image Name:", metadata.get('filename', 'N/A')],
            ["Image Dimensions:", f"{metadata.get('width', 'N/A')} x {metadata.get('height', 'N/A')}"],
            ["Total Defects Detected:", str(metadata.get('total_defects', 0))],
            ["Confidence Threshold:", str(metadata.get('confidence_threshold', 'N/A'))]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(metadata_table)
        elements.append(Spacer(1, 30))
        
        # Original Image
        elements.append(Paragraph("Original Image", heading_style))
        
        # Convert and resize image for PDF
        img_buffer = io.BytesIO()
        original_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img = RLImage(img_buffer, width=5*inch, height=3.75*inch)
        elements.append(img)
        elements.append(PageBreak())
        
        # Processed Image with Detections
        elements.append(Paragraph("Processed Image with Detected Defects", heading_style))
        
        img_buffer2 = io.BytesIO()
        processed_image.save(img_buffer2, format='PNG')
        img_buffer2.seek(0)
        img2 = RLImage(img_buffer2, width=5*inch, height=3.75*inch)
        elements.append(img2)
        elements.append(Spacer(1, 20))
        
        # Defect Summary Table
        elements.append(Paragraph("Defect Summary Statistics", heading_style))
        
        if not summary_df.empty:
            # Convert DataFrame to list of lists for table
            table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
            
            summary_table = Table(table_data, repeatRows=1)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(summary_table)
        else:
            elements.append(Paragraph("No defects detected", styles['Normal']))
        
        elements.append(PageBreak())
        
        # Visualizations
        if charts:
            elements.append(Paragraph("Data Visualizations", heading_style))
            
            for chart_name, fig in charts.items():
                # Save chart to buffer
                chart_buffer = io.BytesIO()
                fig.savefig(chart_buffer, format='PNG', dpi=150, bbox_inches='tight')
                chart_buffer.seek(0)
                
                # Add to PDF
                chart_img = RLImage(chart_buffer, width=6*inch, height=4*inch)
                elements.append(chart_img)
                elements.append(Spacer(1, 20))
                
                # Close figure to free memory
                plt.close(fig)
        
        # Build PDF
        doc.build(elements)
        
        output.seek(0)
        return output.getvalue()

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'quality_passed' not in st.session_state:
        st.session_state.quality_passed = False
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    
    # Title and description
    st.title("☀️ PV Module Visual Defects Detection System")
    st.markdown("---")
    st.markdown("""
    **Advanced AI-powered system for detecting and analyzing defects in photovoltaic modules**
    
    This application provides:
    - 📸 Automated image quality assessment
    - 🔍 AI-based defect detection using Roboflow
    - 📊 Comprehensive data visualization and insights
    - 📁 Professional reports in Excel and PDF formats
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("🔑 Roboflow API Settings")
        api_key = st.text_input("API Key", type="password", 
                                placeholder="Enter your Roboflow API key",
                                help="Get your API key from Roboflow dashboard")
        
        model_endpoint = st.text_input("Model Endpoint", 
                                       value="https://detect.roboflow.com/solar-panel-defects-evhvr-vxi6s/1",
                                       help="Roboflow model inference endpoint")
        
        # Detection Settings
        st.subheader("🎯 Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 
                                        min_value=0.1, 
                                        max_value=1.0, 
                                        value=0.4, 
                                        step=0.05,
                                        help="Minimum confidence score for detections")
        
        # Quality Check Settings
        st.subheader("✅ Quality Check Settings")
        min_image_size = st.number_input("Minimum Image Size (pixels)", 
                                         min_value=320, 
                                         max_value=1920, 
                                         value=640,
                                         help="Minimum width and height in pixels")
        
        st.markdown("---")
        st.info("💡 **Tip**: Higher confidence thresholds reduce false positives but may miss some defects")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("1️⃣ Upload PV Module Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a clear image of the PV module for defect detection"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Store image metadata
            image_metadata = {
                'filename': uploaded_file.name,
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode,
                'filesize': f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            # Quality Check Section
            st.header("2️⃣ Image Quality Assessment")
            
            with st.spinner("Performing quality checks..."):
                quality_results = ImageQualityChecker.perform_quality_check(image, uploaded_file.name)
            
            # Display quality check results
            if quality_results['overall_status']:
                st.success("✅ Image Quality: ACCEPTED")
                st.session_state.quality_passed = True
                
                # Show detailed checks
                with st.expander("View Quality Check Details"):
                    for check_name, check_result in quality_results['checks'].items():
                        if check_result['passed']:
                            st.write(f"✅ **{check_name.replace('_', ' ').title()}**: {check_result['message']}")
                        else:
                            st.write(f"❌ **{check_name.replace('_', ' ').title()}**: {check_result['message']}")
            else:
                st.error("❌ Image Quality: REJECTED")
                st.session_state.quality_passed = False
                
                # Show what failed
                st.write("**Issues Found:**")
                for check_name, check_result in quality_results['checks'].items():
                    if not check_result['passed']:
                        st.write(f"• {check_result['message']}")
                
                # Show recommendations
                if quality_results['recommendations']:
                    st.write("**Recommendations:**")
                    for rec in quality_results['recommendations']:
                        st.write(f"• {rec}")
    
    with col2:
        if uploaded_file is not None and st.session_state.quality_passed:
            st.header("3️⃣ Defect Detection")
            
            if st.button("🔍 Analyze Defects", type="primary"):
                if not api_key:
                    st.error("Please enter your Roboflow API key in the sidebar")
                else:
                    with st.spinner("Detecting defects... This may take a few moments"):
                        # Initialize detector
                        detector = RoboflowDetector(api_key, model_endpoint)
                        
                        # Perform detection
                        detection_results = detector.detect_defects(image, confidence_threshold)
                        
                        if detection_results:
                            st.session_state.detection_results = detection_results
                            predictions = detection_results.get('predictions', [])
                            
                            # Draw detections on image
                            processed_image = image.copy()
                            if predictions:
                                processed_image = detector.draw_detections(processed_image, predictions)
                            st.session_state.processed_image = processed_image
                            
                            # Display results
                            st.success(f"✅ Detection Complete! Found {len(predictions)} defect(s)")
                            st.image(processed_image, caption="Detected Defects", use_column_width=True)
                        else:
                            st.error("Detection failed. Please check your API key and endpoint.")
    
    # Results and Visualization Section
    if st.session_state.detection_results is not None:
        st.markdown("---")
        st.header("4️⃣ Analysis Results & Insights")
        
        predictions = st.session_state.detection_results.get('predictions', [])
        
        if predictions:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "📈 Visualizations", "💾 Export Reports"])
            
            with tab1:
                # Generate summary
                analyzer = DataAnalyzer()
                summary_df = analyzer.create_defect_summary(predictions)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Defects", len(predictions))
                with col2:
                    st.metric("Defect Types", summary_df['Defect Class'].nunique())
                with col3:
                    avg_conf = np.mean([p['confidence'] for p in predictions])
                    st.metric("Avg Confidence", f"{avg_conf:.2%}")
                with col4:
                    most_common = summary_df.iloc[0]['Defect Class'] if not summary_df.empty else "N/A"
                    st.metric("Most Common", most_common)
                
                # Display summary table
                st.subheader("Defect Summary Table")
                st.dataframe(summary_df, use_container_width=True)
                
                # Insights section
                st.subheader("📋 Key Insights")
                insights = []
                
                if len(predictions) > 5:
                    insights.append("⚠️ **High defect count detected** - Consider immediate maintenance")
                
                if avg_conf < 0.6:
                    insights.append("📉 **Low average confidence** - Consider re-imaging for better accuracy")
                
                if summary_df['Defect Class'].str.contains('crack|hotspot', case=False).any():
                    insights.append("🔴 **Critical defects found** - Cracks or hotspots require urgent attention")
                
                if insights:
                    for insight in insights:
                        st.write(insight)
                else:
                    st.write("✅ Module appears to be in acceptable condition")
            
            with tab2:
                # Generate visualizations
                st.subheader("Data Visualizations")
                
                with st.spinner("Generating visualizations..."):
                    charts = analyzer.create_visualizations(predictions)
                
                if charts:
                    # Display charts in a grid
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.pyplot(charts['bar_chart'])
                        st.pyplot(charts['confidence_hist'])
                    
                    with col2:
                        st.pyplot(charts['pie_chart'])
                        st.pyplot(charts['box_plot'])
            
            with tab3:
                st.subheader("Export Options")
                
                # Prepare metadata for export
                export_metadata = {
                    'filename': uploaded_file.name if uploaded_file else 'N/A',
                    'width': image.width if 'image' in locals() else 'N/A',
                    'height': image.height if 'image' in locals() else 'N/A',
                    'total_defects': len(predictions),
                    'confidence_threshold': confidence_threshold,
                    'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("📊 **Excel Report**")
                    st.write("Export detailed defect data and statistics to Excel format")
                    
                    if st.button("Generate Excel Report", type="secondary"):
                        with st.spinner("Generating Excel report..."):
                            generator = ReportGenerator()
                            excel_data = generator.export_to_excel(
                                summary_df, 
                                predictions, 
                                export_metadata
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_data,
                                file_name=f"pv_defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                with col2:
                    st.write("📄 **PDF Report**")
                    st.write("Generate comprehensive PDF report with images and charts")
                    
                    if st.button("Generate PDF Report", type="secondary"):
                        with st.spinner("Generating PDF report... This may take a moment"):
                            generator = ReportGenerator()
                            
                            # Generate charts again for PDF
                            charts = analyzer.create_visualizations(predictions)
                            
                            pdf_data = generator.export_to_pdf(
                                image,
                                st.session_state.processed_image,
                                summary_df,
                                charts,
                                export_metadata
                            )
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_data,
                                file_name=f"pv_defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
        else:
            st.info("No defects detected in the image. The PV module appears to be in good condition!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>PV Module Defect Detection System v1.0</p>
        <p>Powered by Roboflow AI | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
