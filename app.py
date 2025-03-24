import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from roboflow import Roboflow
from ultralytics import YOLO

# Set page config
st.set_page_config(
    page_title="Solar Panel Detection",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #004d99;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("☀️ Solar Panel Detection System")
st.markdown("Upload infrared images to detect and analyze solar panels using YOLOv11.")

# Sidebar configuration
st.sidebar.header("Configuration")

# API Key input with default value
api_key = st.sidebar.text_input("Roboflow API Key", value="PvgDGjKs5OO4yA1LZf1R", type="password")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0, 100, 40)

# Model selection
model_type = st.sidebar.radio("Model Type", ["YOLOv11 (Roboflow)", "Custom YOLO Model"])

# Function to load the Roboflow model
@st.cache_resource
def load_roboflow_model(api_key):
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("solarpanels-qtoj6").project("solar_panel_classical_imgs_data")
        version = project.version(2)
        model = version.model
        return model, None
    except Exception as e:
        return None, str(e)

# Function to load local YOLO model
@st.cache_resource
def load_local_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# Function to process image with Roboflow model
def process_with_roboflow(model, image, confidence):
    # Save image temporarily
    temp_path = "temp_upload.jpg"
    image.save(temp_path)
    
    # Get predictions
    predictions = model.predict(temp_path, confidence=confidence).json()
    
    # Load image for visualization
    img = np.array(image)
    
    # Create figure and axis for visualization
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Count solar panels
    panel_count = 0
    
    # Draw bounding boxes
    if 'predictions' in predictions:
        for pred in predictions['predictions']:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            
            # Calculate box coordinates
            x_min = x - width/2
            y_min = y - height/2
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = pred['class']
            confidence = pred['confidence']
            label = f"{class_name}: {confidence:.2f}"
            ax.text(x_min, y_min-5, label, color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.5))
            
            panel_count += 1
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout()
    
    # Clean up
    os.remove(temp_path)
    
    return fig, panel_count, predictions

# Function to process image with local YOLO model
def process_with_yolo(model, image, confidence):
    # Convert confidence to 0-1 scale
    conf = confidence / 100
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Run prediction
    results = model(img_array, conf=conf)
    
    # Get the first result
    result = results[0]
    
    # Create figure for visualization
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Count solar panels
    panel_count = 0
    predictions = {"predictions": []}
    
    # Process results
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        
        # Get confidence and class
        conf_score = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_name}: {conf_score:.2f}"
        ax.text(x1, y1-5, label, color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5))
        
        panel_count += 1
        
        # Add to predictions
        predictions["predictions"].append({
            "x": x1 + width/2,
            "y": y1 + height/2,
            "width": width,
            "height": height,
            "confidence": conf_score,
            "class": class_name
        })
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout()
    
    return fig, panel_count, predictions

# Main content
tab1, tab2 = st.tabs(["Detection", "About"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Upload an infrared image of solar panels", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Detect Solar Panels"):
            with st.spinner("Processing..."):
                if model_type == "YOLOv11 (Roboflow)":
                    # Load Roboflow model
                    model, error = load_roboflow_model(api_key)
                    if error:
                        st.error(f"Error loading Roboflow model: {error}")
                    else:
                        # Process image
                        fig, panel_count, predictions = process_with_roboflow(model, image, confidence_threshold)
                else:
                    # For custom YOLO model, user should upload the model
                    model_path = st.sidebar.text_input("YOLO Model Path (if using custom model)", "best.pt")
                    model, error = load_local_yolo_model(model_path)
                    if error:
                        st.error(f"Error loading YOLO model: {error}")
                    else:
                        # Process image
                        fig, panel_count, predictions = process_with_yolo(model, image, confidence_threshold)
                
                # Display results
                if 'fig' in locals():
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.markdown(f"### Results: Detected {panel_count} solar panels")
                    st.pyplot(fig)
                    
                    # Show prediction details in expandable section
                    with st.expander("Show Detection Details"):
                        st.json(predictions)
                    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("About the Solar Panel Detection System")
    st.write("""
    This application uses computer vision to detect and analyze solar panels in infrared images. 
    The system is based on YOLOv11 trained on the "solar_panel_classical_imgs_data" dataset from Roboflow.
    
    ### Key Features:
    - Upload and analyze infrared images of solar panels
    - Detect solar panels with customizable confidence threshold
    - Visualize detection results with bounding boxes
    - Count the number of solar panels detected
    - Get detailed information about each detection
    
    ### How to Use:
    1. Configure your settings in the sidebar
    2. Upload an infrared image
    3. Click 'Detect Solar Panels'
    4. View the results and analysis
    
    ### Model Information:
    - Dataset: Solar Panel Classical Images Data (Version 2)
    - Model: YOLOv11
    - Workspace: solarpanels-qtoj6
    """)

# Run the app with: streamlit run app.py