import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os


def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_image = cv2.Canny(gray, 100, 200)
    return edges_image

def median_blur(image):
    median_blurred_image = cv2.medianBlur(image, 15)
    return median_blurred_image

def erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

def dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

def inpainting(image):
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_image

def bilateral_filter(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    return bilateral_filtered_image

def denoising(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    color_filtered_image = cv2.bitwise_and(image, image, mask=mask)
    return color_filtered_image

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh_image

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 11, 2)
    return adaptive_threshold_image

def otsu_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

def border(image):
    bordered_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 255])
    return bordered_image


st.set_page_config(page_title="Image Filters", layout="wide")

def upload_page():
    st.title("Upload Images")
    uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
   
    if uploaded_files:
        images = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            images.append((uploaded_file.name, image_np))
       
        st.session_state['images'] = images
        st.success("Images uploaded successfully!")
        st.button("Apply Filters", on_click=lambda: st.session_state.update({"page": 2}))

def filter_page():
    st.title("Apply Filters to Images")
   
    if 'images' not in st.session_state:
        st.warning("Please upload images first.")
        return

    images = st.session_state['images']
   
    filter_options = [
        "Grayscale", "Gaussian Blur", "Edge Detection", "Median Blur", "Erosion", "Dilation",
        "Inpainting", "Bilateral Filter", "Denoising", "Color Filter", "Thresholding",
        "Adaptive Thresholding", "Otsu Thresholding", "Border"
    ]
   
    selected_filter = st.selectbox("Select a filter", filter_options)
    apply_button = st.button("Apply Filter")
   
    if apply_button:
        for image_name, image in images:
            st.image(image, caption=f"Original Image: {image_name}")
            if selected_filter == "Grayscale":
                filtered_image = grayscale(image)
            elif selected_filter == "Gaussian Blur":
                filtered_image = gaussian_blur(image)
            elif selected_filter == "Edge Detection":
                filtered_image = edge_detection(image)
            elif selected_filter == "Median Blur":
                filtered_image = median_blur(image)
            elif selected_filter == "Erosion":
                filtered_image = erosion(image)
            elif selected_filter == "Dilation":
                filtered_image = dilation(image)
            elif selected_filter == "Inpainting":
                filtered_image = inpainting(image)
            elif selected_filter == "Bilateral Filter":
                filtered_image = bilateral_filter(image)
            elif selected_filter == "Denoising":
                filtered_image = denoising(image)
            elif selected_filter == "Color Filter":
                filtered_image = color_filter(image)
            elif selected_filter == "Thresholding":
                filtered_image = thresholding(image)
            elif selected_filter == "Adaptive Thresholding":
                filtered_image = adaptive_thresholding(image)
            elif selected_filter == "Otsu Thresholding":
                filtered_image = otsu_thresholding(image)
            elif selected_filter == "Border":
                filtered_image = border(image)
           
            st.image(filtered_image, caption=f"Filtered Image: {image_name}")


if 'page' not in st.session_state:
    st.session_state['page'] = 1

if st.session_state['page'] == 1:
    upload_page()
else:
    filter_page()


import streamlit as st

st.set_page_config(page_title="IMAGE FILTERING AND CAPTION GENERATOR", layout="wide")

# Set a formal light background color (light beige or light blue)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F5DC;  /* Light Beige background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("IMAGE FILTER AND CAPTION GENERATOR")

st.markdown("""Designed with simplicity in mind, allowing users to effortlessly apply filters and generate captions with a seamless interaction.


Image Enhancement with Filters: Provides a range of image processing filters,
 including grayscale, blur, edge detection, and more, to modify and improve the quality of images.



Automatic Caption Generation: Leverages AI technology to automatically generate descriptive captions for uploaded images, providing insights into their content.
""")

app_choice = st.radio("Select an Application:", ["IMAGE CAPTION GENERATOR", "IMAGE FILTERING"])

if app_choice == "IMAGE CAPTION GENERATOR":
    st.markdown("[IMAGE CAPTION GENERATOR APPLICATION](https://blank-app-til8dyg4qve.streamlit.app/)")

elif app_choice == "IMAGE FILTERING":
    st.markdown("[IMAGE FILTERING APPLICATION](https://blank-app-rtfferrpi0g.streamlit.app/)")


