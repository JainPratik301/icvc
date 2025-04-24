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


