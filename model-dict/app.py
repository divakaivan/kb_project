import os
import glob
from PIL import Image
import streamlit as st

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def read_png_metadata(file_path):
    image = Image.open(file_path)
    info = image.info
    return info

directory_path = 'mage_orchestration'

folders = glob.glob(os.path.join(directory_path, 'models_*'))

models = {}

for folder in folders:
    if os.path.isdir(folder):
        model_name = os.path.basename(folder)
        models[model_name] = {'location': folder, 'text_files': [], 'images': []}
        
        txt_files = glob.glob(os.path.join(folder, '*.txt'))
        png_files = glob.glob(os.path.join(folder, '*.png'))
        
        for txt_file in txt_files:
            txt_content = read_txt_file(txt_file)
            models[model_name]['text_files'].append({
                'title': os.path.splitext(os.path.basename(txt_file))[0],
                'content': txt_content,
                'path': txt_file
            })
        
        for png_file in png_files:
            png_metadata = read_png_metadata(png_file)
            models[model_name]['images'].append({
                'title': os.path.splitext(os.path.basename(png_file))[0],
                'path': png_file
            })


selected_model = st.selectbox("Select Model", list(models.keys()))
model_info = models[selected_model]

st.markdown(f"# {selected_model}")
st.markdown(f"### Location \n{model_info['location']}")

for text_file in model_info["text_files"]:
    st.markdown(f"### {text_file['title']}")
    if text_file['title'] == 'feature_importance':
        st.text_area("Feature Importance", text_file['content'], height=300)
    else:
        st.code(text_file['content'])

for image in model_info["images"]:
    st.markdown(f"### {image['title']}")
    st.image(image['path'])
