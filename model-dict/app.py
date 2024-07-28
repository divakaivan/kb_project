import streamlit as st

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# load model artifacts from mlflow if artifact store is available
models = {
    "GCN": {
        "title": "Graph Convolutional Network (GCN)",
        "location": "mage_orchestration/models_20240728_050159",
        "images": [
            {
                "path": "mage_orchestration/models_20240728_050159/conf_matrix.png",
                "title": "Confusion Matrix"
            },
            {
                "path": "mage_orchestration/models_20240728_050159/feature_importance.png",
                "title": "Feature Importance"
            }
        ],
        "text_files": [
            {
                "path": "mage_orchestration/models_20240728_050159/model_summary.txt",
                "title": "Summary"
            },
            {
                "path": "mage_orchestration/models_20240728_050159/metrics.txt",
                "title": "Metrics"
            }
        ]
    },
    # add more models
}

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
model_info = models[selected_model]

st.markdown(f"# {model_info['title']}")
st.markdown(f"### Location \n{model_info['location']}")

for text_file in model_info["text_files"]:
    st.markdown(f"### {text_file['title']}")
    st.code(load_text(text_file["path"]))

for image in model_info["images"]:
    st.markdown(f"### {image['title']}")
    st.image(image["path"])
