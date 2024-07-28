import os
import kaggle

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

download_path = "datasets"

@data_loader
def load_data(*args, **kwargs):
    dataset = "kartik2112/fraud-detection"
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    files_exist = any(os.scandir(download_path))
    
    if not files_exist:
        kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
    
    # Collect all file paths in the download_path directory
    file_paths = []
    for root, _, files in os.walk(download_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    return file_paths

@test
def test_output(output, *args) -> None:
    output is not None, 'Output is None'