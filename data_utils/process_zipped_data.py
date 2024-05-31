import os
import shutil
from zipfile import ZipFile

script_dir = os.path.dirname(os.path.abspath(__file__))
# Define paths
source_dir = os.path.join(
    script_dir,
    "..",
    "DataDownloader",
    "data",
    "futures",
    "um",
    "monthly",
    "klines",
    "BTCUSDT",
    "5m",
    "2024-01-01_2024-04-30",
)
downlaod_dir = os.path.join(script_dir, "..", "DataDownloader", "data")
zip_dir = os.path.join(script_dir, ".", "ZippedFiles")
unzip_dir = os.path.join(script_dir, "..", "data")

# Create new folders if they do not exist
os.makedirs(zip_dir, exist_ok=True)
os.makedirs(unzip_dir, exist_ok=True)

# Move .zip files to the zip_dir
for file_name in os.listdir(source_dir):
    if file_name.endswith(".zip"):
        shutil.move(os.path.join(source_dir, file_name), zip_dir)

# Unzip all files in the zip_dir
for file_name in os.listdir(zip_dir):
    if file_name.endswith(".zip"):
        zip_path = os.path.join(zip_dir, file_name)
        extract_path = os.path.join(unzip_dir, os.path.splitext(file_name)[0])
        os.makedirs(extract_path, exist_ok=True)
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

# Remove the zip_dir
shutil.rmtree(zip_dir)

# Remove the download_dir
shutil.rmtree(downlaod_dir)

# Move all CSV files to the destination directory
for root, dirs, files in os.walk(unzip_dir):
    for file in files:
        if file.endswith(".csv"):
            source_file = os.path.join(root, file)
            destination_file = os.path.join(unzip_dir, file)
            shutil.move(source_file, destination_file)

# Remove empty directories
for root, dirs, files in os.walk(unzip_dir, topdown=False):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        try:
            os.rmdir(dir_path)
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")


print("All files have been moved and unzipped successfully.")
