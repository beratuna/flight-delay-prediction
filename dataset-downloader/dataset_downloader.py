import os
import requests
from pyunpack import Archive


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Downloading the dataset.rar from drive
print("Dowloading the dataset.rar file from drive.google.com, please wait..")
file_id = '1FKElsHgrp0v-uXzgaTIC2D7Q5CeCf7JL'
destination = ROOT_DIR + '\\dataset.rar'
download_file_from_google_drive(file_id, destination)
print("The dataset.rar file has downloaded successfully!")

# Extracting the dataset.rar file
print("Extracting the dataset.rar file, please wait..")
Archive("dataset.rar").extractall(ROOT_DIR)
print("The datasets have extracted successfully!")
