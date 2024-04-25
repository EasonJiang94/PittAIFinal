import os
import sys

def ckeck_folder_and_create(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Create folder: {folder}')
    else:
        print(f'Folder already exists: {folder}')