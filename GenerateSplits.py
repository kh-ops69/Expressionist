import os, json
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# test_file_path = 'style_images/emotion_index_mapped_files/test_emo_list.txt'
from collections import defaultdict
from pathlib import Path
import pandas as pd

def create_emotionImage_dataset():
    my_dir_path = "style_images/emotion_index_mapped_files"
    emotion_mapping = {0:'amusement', 1:'excitement', 2:'awe', 3:'contentment', 4:'disgust', 5:'anger', 6:'fear', 7:'sadness'}
    total_length = 0
    results = {}
    for current_file_path in Path(my_dir_path).iterdir():
        with open(current_file_path, 'r') as file_open:
            result = file_open.readlines()
            results[current_file_path] = {}
            total_length += len(result)
            for index_emotion_str in result:
                index_emotion_split = index_emotion_str.split()
                i = index_emotion_split[0]
                results[current_file_path][i] = {}
                # results[i]['index'] = index_emotion_split[0]
                results[current_file_path][i]['emotion_numeric'] = int(index_emotion_split[1])
                results[current_file_path][i]['emotion_str'] = emotion_mapping[int(index_emotion_split[-1])]
                results[current_file_path][i]['image_path'] = str(current_file_path)+f'/{index_emotion_split[0]}.jpg'

    # pre processing to be carried out on image before inserting as numpy array - to experiment how resizing and resampling is going to affect
    # image readability when converted back into normal image for representation purposes.

    # Possible TODO: keep image preprocessing as part of dataframe insertion instead of inserting complete paths to images in dataframe

    # Define the target columns
    data_records = []

    # Iterate through the results dictionary and extract relevant information
    for folder, images_dict in results.items():
        for index, image_data in images_dict.items():
            # Append each record as a dictionary to the list
            data_records.append({
                'index': index,
                'emotion_numeric': image_data['emotion_numeric'],
                'emotion_str': image_data['emotion_str'],
                'image_path': image_data['image_path']
            })

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data_records)

    df.to_csv('image_emotions_dataset.csv', index=False)

dataset_path = 'image_emotions_dataset.csv'
n_samples = 300
selected_images_folder = '/Users/krishhashia/Expressionist/drive_content/selected_images'
os.makedirs(selected_images_folder, exist_ok=True)

# @anvil.server.callable
def return_random_image_partitions(train_opt):
    if train_opt == 'yes':
        dataset = pd.read_csv(dataset_path)
        emotions = dataset['emotion_str'].unique()
        emotion_image_path_mapping = {}
        for emotion in emotions:
            # Take a random sample of n_samples images for each emotion
            sample_df = dataset[dataset['emotion_str'] == emotion].sample(n_samples)
            emotion_folder = os.path.join(selected_images_folder, emotion)
            os.makedirs(emotion_folder, exist_ok=True)
            emotion_image_path_mapping[emotion] = []
            for image_index, image_path in enumerate(sample_df['image_path']):
                # Correct path to image files and load images as arrays
                image_path = image_path.replace('/emotion_index_mapped_files/train_emo_list.txt', '/images')\
                                    .replace('/emotion_index_mapped_files/val_emo_list.txt', '/images')\
                                    .replace('/emotion_index_mapped_files/test_emo_list.txt', '/images')\
                                    .replace('/style_transfer', '')
                img = Image.open(image_path)
                image_path = os.path.join(emotion_folder, f'image_{image_index}.png')
                img = img.resize((256, 256))
                img.save(image_path)
        return 'Training set generated'
    else:
        return 'Please retreive autoencoder summary and initialise layers in encoder and decoder first'

mapping = return_random_image_partitions('yes')
print(mapping)
