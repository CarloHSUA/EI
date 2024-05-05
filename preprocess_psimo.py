import pandas as pd
import json
import os
from rich import print


# Constants variables
EVALUATION_METHODS = ['BFI', 'RSE', 'BPAQ', 'OFER', 'DASS', 'GHQ']


# Variables
silhouettes_json = {}
skeleton_json = {}
labels = []


# Save a JSON file with the metadata of the silouette data
def save_metadata(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


# Open a JSON file with the metadata
def open_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

mode = '_new'

# Create a DataFrame from the silouette data
metadata_labels = pd.read_csv(f'psimo_reduced/metadata_labels_v3{mode}.csv')
walks_labels = pd.read_csv(f'psimo_reduced/walks-v2{mode}.csv')
print("Length metadata_labels: ", len(metadata_labels))
print("Length walks_labels: ", len(walks_labels))
labels_merged = pd.merge(metadata_labels, walks_labels, on='ID', how='inner')

del metadata_labels, walks_labels


if mode != '_new':
    labels_merged = labels_merged[labels_merged['ID'].isin(range(0, 10))]


for col in labels_merged.columns:
    if labels_merged[col].dtype == 'object' and col != 'file_id':
        value_dict = {value: idx for idx, value in enumerate(sorted(labels_merged[col].unique()))}
        labels_merged[col] = labels_merged[col].map(value_dict)
        print(col, value_dict)


print(labels_merged.head())
print("Length: ", len(labels_merged))

# ATTR_Gender =                   {'F': 0, 'M': 1}
# BFI_Openness_Label =            {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BFI_Conscientiousness_Label =   {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BFI_Extraversion_Label =        {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BFI_Agreeableness_Label =       {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BFI_Neuroticism_Label =         {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# RSE_Label =                     {'High': 0, 'Low': 1, 'Normal': 2}
# BPAQ_Hostility_Label =          {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BPAQ_VerbalAggression_Label =   {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BPAQ_Anger_Label =              {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# BPAQ_PhysicalAggression_Label = {'High': 0, 'Low': 1, 'Normal / High': 2, 'Normal / Low': 3}
# DASS_Depression_Label =         {'Extremely Severe': 0, 'Mild': 1, 'Moderate': 2, 'Normal': 3, 'Severe': 4}
# DASS_Anxiety_Label =            {'Extremely Severe': 0, 'Mild': 1, 'Moderate': 2, 'Normal': 3, 'Severe': 4}
# DASS_Stress_Label =             {'Extremely Severe': 0, 'Mild': 1, 'Moderate': 2, 'Normal': 3, 'Severe': 4}
# GHQ_Label =                     {'Major Distress': 0, 'Minor Distress': 1, 'Typical': 2}
# OFER_ChronicFatigue_Label =     {'High': 0, 'Low': 1, 'Moderate / High': 2, 'Moderate / Low': 3}
# OFER_AcuteFatigue_Label =       {'High': 0, 'Low': 1, 'Moderate / High': 2, 'Moderate / Low': 3}
# OFER_Recovery_Label =           {'High': 0, 'Low': 1, 'Moderate / High': 2, 'Moderate / Low': 3}
# variation =                     {'bg': 0, 'cl': 1, 'nm': 2, 'ph': 3, 'txt': 4, 'wsf': 5, 'wss': 6}



for palabra in EVALUATION_METHODS:
    for col in labels_merged.columns:
        if col.startswith(palabra):
            labels.append(col)


# Save the metadata of the silouette data
def save_silhouettes_metadata(labels_merged, silhouettes_json, labels):

    for idx, row in enumerate(labels_merged.iterrows()):
        silhouettes_json[idx] = {}
        files = os.listdir(f'psimo_reduced/semantic_data{mode}/silhouettes/{row[1]['ID']}/{row[1]['file_id']}')
        silhouettes_json[idx]['data_dir'] = f'silhouettes/{row[1]['ID']}/{row[1]['file_id']}'
        silhouettes_json[idx]['data'] = sorted(files, key=lambda x: int(x.split('.')[0]))
        silhouettes_json[idx]['labels'] = {label: row[1][label] for label in labels}

    save_metadata(silhouettes_json, f'data/silhouettes_metadata{mode}.json')



# Save the metadata of skeleton data
def save_skeleton_metadata(labels_merged, skeleton_json, labels):

    for idx, row in enumerate(labels_merged.iterrows()):
        skeleton_json[idx] = {}
        keypoints_list = []
        skeleton = open_json(f'psimo_reduced/semantic_data{mode}/skeletons/{row[1]['ID']}/{row[1]['file_id']}.json')

        for idx_2 in range(len(skeleton)):
            points = skeleton[idx_2]['keypoints']
            keypoints_list.append(points)


        skeleton_json[idx]['data_dir'] = f'skeletons/{row[1]['ID']}/{row[1]['file_id']}.json'
        skeleton_json[idx]['data'] = keypoints_list # transform_val(data).tolist()
        skeleton_json[idx]['labels'] = {label: row[1][label] for label in labels}
        

    save_metadata(skeleton_json, f'data/skeletons_metadata{mode}.json')


if __name__ == '__main__':
    save_silhouettes_metadata(labels_merged, silhouettes_json, labels)
    print('Silhouettes data saved successfully')

    save_skeleton_metadata(labels_merged, skeleton_json, labels)
    print('Skeletons data saved successfully')