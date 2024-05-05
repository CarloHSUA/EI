from ResGCNv1 import create as ResGCN
from st_gcn.graph import Graph

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from rich import print

from CustomDataset import CustomDataset
from ResGCNV2 import ResGCNV2

import json
import numpy as np
import os


torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)


def open_json(filename):
    '''
    Open a JSON file with the metadata
    '''
    with open(filename) as f:
        data = json.load(f)
    return data


def load_weight(model, weight_path):
    '''
    Load the weights of the model, the weights are loaded from the weight_path
    '''
    weights = torch.load(weight_path, map_location=device)
    weights = {i.replace('backbone.',''):weights['state_dict'][i] for i in weights['state_dict'].keys()}
    print(len(weights.keys()))
    model.load_state_dict(weights, strict=False)
    return model


def preprocess_data(json_path, eval_method = 'subject-level'):
    '''
    Preprocess the data to be used in the model
    '''
    assert eval_method in ['subject-level', 'run-level'], 'Evaluation method must be either "subject-level" or "run-level"'

    data = open_json(json_path)
    keys = list(data.keys())
    percentage = int(len(keys) * 0.8)
    
    if eval_method == 'subject-level':
        np.random.shuffle(keys)
        data = {str(idx): data[key] for idx, key in enumerate(keys)}

    keys_selected_train = keys[:percentage] 
    keys_selected_test = keys[percentage:]

    data_train = {str(idx): data[key] for idx, key in enumerate(keys_selected_train)}
    data_test = {str(idx): data[key] for idx, key in enumerate(keys_selected_test)}

    return data_train, data_test


def moving_average(losses, window_size):
    '''
    Calculate the moving average of a list of losses
    '''
    losses = torch.tensor(losses)
    weights = torch.ones(window_size) / window_size
    return torch.conv1d(losses.view(1, 1, -1), weights.view(1, 1, -1), padding=window_size - 1).squeeze()


def test_prediction(extended_model, dataloader_skeleton_test, criterion):
    '''
    Test the model with the test dataset
    '''
    extended_model.eval()
    accuracy = []
    f1_scores = []  # Lista para almacenar los valores de F1-score
    global_loss = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader_skeleton_test):
            data = data.to(device)
            label = label.to(device)
            out, _ = extended_model(data)
            loss = criterion(out, label)
            
            pred = torch.argmax(out, dim=1)
            acc = torch.sum(pred == label).item() / len(label)
            accuracy.append(acc)
            global_loss.append(loss.item())

            f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='weighted')
            f1_scores.append(f1)

            print(f"Batch [{idx+1}/{len(dataloader_skeleton_test)}], Loss: {loss.item()}, Accuracy: {acc}, F1-Score: {f1}")
    
    print(f"Average Accuracy: {sum(accuracy) / len(accuracy)}, Average F1-Score: {sum(f1_scores) / len(f1_scores)}")
    


def show_result(global_accuracy, global_f1_scores, global_loss):
    '''
    Show the result of the training
    '''
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(global_accuracy, label='Accuracy')
    axs[0].plot(global_f1_scores, label='F1-Score')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].set_title('Accuracy and F1-Score')

    axs[1].plot(moving_average(global_loss, 25), label='Loss')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].set_title('Loss')
    plt.show()


if __name__=='__main__':

    # load the graph
    graph = Graph('coco')

    # load the model
    model_args = {
        "A": torch.tensor(graph.A, dtype=torch.float32, requires_grad=False),
        "num_class": 128,          # 128 output classes
        "num_input": 3,            # 3D coordinates
        "num_channel": 5,          
        "parts": graph.parts,
    }
    # model = StGCN(in_channels=3, graph=graph, edge_importance_weighting=True, temporal_kernel_size=9, embedding_layer_size=128).to(device)
    model = ResGCN('resgcn-n21-r8', **model_args)
    model = load_weight(model, 'model/gaitgraph-casia-b-epoch=69-val_loss_epoch=0.92.ckpt')

    extended_model = ResGCNV2(model).to(device)
    print("Extended model: ", extended_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Load the data
    data_train, data_test = preprocess_data('data/skeletons_metadata_new.json', eval_method='run-level')

    dataset_train = CustomDataset(data = data_train, mode = 'skeleton', label_metric='GHQ_Label')
    dataset_test = CustomDataset(data = data_test, mode = 'skeleton', label_metric='GHQ_Label')

    dataloader_skeleton_train = DataLoader(dataset_train,
                                        batch_size=64,
                                        shuffle=True)
    
    dataloader_skeleton_test = DataLoader(dataset_test,
                                        batch_size=64,
                                        shuffle=True)

    # Check if the model is already trained
    if os.path.exists('model/extended_model.pth'):
        extended_model.load_state_dict(torch.load('model/extended_model.pth'))
        test_prediction(extended_model, dataloader_skeleton_test, criterion)

    else:
        # Train model
        global_accuracy = []
        global_f1_scores = []
        global_loss = []

        extended_model.train()
        for epoch in range(10):
            accuracy = []
            f1_scores = []  # Lista para almacenar los valores de F1-score
            for idx, (data, label) in enumerate(dataloader_skeleton_train):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                out, embedding = extended_model(data)

                # print(out, label)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                
                # Calcula la precisión
                pred = torch.argmax(out, dim=1)
                # print(pred, '\n', label)
                acc = torch.sum(pred == label).item() / len(label)
                accuracy.append(acc)

                # Calcula el F1-score
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                f1_scores.append(f1)
                
                # Media movil de la perdida
                global_loss.append(loss.item())
                


                # Imprime métricas
                print(f"Epoch [{epoch+1}/{10}], Batch [{idx+1}/{len(dataloader_skeleton_train)}], Loss: {loss.item()}, Accuracy: {acc}, F1-Score: {f1}")

            # Calcula la precisión promedio por época
            avg_accuracy = sum(accuracy) / len(accuracy)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            global_accuracy.append(avg_accuracy)
            global_f1_scores.append(avg_f1)
            print(f"Epoch [{epoch+1}/{10}], Average Accuracy: {avg_accuracy}, Average F1-Score: {avg_f1}\n")


        # save the model
        torch.save(extended_model.state_dict(), 'model/extended_model.pth')


        # Show the result
        show_result(global_accuracy, global_f1_scores, global_loss)


        print("Start testing the model...")
        test_prediction(extended_model, dataloader_skeleton_test, criterion)
        