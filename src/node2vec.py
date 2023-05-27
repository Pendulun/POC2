from __future__ import annotations
import pathlib

from data_loader import IdBasedGraphDataLoader
from model_wrappers import ModelWrapper
from graph_embedder import GraphsEmbedder

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric import seed_everything


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class Node2VecModelWrapper(ModelWrapper):
    """
    This is the Node2Vec PyGeometric model wrapper
    """
    def __init__(self, base_params:dict):
        super().__init__(base_params)

    def _construct_model(self, base_params:dict) -> Node2Vec:
        return Node2Vec(**base_params)
    
    def get_specific_loader(self, batch_size:int=128, 
                            shuffle:bool=True) -> DataLoader:
        return self.model.loader(batch_size=batch_size,
                                  shuffle=shuffle)
    
    def train(self, loader:DataLoader, optimizer:Optimizer,
              criterion:Module, epochs:int, device:str):
        """
        Trains the Node2Vec model saving the final_train_loss.
        The criterion param is ignored as Node2Vec has its own
        """
        for epoch in range(epochs):
            loss = self._model_train(loader, optimizer, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            self._final_train_loss = loss
    
    #From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
    # and https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial11/Tutorial11.ipynb
    def _model_train(self, loader:DataLoader, optimizer:Optimizer, 
                     device:str) -> float:
        self._model.train()
        total_loss = 0
        num_loaded = 0
        for pos_rw, neg_rw in loader:
            num_loaded+=1
            optimizer.zero_grad()
            loss = self._model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    #Preciso testar oq? Não tenho divisão de treino e teste
    def test(model:Node2Vec, data):
        model.eval()
        with torch.inference_mode():
            z = model()
            acc = model.test(z[data.train_mask], data.y[data.train_mask],
                                z[data.test_mask], data.y[data.test_mask],
                                max_iter=150)
        return acc

def plot_embeddings(embeddings_list):
    embed_np = np.array([embed.numpy() for embed in embeddings_list])
    tsne_repr = TSNE(n_components=2, perplexity=1).fit_transform(embed_np)
    tsne_points = tsne_repr.tolist()
    print(f"tsne_points:\n{tsne_points}")

    plt.figure(figsize=(8, 8))
    x = [p[0] for p in tsne_points]
    y = [p[1] for p in tsne_points]
    plt.scatter(x, y)
    plt.savefig("embeddings.jpg")
    # plt.show()

if __name__ == "__main__":
    planned_cities_ids_path = pathlib.Path("../data/POC2_data/random_planned_cities_id.pkl")
    not_planned_cities_ids_path = pathlib.Path("../data/POC2_data/random_not_planned_cities_id.pkl")
    graphs_folder = pathlib.Path("../data/graphml/dataverse_files")

    planned_data_loader = IdBasedGraphDataLoader.from_ids_path(graphs_folder,
                                                               planned_cities_ids_path)
    not_planned_data_loader = IdBasedGraphDataLoader.from_ids_path(graphs_folder,
                                                               not_planned_cities_ids_path)

    print(f"num_planned_cities: {len(planned_data_loader)}")
    print(f"num_not_planned_cities: {len(not_planned_data_loader)}")

    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    target_embeddings_folder = pathlib.Path("../data/POC2_data/graph_embeddings/node2vec/")
    target_embeddings_folder.mkdir(exist_ok=True, parents=True)

    model_wrapper_class = Node2VecModelWrapper

    #Based on the best values reported in:
    #On Network Embedding for Machine Learning on Road Networks: 
    # A Case Study on the Danish Road Network 
    mdl_params_without_edge_idx = {
        'embedding_dim':64,
        'walk_length':80,
        'context_size':15,
        'walks_per_node':10,
        'num_negative_samples':2,
        'p':2,
        'q':0.25,
        'sparse':True
    }

    NUM_EPOCHS = 1
    START_IDX = 0
    STOP_IDX = 2
    print("STARTING PLANNED CITIES EMBEDDINGS")
    target_base_file_name = "planned_embeddings"
    planned_cities_embeds = GraphsEmbedder.embedd_and_save_to_folder(
        planned_data_loader, device, target_embeddings_folder,
        target_base_file_name, model_wrapper_class, 
        mdl_params_without_edge_idx, start=START_IDX, 
        stop=STOP_IDX, epochs=NUM_EPOCHS
        )
    
    print("STARTING NOT PLANNED CITIES EMBEDDINGS")
    target_base_file_name = "not_planned_embeddings"
    not_planned_cities_embeds = GraphsEmbedder.embedd_and_save_to_folder(
        not_planned_data_loader, device, target_embeddings_folder,
        target_base_file_name, model_wrapper_class, 
        mdl_params_without_edge_idx, start=START_IDX, 
        stop=STOP_IDX, epochs=NUM_EPOCHS
        )

    # planned_cities_embeds.extend(not_planned_cities_embeds)
    # plot_embeddings(planned_cities_embeds)