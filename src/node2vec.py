import pathlib
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from torch_geometric import seed_everything
import torch
import osmnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

def edge_list_coo_format(city, city_stats):
    """
    Not used. Its a slower version of edge_list_coo_format_using_scipy
    """
    graph_edge_index = np.empty((city_stats['m'], 2))
    node_map = dict()
    for edge_idx, line in enumerate(nx.generate_edgelist(city, data=False)):
        origin_node_id, dest_node_id = line.split(" ")
        origin_node_id = int(origin_node_id)
        dest_node_id = int(dest_node_id)

        mapped_origin_node_id = node_map.setdefault(origin_node_id, len(node_map))
        mapped_dest_node_id = node_map.setdefault(dest_node_id, len(node_map))
        graph_edge_index[edge_idx][0] = mapped_origin_node_id
        graph_edge_index[edge_idx][1] = mapped_dest_node_id
    
    graph_edge_index = graph_edge_index.T

    return graph_edge_index,node_map

#Based on https://stackoverflow.com/a/50665264
def edge_list_coo_format_using_scipy(city) -> torch.Tensor:
    coo_style = nx.to_scipy_sparse_array(city, format='coo')
    indices = np.vstack((coo_style.row, coo_style.col))
    return torch.from_numpy(indices).type(torch.IntTensor)

#From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
# and https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial11/Tutorial11.ipynb
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
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

def get_graph_embedding(model:Node2Vec, data:Data, device:str):
    """
    Returns the mean of the nodes embeddings
    """
    with torch.inference_mode():
        #z row represents a node
        z = model(torch.arange(data.num_nodes, device=device))
        # print(z)
        print(z.size())
        print(type(z))
        #Now, each col represents a node
        z = z.T
        mean = torch.mean(z, 1).to('cpu')
        print(f"Mean Size: {mean.size()}")
        return mean

def get_num_nodes_of(city):
    city_stats = osmnx.stats.basic_stats(city)
    n_nodes = city_stats['n']
    return n_nodes

def plot_embeddings(embeddings_list):
    embed_np = np.array([embed.numpy() for embed in embeddings_list])
    tsne_repr = TSNE(n_components=2, perplexity=1).fit_transform(embed_np)
    tsne_points = tsne_repr.tolist()
    print(f"tsne_points: {tsne_points}")

    plt.figure(figsize=(8, 8))
    x = [p[0] for p in tsne_points]
    y = [p[1] for p in tsne_points]
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    GRAPH_DATA_FOLDER = pathlib.Path("../data/graphml/")
    print(f"Data folder: {GRAPH_DATA_FOLDER}")
    
    graph_data_files = GRAPH_DATA_FOLDER.glob("*/*.graphml")

    embeddings_list:torch.Tensor = list()
    for data_file in tqdm(list(graph_data_files)[:2]):
        city = osmnx.io.load_graphml(data_file)
        print(f"\nCity: {data_file}")
        
        city_edge_list = edge_list_coo_format_using_scipy(city)
        print(city_edge_list[:15])
        print(f"EDGE LIST MEM SIZE: {(city_edge_list.element_size() * city_edge_list.nelement())/1024} MB")

        #Ver https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html para criar um dataset
        #https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
        n_nodes = get_num_nodes_of(city)
        print(f"num nodes: {n_nodes}")
        data = Data(edge_index=city_edge_list, num_nodes=n_nodes)
        data.validate(raise_on_error=True)

        #Based on the best values reported in:
        #On Network Embedding for Machine Learning on Road Networks: 
        # A Case Study on the Danish Road Network
        model = Node2Vec(
            data.edge_index,
            embedding_dim=64,
            walk_length=80,
            context_size=15,
            walks_per_node=10,
            num_negative_samples=2,
            p=2,
            q=0.25,
            sparse=False,
        ).to(device)

        loader = model.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        for epoch in range(1, 2):
            loss = train(model, loader, optimizer, device)
            print(f"Loss: {loss:.4f}")
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        
        graph_emb = get_graph_embedding(model, data, device)
        embeddings_list.append(graph_emb)

    plot_embeddings(embeddings_list)