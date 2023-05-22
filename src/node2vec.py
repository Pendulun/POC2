import pathlib
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import torch
import osmnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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
def edge_list_coo_format_using_scipy(city):
    coo_style = nx.to_scipy_sparse_array(city, format='coo')
    indices = np.vstack((coo_style.row, coo_style.col))
    return torch.from_numpy(indices).type(torch.LongTensor)

#From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
# and https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial11/Tutorial11.ipynb
def train(loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
    return acc

# @torch.no_grad()
# def plot_points(colors):
#     model.eval()
#     z = model(torch.arange(data.num_nodes, device=device))
#     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
#     y = data.y.cpu().numpy()

#     plt.figure(figsize=(8, 8))
#     for i in range(dataset.num_classes):
#         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
#     plt.axis('off')
#     plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    GRAPH_DATA_FOLDER = pathlib.Path("../data/graphml/")
    print(f"Data folder: {GRAPH_DATA_FOLDER}")
    
    graph_data_files = list(GRAPH_DATA_FOLDER.glob("*/*.graphml"))

    data_file = graph_data_files[0]
    city = osmnx.io.load_graphml(data_file)
    print(f"City: {data_file}")
    city_stats = osmnx.stats.basic_stats(city)
    # print(f"Stats: {city_stats}")
    
    city_edge_list = edge_list_coo_format_using_scipy(city)
    print(f"edge list: {city_edge_list}")

    #Ver https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html para criar um dataset
    #https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
    data = Data(edge_index=city_edge_list, num_nodes=city_stats['n'])
    data.validate(raise_on_error=True)
    print(data.is_directed())

    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(1, 101):
        loss = train(loader, optimizer, device)
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    


    # colors = [
    #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700'
    # ]
    # plot_points(colors)

