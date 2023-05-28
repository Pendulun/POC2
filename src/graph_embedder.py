import numpy as np
import networkx as nx
import osmnx
import pathlib
import pickle
import torch
import time
from tqdm import tqdm
from typing import Tuple

from torch_geometric.data import Data
import osmnx

from data_loader import IdBasedGraphDataLoader
from model_wrappers import ModelWrapper

class GraphsEmbedder():
    MB_SIZE = 1024 * 1024
    @classmethod
    def embedd_and_save_to_folder(cls, data_loader:IdBasedGraphDataLoader, device:str, 
                              target_embeddings_folder:pathlib.Path,
                              target_base_file_name:pathlib.Path,
                              model_wrapper_class:type[ModelWrapper], 
                              model_params_without_edge_idx:dict,
                              start:int=0, stop:int=None, epochs:int=15, 
                              save_every:int=5):
        """
        Compute embeddings based on the model_wrapper. Save the embeddins and final losses
        for each graph at the target folder.
        Return the embeddings
        """
        if stop == None:
            stop = len(data_loader)

        all_embeddings = list()
        for starting_city_idx in tqdm(range(start, stop, save_every), desc='outer'):
            print("\n")
            curr_stop = starting_city_idx + save_every

            curr_embeddings, losses = GraphsEmbedder.embedd_graphs(
                data_loader, device, model_wrapper_class, model_params_without_edge_idx,
                starting_city_idx, curr_stop, epochs
                )
            
            if len(curr_embeddings) > 0:
                print(f"CITIES LOSSES:\n{losses}")
                GraphsEmbedder._save_embedd_and_losses_to_file(
                    target_embeddings_folder,  target_base_file_name, 
                    curr_embeddings, losses, starting_city_idx, curr_stop
                    )
            all_embeddings.extend(curr_embeddings)

        return all_embeddings
    
    @classmethod
    def _save_embedd_and_losses_to_file(cls, target_folder:pathlib.Path, target_file_name:str, 
                                   embeddings:list, losses:list[float],
                                   start_idx:int, stop_idx:int):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        planned_embeddings_file_name = f"{target_file_name}_{start_idx}_{stop_idx}_{timestr}.pt"
        target_planned_embeddings_file = target_folder / planned_embeddings_file_name
        torch.save(embeddings, target_planned_embeddings_file)

        target_losses_file = target_folder / f"losses_{start_idx}_{stop_idx}_{timestr}.pkl"
        with open(target_losses_file, 'wb') as losses_file:
            pickle.dump(losses, losses_file)

    @classmethod
    def embedd_graphs(cls, data_loader:IdBasedGraphDataLoader, device:str, 
                  model_wrapper_class:type[ModelWrapper], 
                  model_params_without_edge_idx:dict,
                  start:int=0, stop:int=None, epochs:int=15) -> Tuple[list[torch.Tensor], list[float]]:
        """
        Embedds the graphs in the data_loader.
        Returns a list with each graph embedding and a list of final train losses
        """
        if stop == None:
            stop = len(data_loader)

        print(f"start: {start}; stop: {stop}")
        
        embeddings_list:list[torch.Tensor] = list()
        losses:list[float] = list()
        for city_idx, city_graph in enumerate(
            tqdm(data_loader[start:stop], 
                 total=stop-start, 
                 desc="inner")):
            real_city_idx = city_idx + start
            print(f"\nCity idx: {real_city_idx}\n")
            # city_graph = osmnx.io.load_graphml(city_graph_path)
            city_edge_list, n_nodes = GraphsEmbedder._get_city_edge_list_and_num_nodes(city_graph)
            data = Data(edge_index=city_edge_list, num_nodes=n_nodes)
            data.validate(raise_on_error=True)

            model_params_without_edge_idx['edge_index'] = data.edge_index

            model, final_loss = GraphsEmbedder._get_trained_model_on_data(device, epochs, 
                                                        data, model_wrapper_class,
                                                        model_params_without_edge_idx
                                                        )
            losses.append(final_loss)
            graph_emb = GraphsEmbedder._get_graph_embedding(model, data, device)
            embeddings_list.append(graph_emb)
        
        return embeddings_list, losses

    @classmethod
    def _get_trained_model_on_data(cls, device:str, epochs:int, data:Data, 
                                model_wrapper_class:type[ModelWrapper], 
                                model_params:dict
                                ) -> Tuple[torch.nn.Module, float]:
        
        model_wrapper = model_wrapper_class(model_params, device)
        loader = None
        specific_loader = model_wrapper.get_specific_loader(batch_size = 128, 
                                                        shuffle = True)
        if specific_loader == None:
            #TODO - Use a DataLoader with batch size and shuffle
            raise NotImplementedError("DataLoader creation not implemented!")
        else:
            #Node2Vec model has its own loader
            loader = specific_loader

        criterion = None

        #Maybe this is common to all models.
        optimizer = torch.optim.SparseAdam(model_wrapper.model.parameters(), lr=0.01)

        model_wrapper.train(loader, optimizer, criterion, epochs, device)
        trained_model = model_wrapper.model
        final_loss = model_wrapper.final_train_loss

        return trained_model, final_loss

    @classmethod
    def _get_graph_embedding(cls, model:torch.nn.Module, data:Data, device:str) -> torch.Tensor:
        """
        Returns the mean of the nodes embeddings
        """
        with torch.inference_mode():
            #z row represents a node
            z = model(torch.arange(data.num_nodes, device=device))
            #Now, each col represents a node
            z = z.T
            mean = torch.mean(z, 1).to('cpu')
            return mean

    @classmethod
    def _get_city_edge_list_and_num_nodes(cls, city_graph) -> Tuple[torch.Tensor, int]:
        """
        Get graph edge list as torch.Tensor in COO format and num of nodes
        """
        #Ver https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html para criar um dataset
        #https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
        n_nodes, n_edges = GraphsEmbedder._get_num_nodes_and_edgesof(city_graph)
            
        city_edge_list = GraphsEmbedder._edge_list_coo_format_using_scipy(city_graph)
        total_size = (city_edge_list.element_size() * city_edge_list.numel()) / GraphsEmbedder.MB_SIZE
        print(f"num nodes: {n_nodes}, num edges: {n_edges}, total size edge list: {total_size:.4f} MB")

        return city_edge_list, n_nodes

    @classmethod
    def _get_num_nodes_and_edgesof(cls, city) -> Tuple[int, int]:
        city_stats = osmnx.stats.basic_stats(city)
        n_nodes = city_stats['n']
        n_edges = city_stats['m']
        return n_nodes, n_edges

    @classmethod
    #Based on https://stackoverflow.com/a/50665264
    def _edge_list_coo_format_using_scipy(cls, city) -> torch.Tensor:
        coo_style = nx.to_scipy_sparse_array(city, format='coo')
        indices = np.vstack((coo_style.row, coo_style.col))
        return torch.from_numpy(indices).type(torch.LongTensor)

    @classmethod
    def _edge_list_coo_format(cls, city, city_stats):
        """
        Not used. Its a slower version of _edge_list_coo_format_using_scipy
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