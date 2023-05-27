from __future__ import annotations
import pathlib
import pickle
import osmnx

class IdBasedGraphDataLoaderIterator:
    def __init__(self, graph_dl:IdBasedGraphDataLoader):
       self._graph_dl = graph_dl
       self._index = 0
    
    def __next__(self):
        ''''Returns the next loaded graph using osmnx.io.load_graphml from graph_dl files list'''
        if self._index < (len(self._graph_dl._graphs_files)):
            graph_file = self._graph_dl._graphs_files[self._index]
            self._index +=1
            return osmnx.io.load_graphml(graph_file)
        
        raise StopIteration

class IdBasedGraphDataLoader():
    """
    This dataloader loads .graphml files that contain a certain id on its name.
    The file names should be: something-id.graphml and should all be at the folder provided.
    """
    def __init__(self, graphs_folder:pathlib.Path, graphs_ids:set[int]):
        self._graphs_files = self._get_graphs_files_with_ids(graphs_folder, graphs_ids)
    
    def _get_graphs_files_with_ids(self, folder:pathlib.Path, ids:list[int]) -> list[pathlib.Path]:
        num_files_expected = len(ids)
        target_files = list()
        for file_path in folder.glob("*.graphml"):
            city_id = self._get_city_id_from_file_name(file_path)

            if city_id in ids:
                target_files.append(file_path)
            
            if len(target_files) == num_files_expected:
                break
        return target_files
    
    def _get_city_id_from_file_name(self, file_path):
        """
        Return the id present in the file name.
        Assumes that the file name is like name-id.extension where the id is an int
        """
        city_name_and_id = file_path.stem
        city_id = int(city_name_and_id.split('-')[1])
        return city_id

    @classmethod
    def from_ids_path(cls, graphs_folder:pathlib.Path, 
                      graphs_ids_pickle_file_path:pathlib.Path):
        
        with open(graphs_ids_pickle_file_path, 'rb') as my_file:
            data = pickle.load(my_file)

        ids = set(data)
        return IdBasedGraphDataLoader(graphs_folder, ids)
        
    def __iter__(self):
        ''' Returns the Iterator object '''
        return IdBasedGraphDataLoaderIterator(self)
    
    def __len__(self):
        return len(self._graphs_files)
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            graph_files = self._graphs_files[i]
            for graph_file in graph_files:
                yield osmnx.io.load_graphml(graph_file)
        else:
            return osmnx.io.load_graphml(self._graphs_files[i])