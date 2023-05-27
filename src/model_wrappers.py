from abc import ABC, abstractmethod
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module

#Based from https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")

    def early_stop(self, loss:float) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ModelWrapper(ABC):
    """
    This is a torch.nn.Module wrapper
    """
    def __init__(self, base_params:dict, device:str):
        self._base_params = base_params
        self._final_train_loss = None
        self._model = self._construct_model(self._base_params, device)
    
    @abstractmethod
    def train(self, loader:DataLoader, optimizer:Optimizer,
              criterion:Module, epochs:int, device:str):
        pass
    
    @abstractmethod
    def _construct_model(self, base_params:dict, device:str) -> Module:
        pass

    def get_specific_loader(self, batch_size:int=128, 
                            shuffle:bool=True) -> DataLoader:
        """
        Returns the specific loader for this model.
        If the model doesn't have one, returns None
        """
        return
    
    @property
    def model(self) -> Module:
        return self._model
    
    @property
    def final_train_loss(self):
        return self._final_train_loss