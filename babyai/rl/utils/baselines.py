from abc import abstractmethod, ABC
from collections import defaultdict

import torch


class Baseline(ABC):
    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        """Update internal state according to the observed loss
            loss (torch.Tensor): batch of losses
        """
        pass

    @abstractmethod
    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        """Return baseline for the loss
        Args:
            loss (torch.Tensor): batch of losses be baselined
        """
        pass


class MeanBaseline(Baseline):
    """Running mean baseline; all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """

    def __init__(self):
        super().__init__()

        self.mean_baseline = torch.zeros(1, requires_grad=False)
        self.n_points = 0.0

    def update(self, loss: torch.Tensor) -> None:
        self.n_points += 1
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)

        self.mean_baseline += (loss.detach().mean().item() -
                               self.mean_baseline) / self.n_points

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)
        return self.mean_baseline

    
class MissionMeanBaseline(object):
    """Running mean baseline seperatedly for each mission; 
    all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """
    
    def __init__(self):
        self.mission_dict = {}
    
    def update(self, missions, loss: torch.Tensor) -> None:
        for i, mission in enumerate(missions):
            if mission not in self.mission_dict.keys():
                self.mission_dict[mission] = MeanBaseline()
            self.mission_dict[mission].update(loss[i])
    
    def predict(self, missions, loss: torch.Tensor) -> torch.Tensor:
        prediction = [None] * loss.size()[0]
        for i, mission in enumerate(missions):
            if mission not in self.mission_dict.keys():
                self.mission_dict[mission] = MeanBaseline()
            prediction[i] = self.mission_dict[mission].predict(loss[i])
        return torch.stack(prediction).to(loss.device)
