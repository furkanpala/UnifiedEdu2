"""
federation/fedavg.py

Baseline: standard FedAvg.

All institutions synchronise every round with a plain unweighted
average of their Theta vectors. No clustering, no warm-up.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..config import FederationConfig, ModelGraphConfig


class FedAvgServer:
    """
    Drop-in replacement for FederationServer implementing vanilla FedAvg.

    The interface mirrors FederationServer so the same training harness
    (train.py) can be used for both by swapping server implementations.
    """

    def __init__(
        self,
        num_clients: int,
        fed_config:  FederationConfig,
        mg_config:   ModelGraphConfig,
    ) -> None:
        self.num_clients = num_clients
        self.cfg         = fed_config

        # Lazy-initialised from the first aggregate() call
        self._global_theta: Optional[torch.Tensor] = None

        self._client_ids: List[int] = list(range(num_clients))
        self._round: int = 0

    @property
    def round(self) -> int:
        return self._round

    def broadcast(self) -> Dict[int, torch.Tensor]:
        """Every client receives the same global Theta."""
        if self._global_theta is None:
            return {cid: None for cid in self._client_ids}
        return {cid: self._global_theta.clone() for cid in self._client_ids}

    def aggregate(
        self,
        round_num: int,
        uploads:   Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Plain average of all uploaded Theta vectors.

        Returns the same averaged Theta to every client.
        """
        self._round      = round_num
        self._client_ids = sorted(uploads.keys())

        stacked = torch.stack(list(uploads.values()))   # (m, p)
        mean    = stacked.mean(dim=0)                   # (p,)
        self._global_theta = mean.clone()

        return {cid: mean.clone() for cid in self._client_ids}

    def run(self, clients, num_rounds: Optional[int] = None) -> None:
        T = num_rounds if num_rounds is not None else self.cfg.num_rounds
        for t in range(1, T + 1):
            thetas_in = self.broadcast()
            uploads   = {
                self._client_ids[i]: client.local_train(thetas_in[self._client_ids[i]])
                for i, client in enumerate(clients)
            }
            self.aggregate(t, uploads)
