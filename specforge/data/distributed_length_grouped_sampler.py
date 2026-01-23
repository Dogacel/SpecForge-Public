import math
from typing import Iterator, Optional

import torch
import torch.distributed as dist

from datasets import Dataset
from torch.utils.data import DistributedSampler
from transformers.trainer_pt_utils import get_length_grouped_indices
from transformers.tokenization_utils_base import BatchEncoding

class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Patched version of huggingface's DistributedLengthGroupedSampler to support custom length of input sequence.
    Patch: https://github.com/huggingface/transformers/pull/43363
    """

    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: Optional[list[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if not isinstance(dataset[0], (dict, BatchEncoding)) or model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [feature[model_input_name].shape[1] for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            print(
                "If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to"
                " list[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)