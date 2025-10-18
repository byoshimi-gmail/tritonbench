import argparse
import torch
import nvshmem.core
from typing import List


class DistributedOperator:
    def __init__(self, tb_args: argparse.Namespace, extra_args: List[str]):
        self.tb_args = tb_args
        self.extra_args = extra_args

    def run(self):
        pass
        # setup context
        ctx = create_distributed_context()

    def accuracy(self):
        pass

    def finalize(self):
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group()
