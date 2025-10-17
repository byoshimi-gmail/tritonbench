import argparse

from tritonbench.utils.triton_op import register_benchmark
from typing import List


class DistributedOperator:
    def __init__(self, tb_args: argparse.Namespace, extra_args: List[str]):
        self.tb_args = tb_args
        self.extra_args = extra_args

    def run(self):
        pass
        # setup context
        ctx = create_distributed_context()
