import argparse
import torch.distributed

from tritonbench.distributed_utils.distributed_op import (
    DistributedOperator,
    register_benchmark,
)
from tritonbench.distributed_utils.env_utils import initialize_distributed

from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context

from typing import List, Dict

layer_configs = {
    "LLaMA-7B": {"N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "stage": 5},
    "LLaMA-3.1-8B": {"N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "stage": 5},
    "LLaMA-3.1-70B": {
        "N": 28672,
        "K": 8192,
        "BM": 128,
        "BN": 256,
        "BK": 64,
        "stage": 3,
    },
    "Qwen2-72B": {"N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
    "GPT-3-175B": {"N": 49152, "K": 12288, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
    "LLaMA-3.1-405B": {
        "N": 53248,
        "K": 16384,
        "BM": 128,
        "BN": 256,
        "BK": 64,
        "stage": 3,
    },
}


class Operator(DistributedOperator):
    def __init__(self, tb_args: argparse.Namespace, extra_args: List[str]):
        self.tb_args = tb_args
        self.td_group = initialize_distributed()

    @register_benchmark()
    def torch(
        self,
        pg: torch.distributed.ProcessGroup,
        local_input: torch.Tensor,
        local_weight: torch.Tensor,
        ag_out: torch.Tensor,
    ):
        def _inner():
            torch.distributed.all_gather_into_tensor(ag_out, local_input, group=pg)
            ag_gemm_output = torch.matmul(ag_out, local_weight)
            return ag_gemm_output

        return _inner

    @register_benchmark()
    def triton(
        self,
        pg: torch.distributed.ProcessGroup,
        local_input: torch.Tensor,
        local_weight: torch.Tensor,
        ag_out: torch.Tensor,
    ):
        A = local_input
        B = local_weight
        ctx = create_ag_gemm_context(
            A,
            B,
            rank=pg.rank(),
            world_size=pg.size(),
            max_M=M,
            BLOCK_M=config["BM"],
            BLOCK_N=config["BN"],
            BLOCK_K=config["BK"],
            stages=config["stage"],
        )
        inner = lambda: ag_gemm(A, B, ctx=ctx, autotune=False)
        return inner

    def get_input_iter(self):
        M = 8192
        pass
