import os
import torch
import datetime
import numpy as np
import random


def is_cuda():
    if torch.cuda.is_available() and (torch.version.hip is None):
        return True


def is_hip():
    if torch.cuda.is_available() and (torch.version.hip is not None):
        return True


if is_cuda():
    from cuda import cuda, cudart

    import nvshmem
    import nvshmem.core
    from nvshmem.core.utils import _get_device
elif is_hip():
    from hip import hip
else:
    pass


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    # zero empty takes more kernel launch and may hide uninitialized problem. always set to False
    # available since torch 2.2: https://docs.pytorch.org/docs/2.2/deterministic.html
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:
        logging.warning(
            "torch.utils.fill_uninitialized_memory is available only for torch >=2.2"
        )
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


def finalize_distributed():
    if is_cuda():
        nvshmem.core.finalize()
    torch.distributed.destroy_process_group()


def init_nvshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    # Extract rank, nranks from process group
    num_ranks = pg.size()
    rank_id = pg.rank()

    # Create an empty uniqueid for all ranks
    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=0, group=pg)
    torch.distributed.barrier(group=pg)
    from cuda.core.experimental import Device

    nvshmem.core.init(
        device=Device(torch.cuda.current_device()),
        uid=broadcast_objects[0],
        rank=rank_id,
        nranks=num_ranks,
        initializer_method="uid",
    )
    # nvshmem.core.utils._configure_logging("DEBUG")


def initialize_distributed(seed=None) -> torch.distributed.ProcessGroup:
    global _TP_GROUP
    assert _TP_GROUP is None, "TP_GROUP has already been initialized"

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    assert WORLD_SIZE > 0, "WORLD_SIZE must be greater than 0"

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )

    assert torch.distributed.is_initialized(), (
        "Distributed process group is not initialized"
    )

    _TP_GROUP = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)), backend="nccl"
    )
    torch.distributed.barrier(_TP_GROUP)
    _TP_GROUP_GLOO = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)), backend="gloo"
    )
    torch.distributed.barrier(_TP_GROUP_GLOO)

    init_seed(seed=seed if seed is not None else RANK)
    init_nvshmem_by_torch_process_group(_TP_GROUP)
    return _TP_GROUP
