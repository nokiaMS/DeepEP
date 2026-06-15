import functools
import inspect
import os
import random
import re
import subprocess
import torch
import torch.distributed as dist
from typing import Tuple

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .comm import get_nccl_comm_handle

_local_rank = None
_local_seed = 0
_global_seed = 0

# Default NIC name for RDMA operations, configurable via environment variable
_DEFAULT_NIC_NAME = os.getenv('EP_NIC_NAME', 'mlx5_0')


def init_seed(global_seed: int) -> None:
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()
    _global_seed = global_seed
    torch.manual_seed(_local_seed)
    random.seed(_local_seed)


def get_local_seed() -> int:
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed


def get_global_seed() -> int:
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.

    Returns:
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.

    中文函数注释：
        初始化当前进程的 PyTorch 分布式环境，使用 NCCL backend 建立 GPU 通信。
        该函数会根据环境变量计算全局 rank/world size，设置当前 CUDA 设备，
        初始化随机种子，并返回当前 rank、总 rank 数和一个包含所有 rank 的通信 group。

    中文参数说明：
        local_rank：当前进程在本机内部的 rank 编号，通常对应本机第几张 GPU。
        num_local_ranks：本机启动的 rank 数量，通常等于本机参与测试的 GPU 数量。
        seed：全局随机种子，后续会结合全局 rank 生成每个 rank 自己的 local seed。

    中文返回值说明：
        rank：当前进程在全局通信域中的 rank 编号。
        world_size：整个分布式任务中的总 rank 数。
        group：包含所有 rank 的 PyTorch 分布式通信组。
    """
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')  # 读取 master 节点地址；未设置时默认使用本机 127.0.0.1。
    port = int(os.getenv('MASTER_PORT', '8361'))  # 读取 master 端口；未设置时默认使用 8361，并转换为整数。
    num_nodes = int(os.getenv('WORLD_SIZE', 1))  # 读取节点数量；未设置时默认单节点，并转换为整数。
    node_rank = int(os.getenv('RANK', 0))  # 读取当前节点编号；未设置时默认是第 0 个节点，并转换为整数。

    # Set local rank
    global _local_rank  # 声明要修改模块级全局变量 `_local_rank`。
    _local_rank = local_rank  # 保存当前进程的本地 rank，供 dist_print 等工具函数使用。

    sig = inspect.signature(dist.init_process_group)  # 获取 `dist.init_process_group` 的函数签名，用于兼容不同 PyTorch 版本。
    params = {  # 构造初始化 PyTorch 分布式进程组所需的参数字典。
        'backend': 'nccl',  # 使用 NCCL backend，适合 GPU 间通信。
        'init_method': f'tcp://{ip}:{port}',  # 使用 TCP 地址和端口作为进程组初始化 rendezvous 地址。
        'world_size': num_nodes * num_local_ranks,  # 计算全局 rank 总数：节点数乘以每节点 rank 数。
        'rank': node_rank * num_local_ranks + local_rank,  # 计算当前进程的全局 rank。
    }  # 分布式初始化参数构造完成。
    if 'device_id' in sig.parameters:  # 如果当前 PyTorch 版本支持 `device_id` 参数，则显式传入 CUDA 设备。
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')  # 将当前 local rank 对应的 CUDA 设备写入初始化参数。
    dist.init_process_group(**params)  # 根据参数初始化 PyTorch 分布式进程组。
    torch.set_default_dtype(torch.bfloat16)  # 将 PyTorch 默认 dtype 设置为 bfloat16，贴近 DeepEP 测试数据类型。
    torch.set_default_device('cuda')  # 将 PyTorch 默认设备设置为 CUDA，后续 tensor 默认创建在 GPU 上。
    torch.cuda.set_device(local_rank)  # 将当前进程绑定到 local_rank 对应的 GPU。

    init_seed(seed)  # 初始化随机种子；内部会用 global seed 加当前全局 rank 得到 local seed。
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))  # 返回当前全局 rank、总 rank 数，以及包含所有 rank 的新通信组。


def get_physical_domain_size(group: dist.ProcessGroup) -> Tuple[int, int]:
    """
    Get the physical domain sizes (RDMA ranks and NVLink ranks).

    Arguments:
        group: the communication group.

    Returns:
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
    """
    return _C.get_physical_domain_size(get_nccl_comm_handle(group).get())


def get_logical_domain_size(group: dist.ProcessGroup, allow_hybrid_mode: bool = True) -> Tuple[int, int]:
    """
    Get the logical domain sizes (scaleout ranks and scaleup ranks).

    Arguments:
        group: the communication group.
        allow_hybrid_mode: whether to enable hybrid mode.

    Returns:
        num_scaleout_ranks: the number of logical scaleout ranks.
        num_scaleup_ranks: the number of logical scaleup ranks.
    """
    return _C.get_logical_domain_size(get_nccl_comm_handle(group).get(), allow_hybrid_mode)


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'

        # noinspection PyUnresolvedReferences
        import pynvml
        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [0, ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                assert status == pynvml.NVML_P2P_STATUS_OK, \
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'

        # Close NVML
        pynvml.nvmlShutdown()


def check_torch_deterministic() -> None:
    """
    Ensure PyTorch deterministic algorithms and fill_uninitialized_memory are not both enabled.
    When both are on, `torch.empty()` calls an initialization kernel that may overlap with communication streams,
    causing errors.
    """
    assert not (torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory)


@functools.lru_cache()
def get_nvlink_gbs(factor: float = 0.9) -> float:
    """
    Get the total NVLink bandwidth in GB/s, cached.

    Arguments:
        factor: the bandwidth efficiency factor.

    Returns:
        gbs: the total NVLink bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                                capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
        assert match

        gpu_block = match.group(0)
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'
        link_matches = re.findall(link_pattern, gpu_block)
        assert link_matches
        return sum(float(bw) for bw in link_matches) * factor
    except Exception as e:
        print(f'Failed to get NVLink connection speed: {e}')
        return 0


@functools.lru_cache()
def check_fast_rdma_atomic_support(nic_name: str = _DEFAULT_NIC_NAME) -> bool:
    """
    Check whether the NIC supports fast RDMA atomic operations (MT4131 or newer).

    Arguments:
        nic_name: the NIC device name.

    Returns:
        supported: `True` if fast RDMA atomics are supported.
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = rf"CA '{nic_name}'.*?CA type:\s*(\S+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        return match.group(1) == 'MT4131'
    except Exception:
        return False


@functools.lru_cache()
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout

        pattern = rf"CA '{nic_name}'.*?Port \d+:\s*.*?Rate:\s*(\d+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        rate = int(match.group(1))
        return rate / 8
    except Exception as e:
        print(f'Failed to get RDMA connection speed: {e}')
        return 0
