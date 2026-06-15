# Torch 笔记

本文记录阅读 DeepEP 源码时遇到的 PyTorch 相关概念。

## 1. torch.distributed

### 1.1 作用

`torch.distributed` 是 PyTorch 提供的分布式通信模块，用于让多个进程、多个 GPU、甚至多个节点一起参与训练或推理。

在大模型场景中，单张 GPU 往往放不下完整模型或无法提供足够吞吐，因此需要把任务拆到多张 GPU 上。`torch.distributed` 提供了这些进程之间通信和同步的基础能力。

### 1.2 常见概念

- `rank`：当前进程在全局通信组里的编号。
- `world_size`：总共有多少个 rank。
- `local_rank`：当前进程在本机内部的编号，通常对应本机第几张 GPU。
- `ProcessGroup`：通信组，表示哪些 rank 参与同一组通信。
- `backend`：通信后端，例如 `nccl`、`gloo`。

GPU 分布式训练/推理中最常见的 backend 是 `nccl`。

### 1.3 初始化分布式环境

典型初始化方式：

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method="tcp://127.0.0.1:8361",
    world_size=8,
    rank=0,
)
```

含义：

- `backend="nccl"`：使用 NCCL 做 GPU 通信。
- `init_method`：指定所有进程 rendezvous 的地址。
- `world_size`：总 rank 数。
- `rank`：当前进程的全局 rank。

### 1.4 什么是进程 rendezvous 的地址

`rendezvous` 原意是“会合”。在 `torch.distributed` 里，rendezvous 地址就是所有分布式进程启动时用来“碰头”的地址。

分布式任务通常会启动多个进程。每个进程一开始只知道自己的 rank、world size 和一个 rendezvous 地址。它们会通过这个地址找到彼此，交换初始化通信所需的信息，然后建立真正的分布式通信组。

例如：

```python
init_method="tcp://127.0.0.1:8361"
```

含义是：

- `127.0.0.1`：master 地址。
- `8361`：master 端口。
- 所有 rank 都连接到这个地址完成初始化会合。

在单机多 GPU 测试中，`127.0.0.1` 通常够用，因为所有进程都在同一台机器上。

在多节点场景中，`MASTER_ADDR` 应该设置为某个所有节点都能访问到的 master 节点 IP，例如：

```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=8361
```

DeepEP 的 `init_dist` 中有：

```python
ip = os.getenv('MASTER_ADDR', '127.0.0.1')
port = int(os.getenv('MASTER_PORT', '8361'))
```

这表示：

- 如果环境变量里设置了 `MASTER_ADDR` 和 `MASTER_PORT`，就使用它们作为 rendezvous 地址。
- 如果没有设置，就默认使用 `127.0.0.1:8361`，适合单机测试。

注意：rendezvous 地址只负责初始化阶段让进程互相发现，并不等于后续所有 GPU 数据通信都走这个 TCP 地址。初始化完成后，GPU 间通信会根据 backend 使用 NCCL、NVLink、RDMA 等机制。

初始化完成后，可以通过以下接口查询当前进程信息：

```python
rank = dist.get_rank()
world_size = dist.get_world_size()
```

### 1.5 通信组

`torch.distributed` 可以创建新的通信组：

```python
group = dist.new_group([0, 1, 2, 3])
```

这表示 rank 0、1、2、3 组成一个新的 `ProcessGroup`。后续通信可以只在这个 group 内发生。

DeepEP 的 `ElasticBuffer` 和 legacy `Buffer` 都需要传入 `ProcessGroup`，用它确定参与 EP 通信的 rank 集合。

### 1.6 常见通信操作

`torch.distributed` 提供多种通信操作：

- `dist.barrier()`：所有 rank 在这里同步等待。
- `dist.all_reduce()`：把所有 rank 的 tensor 做规约，并让每个 rank 得到相同结果。
- `dist.all_gather()`：每个 rank 收集其他 rank 的 tensor。
- `dist.broadcast()`：从一个源 rank 向其他 rank 广播 tensor。
- `dist.send()` / `dist.recv()`：点对点通信。

在 DeepEP 中，测试和工具代码里经常用 `dist.barrier()` 做跨 rank 同步。

### 1.7 DeepEP 中的用法

在 `deep_ep/utils/envs.py` 的 `init_dist` 中：

```python
dist.init_process_group(**params)
```

这里会初始化 PyTorch 分布式环境。`params` 中包含：

```python
{
    "backend": "nccl",
    "init_method": f"tcp://{ip}:{port}",
    "world_size": num_nodes * num_local_ranks,
    "rank": node_rank * num_local_ranks + local_rank,
}
```

然后函数返回：

```python
return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))
```

也就是：

- 当前进程的全局 rank。
- 总 rank 数。
- 一个包含所有 rank 的通信 group。

在 `tests/elastic/test_ep.py` 中：

```python
rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
```

随后这个 `group` 会传入：

```python
deep_ep.ElasticBuffer(group, ...)
```

这表示 DeepEP 的 EP 通信会在这个 PyTorch 分布式 group 内进行。

### 1.8 清理分布式环境

测试结束后，通常需要销毁进程组：

```python
dist.destroy_process_group()
```

DeepEP 的 `test_loop` 末尾也会调用它，避免分布式资源泄漏。

### 1.9 小结

`torch.distributed` 是 PyTorch 的分布式通信模块。它负责初始化多进程通信环境、管理 rank/world size/process group，并提供 barrier、all-reduce、all-gather 等通信操作。DeepEP 使用它来建立分布式基础环境，再把 `ProcessGroup` 交给 `ElasticBuffer` 或 `Buffer` 执行专家并行通信。
