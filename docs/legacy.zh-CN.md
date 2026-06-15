# DeepEP V1（Legacy）

> **注意：** 这是 DeepEP V1（基于 NVSHMEM）的归档文档。最新 V2 文档见[主 README](../README.md)。

---

DeepEP（DeepEveryParallel）V1 是面向现代机器学习的原始高性能通信库，重点关注专家并行（EP）。它提供高吞吐和低延迟的 all-to-all GPU kernels，也就是 MoE dispatch 和 combine。该库也支持 FP8 等低精度操作。

为了匹配 [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) 论文中提出的 group-limited gating 算法，DeepEP V1 提供了一组针对非对称域带宽转发优化的 kernels，例如将数据从 NVLink domain 转发到 RDMA domain。这些 kernels 提供高吞吐能力，适合训练和推理 prefilling 任务。此外，它们也支持控制 SM（Streaming Multiprocessors）数量。

对于延迟敏感的推理 decoding，DeepEP V1 包含一组基于纯 RDMA 的低延迟 kernels，用于最小化延迟。该库还引入了一种基于 hook 的通信-计算 overlap 方法，不占用任何 SM 资源。

注意：该库中的实现可能与 [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) 论文存在一些细微差异。

## 性能

### 使用 NVLink 和 RDMA forwarding 的普通 kernels

我们在 H800 上测试普通 kernels。每张 H800 的 NVLink 最大带宽约为 160 GB/s，并连接到一张 CX7 InfiniBand 400 Gb/s RDMA 网卡，最大带宽约为 50 GB/s。测试遵循 DeepSeek-V3/R1 的预训练设置：每 batch 4096 tokens、7168 hidden、top-4 groups、top-8 experts、FP8 dispatching 和 BF16 combining。

|   Type    | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:------------:|:--------------------:|:-----------:|:--------------------:|
| Intranode |      8       |  153 GB/s (NVLink)   |      8      |  158 GB/s (NVLink)   |
| Internode |      16      |    43 GB/s (RDMA)    |     16      |    43 GB/s (RDMA)    |
| Internode |      32      |    58 GB/s (RDMA)    |     32      |    57 GB/s (RDMA)    |
| Internode |      64      |    51 GB/s (RDMA)    |     64      |    50 GB/s (RDMA)    |

### 使用纯 RDMA 的低延迟 kernels

我们在 H800 上测试低延迟 kernels。每张 H800 都连接到一张 CX7 InfiniBand 400 Gb/s RDMA 网卡，最大带宽约为 50 GB/s。测试遵循典型的 DeepSeek-V3/R1 生产设置：每 batch 128 tokens、7168 hidden、top-8 experts、FP8 dispatching 和 BF16 combining。

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:------------:|:-------:|:--------------:|:-----------:|:-------:|:--------------:|
|      8       |  77 us  |    98 GB/s     |      8      | 114 us  |    127 GB/s    |
|      16      | 118 us  |    63 GB/s     |     16      | 195 us  |    74 GB/s     |
|      32      | 155 us  |    48 GB/s     |     32      | 273 us  |    53 GB/s     |
|      64      | 173 us  |    43 GB/s     |     64      | 314 us  |    46 GB/s     |
|     128      | 192 us  |    39 GB/s     |     128     | 369 us  |    39 GB/s     |
|     256      | 194 us  |    39 GB/s     |     256     | 360 us  |    40 GB/s     |

## 快速开始

### 环境要求

- Ampere（SM80）、Hopper（SM90）GPU，或者其他支持 SM90 PTX ISA 的架构
- Python 3.8 及以上
- CUDA 版本
    - 对于 SM80 GPU，需要 CUDA 11.0 及以上
    - 对于 SM90 GPU，需要 CUDA 12.3 及以上
- PyTorch 2.1 及以上
- 节点内通信需要 NVLink
- 节点间通信需要 RDMA 网络

### 下载并安装 NVSHMEM 依赖

DeepEP V1 依赖 NVSHMEM。具体说明请参考 NVSHMEM 安装指南。

### 开发

```bash
# 构建并为 SO 文件创建符号链接
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py build
# 可以根据自己的平台修改具体的 SO 文件名
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# 运行测试用例
# 说明：可以根据自己的集群设置修改 `tests/utils.py` 中的 `init_dist` 函数，
# 并在多节点上启动
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py
```

### 安装

```bash
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

#### 安装环境变量

- `NVSHMEM_DIR`：NVSHMEM 目录路径；如果未指定，会禁用所有 internode 和 low-latency 功能
- `DISABLE_SM90_FEATURES`：`0` 或 `1`，是否禁用 SM90 特性；对于 SM90 设备或 CUDA 11，该变量是必需的
- `TORCH_CUDA_ARCH_LIST`：目标架构列表，例如 `TORCH_CUDA_ARCH_LIST="9.0"`
- `DISABLE_AGGRESSIVE_PTX_INSTRS`：`0` 或 `1`，是否禁用激进的 load/store 指令；更多细节见[未定义行为 PTX 用法](#未定义行为-ptx-用法)

## 网络配置

DeepEP 已经在 InfiniBand 网络上完成充分测试。不过从理论上讲，它也兼容 RDMA over Converged Ethernet（RoCE）。

### 流量隔离

InfiniBand 通过 Virtual Lanes（VL）支持流量隔离。

为了避免不同类型流量互相干扰，建议将工作负载隔离到不同的 virtual lanes：

- 使用普通 kernels 的 workloads
- 使用低延迟 kernels 的 workloads
- other workloads

对于 DeepEP V1，可以通过设置 `NVSHMEM_IB_SL` 环境变量来控制 virtual lane 分配。

### 自适应路由

自适应路由是 InfiniBand 交换机提供的高级路由能力，可以把流量均匀分布到多条路径上。启用自适应路由可以完全消除由路由冲突导致的网络拥塞，但也会引入额外延迟。为了获得最佳性能，建议采用以下配置：

- 在网络负载较重的环境中启用自适应路由
- 在网络负载较轻的环境中使用静态路由

### 拥塞控制

建议关闭拥塞控制，因为在生产环境中尚未观察到显著拥塞。

## 接口和示例

### 用于模型训练或推理 prefilling

普通 kernels 可以用于模型训练或推理 prefilling 阶段（不包含 backward 部分），如下方示例代码所示。

```python
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# 通信 buffer（将在运行时分配）
_buffer: Optional[Buffer] = None

# 设置要使用的 SM 数量
# 说明：这是一个静态变量
Buffer.set_num_sms(24)


# 可以在框架初始化时调用该函数
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # 说明：也可以将 `get_*_config` 替换成通过所有测试得到的 auto-tuned 结果
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # 如果 buffer 不存在或大小不足，则分配一个 buffer
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # 说明：可选的 `previous_event` 表示一个已捕获的 CUDA event，
    # 你希望将它作为 dispatch kernel 的依赖。
    # 这在通信-计算 overlap 中可能有用。更多信息请参考 `Buffer.dispatch` 文档
    global _buffer

    # 在实际 dispatch 前计算 layout
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # 执行 MoE dispatch
    # 说明：CPU 会等待 GPU 的信号到达，因此这与 CUDA graph 不兼容
    # 除非指定 `num_worst_tokens`，但该 flag 仅适用于 intranode
    # 更多高级用法请参考 `dispatch` 函数文档
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # event 管理请参考 `EventOverlap` 类文档
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # MoE dispatch 的 backward 过程本质上是一次 combine
    # 更多高级用法请参考 `combine` 函数文档
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # event 管理请参考 `EventOverlap` 类文档
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # 执行 MoE combine
    # 更多高级用法请参考 `combine` 函数文档
    combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
                                           allocate_on_comm_stream=previous_event is not None)

    # event 管理请参考 `EventOverlap` 类文档
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # MoE combine 的 backward 过程本质上是一次 dispatch
    # 更多高级用法请参考 `dispatch` 函数文档
    grad_x, _, _, _, _, event = _buffer.dispatch(grad_combined_x, handle=handle, async_finish=True,
                                                 previous_event=previous_event,
                                                 allocate_on_comm_stream=previous_event is not None)

    # event 管理请参考 `EventOverlap` 类文档
    return grad_x, event
```

此外，在 dispatch 函数内部，当前 rank 可能不知道需要接收多少 token。因此会涉及一次隐式 CPU 等待 GPU 接收计数信号，如下图所示。

![normal](../figures/normal.png)

### 用于推理 decoding

低延迟 kernels 可以用于推理 decoding 阶段，如下方示例代码所示。

```python
import torch
import torch.distributed as dist
from typing import Tuple, Optional

from deep_ep import Buffer

# 通信 buffer（将在运行时分配）
# 说明：low-latency kernels 没有 SM 控制 API
_buffer: Optional[Buffer] = None


# 可以在框架初始化时调用该函数
def get_buffer(group: dist.ProcessGroup, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> Buffer:
    # 说明：low-latency 模式比 normal 模式消耗更多空间
    # 因此建议 `num_max_dispatch_tokens_per_rank`（decoding engine 中的实际 batch size）小于 256
    global _buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts)

    # 如果 buffer 不存在或大小不足，则分配一个 buffer
    if _buffer is None or _buffer.group != group or not _buffer.low_latency_mode or _buffer.num_rdma_bytes < num_rdma_bytes:
        # 说明：为了获得最佳性能，QP 数量**必须**等于 local experts 数量
        assert num_experts % group.size() == 0
        _buffer = Buffer(group, 0, num_rdma_bytes, low_latency_mode=True, num_qps_per_rank=num_experts // group.size())
    return _buffer


def low_latency_dispatch(hidden_states: torch.Tensor, topk_idx: torch.Tensor, num_max_dispatch_tokens_per_rank: int, num_experts: int):
    global _buffer

    # 执行 MoE dispatch，兼容 CUDA graph（但 replay 后可能需要恢复一些 buffer 状态）
    recv_hidden_states, recv_expert_count, handle, event, hook = \
        _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                                     async_finish=False, return_recv_hook=True)

    # 说明：只有在调用 `hook()` 后，实际 tensor 才会被接收。
    # 这对 double-batch overlapping 很有用，并且**不占用任何 SM**
    # 如果不需要 overlap，请设置 `return_recv_hook=False`
    # 后续可以使用我们的 GEMM library 基于这种特定格式执行计算
    return recv_hidden_states, recv_expert_count, handle, event, hook


def low_latency_combine(hidden_states: torch.Tensor,
                        topk_idx: torch.Tensor, topk_weights: torch.Tensor, handle: Tuple):
    global _buffer

    # 执行 MoE combine，兼容 CUDA graph（但 replay 后可能需要恢复一些 buffer 状态）
    combined_hidden_states, event_overlap, hook = \
        _buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle,
                                    async_finish=False, return_recv_hook=True)

    # 说明：行为与 dispatch kernel 中描述的一致
    return combined_hidden_states, event_overlap, hook
```

关于 two-micro-batch overlapping，可以参考下图。通过接收 hook 接口，RDMA 网络流量会在后台发生，不会从计算部分消耗任何 GPU SM。但需要注意，overlap 的部分可以调整，也就是说 attention/dispatch/MoE/combine 这 4 个部分不一定具有完全相同的执行时间。可以根据 workload 调整 stage 设置。

![low-latency](../figures/low-latency.png)

## Roadmap（V1）

- [x] AR 支持
- [x] 重构 low-latency mode AR 代码
- [x] A100 支持（仅 intranode）
- [x] 支持 low-latency dispatch kernel 的 BF16
- [x] 支持 intranode low-latency kernels 的 NVLink 协议
- [ ] 用 TMA copy 替代 LD/ST
    - [x] Intranode kernels
    - [ ] Internode kernels
    - [ ] Low-latency kernels
- [ ] SM-free kernels 和重构
- [ ] 完全移除未定义行为 PTX 指令

## 注意事项

#### 更简单的潜在整体设计

V1 实现为通信 buffers 使用了 queues，这可以节省内存，但也引入了复杂性和潜在死锁。如果你基于 DeepEP V1 实现自己的版本，可以考虑使用按最大容量分配的固定大小 buffers，以获得更简单的实现和更好的性能。关于这种替代方案的详细讨论，见 https://github.com/deepseek-ai/DeepEP/issues/39。

#### 未定义行为 PTX 用法

- 为了极致性能，我们发现并使用了一种未定义行为 PTX 用法：使用 read-only PTX `ld.global.nc.L1::no_allocate.L2::256B` 来**读取 volatile data**。PTX modifier `.nc` 表示使用 non-coherent cache。但在 Hopper 架构上，经过测试，配合 `.L1::no_allocate` 可以保证正确性，并显著提升性能。我们猜测原因可能是：non-coherent cache 与 L1 统一，而 L1 modifier 不只是 hint，而是一个强选项，因此通过确保 L1 中没有 dirty data 可以保证正确性。
- 一开始，因为 NVCC 无法自动 unroll volatile read PTX，我们尝试使用 `__ldg`（即 `ld.nc`）。即使与手动 unroll 的 volatile reads 相比，它也明显更快，可能是因为额外的编译器优化。然而结果可能不正确或出现 dirty data。查阅 PTX 文档后，我们发现 Hopper 架构上的 L1 和 non-coherent cache 是统一的。我们推测 `.L1::no_allocate` 可能解决该问题，并由此得到这一发现。
- 如果发现 kernels 在其他平台上无法工作，可以在 `setup.py` 中添加 `DISABLE_AGGRESSIVE_PTX_INSTRS=1` 来禁用该用法，或者提交 issue。

#### 在你的集群上 auto-tuning

为了在你的集群上获得更好性能，建议运行所有测试，并使用最佳 auto-tuned 配置。默认配置是在 DeepSeek 内部集群上优化得到的。
