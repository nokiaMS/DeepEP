# DeepEP

DeepEP（DeepEveryParallel）是一个面向现代机器学习训练和推理的高性能通信库。当前该库主要聚焦专家并行（EP），提供高吞吐、低延迟的 all-to-all GPU kernels（MoE dispatch 和 combine），并支持 FP8 等低精度；同时也提供用于流水线并行（PP）、上下文并行（CP）和远程内存访问（Engram）的实验性原语，这些能力都以零 SM 占用或最小 SM 占用为设计目标。所有 kernels 都通过轻量级 Just-In-Time（JIT）模块在运行时编译，安装时不需要 CUDA 编译。

尽管 DeepEP 设计轻量，但在多种配置下，它的性能能够达到甚至超过硬件带宽上限。

## 新闻

- **V2 发布**：专家并行的完整重构。与 V1 相比，V2 在使用少数倍 SM 资源的情况下实现了极致性能，同时支持显著更大的 scale-up 和 scale-out 域。V2 也已经从 NVSHMEM 后端切换到更轻量的 **NCCL Gin 后端**。

### 新特性

- **完全 JIT 化**（Just-In-Time compilation，即即时编译）
- **NCCL Gin 后端**
  - Header-only，轻量化
  - 能够复用已有的 NCCL communicators
- **EPv2**
  - 高吞吐和低延迟 API 统一到单一的 `ElasticBuffer` 接口，并采用新的 GEMM 布局
  - 支持更大的 scale-up 和 scale-out 域，最高可达 EP2048
  - SM 和 QP 数量可解析计算，不再需要 auto-tuning
  - 仍然支持 hybrid 和 direct 两种模式
  - 对于类似 V3 的 legacy training 场景，在保持相当或更好性能的同时，SM 使用量从 24 个降低到 4-6 个
- **0 SM Engram**（基于 RDMA）
- **0 SM PP**（基于 RDMA）
- **0 SM CP**（基于 Copy Engine）

### 说明

- Buffer 大小消耗比 V1 更大
- 不再支持 0 SM RDMA low-latency EP
- Engram、PP 和 CP 都是实验性功能

### 仍在进行中的功能

- **弹性 GPU & CPU buffers**：一个连续的虚拟地址空间，底层映射到 GPU 物理内存和 CPU 物理内存的混合区域，从而支持完全自动、透明的 Engram 或不均衡 EP
- 通过 EP replay 处理负载不均衡，从而减少中间 buffer 大小
- 面向 DP 和 TP 的 all-gather 更新，以及 reduce-scatter 实现

legacy V1 文档（基于 NVSHMEM）见 [docs/legacy.md](docs/legacy.md)。

## 性能

按照 V3 的配置，DeepEP 使用每 batch 8K tokens、7168 hidden dimensions、top 8 experts、FP8 dispatching 和 BF16 combining 进行测试，得到以下结果：

| Arch | NIC type | Topo | Dispatch Bottleneck Bandwidth | Combine Bottleneck Bandwidth | #SMs |
|--|--|--|--|--|--|
| SM90 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 | EP 8 x 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (Max perf) |
| SM100 | N/A | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (Min #SM) |

说明：这些结果是逻辑带宽。例如在 `EP 8 x 2` 场景下，90 GB/s 实际上包含了 local rank 流量。

与 V1 相比，**V2 最高可达到 1.3 倍峰值性能，同时最多节省 4 倍 SM 数量**。

DeepEP 暂时省略了更大 EP 配置下的结果，但鼓励感兴趣的用户自行 benchmark。基于内部经验，团队预计 kernel 在更大规模下仍会继续接近或打满硬件带宽。

V1 性能数据见 [docs/legacy.md](docs/legacy.md#performance)。

## 快速开始

### 环境要求

- Hopper（SM90）GPU，或者其他支持 SM90 PTX ISA 的架构
- Python 3.8 及以上
- CUDA 版本
  - 对于 SM90 GPU，需要 CUDA 12.3 及以上
- PyTorch 2.10 及以上
- NCCL 2.30.4 及以上
- 节点内通信需要 NVLink
- 节点间通信需要 RDMA 网络

### 安装 NCCL 依赖

DeepEP 推荐使用 pip 安装 NCCL，这样 DeepEP 可以在 Python 环境中自动定位 NCCL。可以使用以下命令安装：

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

### 安装 NVSHMEM 依赖

DeepEP 也依赖 NVSHMEM 来支持 legacy 方法。具体说明请参考 [NVSHMEM Installation Guide](docs/nvshmem.md)。

### 开发

```bash
# 构建并为 SO 文件创建符号链接
python setup.py build
# 可以根据自己的平台修改具体的 SO 文件名
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# 运行测试用例
# 说明：可以根据自己的集群设置修改 `tests/utils/envs.py` 中的 `init_dist` 函数，
# 并在多节点上启动
python tests/elastic/test_ep.py
python tests/elastic/test_agrs.py
python tests/elastic/test_engram.py
python tests/elastic/test_pp.py
```

### 安装

```bash
python setup.py install
```

然后，在你的 Python 项目中导入 `deep_ep` 即可使用。

## 接口和示例

### Buffer 初始化

在 V2 中，所有 EP 操作，不论是高吞吐还是低延迟，都统一到单一的 `ElasticBuffer` 接口下。可以通过直接指定 MoE 设置来初始化 buffer，并且最优 SM 和 QP 数量会被解析计算出来。

```python
import torch
import torch.distributed as dist
from typing import Optional

from deep_ep import ElasticBuffer

# 通信 buffer（将在运行时分配）
_buffer: Optional[ElasticBuffer] = None

# 通信 kernel 使用的 SM 数量（将在创建 buffer 时设置）
_num_comm_sms: int = 0


def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> ElasticBuffer:
    """初始化或获取用于 EP 通信的 ElasticBuffer。"""
    global _buffer, _num_comm_sms

    # 检查是否可以复用已有 buffer
    required_bytes = ElasticBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        return _buffer

    # 使用 MoE 设置分配新的 buffer
    # 说明：V2 的 buffer 大小消耗比 V1 更大
    _buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # V2 会解析计算最优 SM 数量，不再需要 auto-tuning
    # 也可以在 dispatch/combine 调用中手动指定 `num_sms` 来覆盖
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
```

### 用于模型训练或推理 prefilling

V2 将 `dispatch` 和 `combine` API 统一到单一的 `ElasticBuffer` 接口。下面的示例展示了如何在训练（包含 backward pass）或推理 prefilling 中使用它们。

```python
import torch
import torch.distributed as dist
from typing import Tuple, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int,
                     num_max_tokens_per_rank: int,
                     expert_alignment: int = 1) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    MoE dispatch：将 token 路由到所有 rank 上对应的 expert。
    同时支持 BF16 和 FP8 输入；FP8 输入时，x 是 [data, scale_factors] 形式的 tuple。
    """
    global _buffer, _num_comm_sms

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    # `handle` 包含后续 combine 调用所需的路由元数据
    # `handle.num_recv_tokens_per_expert_list` 提供每个 expert 的 token 数量，可供 GEMM 使用
    # 使用 `event.current_stream_wait()` 在使用结果前同步 compute stream
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor,
                      grad_recv_topk_weights: torch.Tensor,
                      handle: EPHandle) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    """MoE dispatch 的 backward pass 本质上是一次 combine。"""
    global _buffer, _num_comm_sms

    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_topk_weights,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
                    handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """MoE combine：将 expert 输出归并回原始 rank。"""
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: EPHandle) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    """MoE combine 的 backward pass 本质上是一次 dispatch。"""
    global _buffer, _num_comm_sms

    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return grad_x, event
```

为了实现通信和计算 overlap，可以使用 `EventOverlap` 接口管理 communication stream 和 compute stream 之间的依赖关系：

```python
# dispatch 后，通信仍在进行时可以执行其他独立计算
recv_x, recv_topk_idx, recv_topk_weights, handle, event = dispatch_forward(...)

# ... 在这里执行一些独立计算 ...

# 使用结果前，等待通信完成
event.current_stream_wait()

# 现在可以安全使用 recv_x、recv_topk_idx、recv_topk_weights
```

### 用于推理 decoding

推理 decoding 也使用同一个 `ElasticBuffer`。当 gating 决策保持不变时，可以通过 handle-caching 模式复用路由元数据，避免重复 CPU 同步。

```python
import torch
from typing import Tuple, Optional, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def decode_dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                    topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                    num_experts: int,
                    num_max_tokens_per_rank: int,
                    cached_handle: Optional[EPHandle] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    推理 decoding 中的 MoE dispatch。
    如果提供 `cached_handle`，则复用布局，不需要 CPU 同步。
    """
    global _buffer, _num_comm_sms

    if cached_handle is not None:
        # 复用 cached handle：跳过布局重算和 CPU 同步
        recv_x, _, _, handle, event = _buffer.dispatch(
            x,
            handle=cached_handle,
            num_sms=_num_comm_sms,
            async_with_compute_stream=True,
        )
        return recv_x, cached_handle.topk_idx, None, handle, event

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def decode_combine(x: torch.Tensor,
                   handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """推理 decoding 中的 MoE combine。"""
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event
```

### 环境变量

该库提供了一些可能有用的环境变量：

- 通用
    - `EP_BUFFER_DEBUG`：`0` 或 `1`，打印 buffer 初始化、SM 近似计算和后端调试信息，默认 `0`
    - `EP_SUPPRESS_NCCL_CHECK`：`0` 或 `1`，关闭 NCCL 版本不匹配检查，默认 `0`
    - `EP_AVOID_RECORD_STREAM`：`0` 或 `1`，避免对输出 tensor 调用 `record_stream`，默认 `0`
    - `EP_NUM_TOPK_IDX_BITS`：整数，覆盖 top-k index 编码使用的 bit 数，默认 `0`，表示自动选择
- 网络
    - `EP_NIC_NAME`：字符串，用于查询 NIC 属性的默认 NIC 名称，默认 `mlx5_0`
    - `EP_OVERRIDE_RDMA_SL`：整数，覆盖 RDMA service level index，用于流量隔离
    - `EP_DISABLE_GIN`：`0` 或 `1`，禁用 NCCL Gin 后端，回退到非 Gin 路径，默认 `0`
- JIT
    - `EP_JIT_DEBUG`：`0` 或 `1`，打印 JIT 调试信息，默认 `0`
    - `EP_JIT_CACHE_DIR`：字符串，已编译 kernel 的缓存目录，默认 `$HOME/.deep_ep`
    - `EP_JIT_NVCC_COMPILER`：字符串，NVCC 编译器路径；默认使用 `torch.utils.cpp_extension.CUDA_HOME`
    - `EP_JIT_CPP_STANDARD`：整数，C++ 标准版本，默认 `20`
    - `EP_JIT_PRINT_COMPILER_COMMAND`：`0` 或 `1`，打印编译命令，默认 `0`
    - `EP_JIT_PTXAS_VERBOSE`：`0` 或 `1`，显示详细 PTXAS 输出，默认 `0`
    - `EP_JIT_PTXAS_CHECK`：`0` 或 `1`，断言编译出的 kernel 没有使用 local memory，默认 `0`
    - `EP_JIT_WITH_LINEINFO`：`0` 或 `1`，为 profiling 工具嵌入源码行号信息，默认 `0`
    - `EP_JIT_DUMP_ASM`：`0` 或 `1`，dump PTX 和 SASS，默认 `0`
    - `EP_JIT_DUMP_PTX`：`0` 或 `1`，dump PTX 输出，默认 `0`
    - `EP_JIT_DUMP_SASS`：`0` 或 `1`，dump SASS 输出，默认 `0`
- 调试和 profiling
    - `EP_GIN_GDAKI_DEBUG`：`0` 或 `1`，开启 NCCL Gin GDAKI 调试输出，默认 `0`
    - `EP_USE_NVIDIA_TOOLS`：`0` 或 `1`，在外部 NVIDIA 工具下运行时跳过内部 profiling，默认 `0`
    - `EP_DISABLE_BARRIER_PROFILING`：`0` 或 `1`，在 benchmark 中禁用基于 barrier 的通信 profiling，默认 `0`
- 构建
    - `EP_NCCL_ROOT_DIR`：字符串，NCCL 安装目录路径；如果未设置，则从 Python 环境自动检测
    - `EP_NVSHMEM_ROOT_DIR`：字符串，NVSHMEM 安装目录路径；如果未设置，则从 Python 环境自动检测
    - `TORCH_CUDA_ARCH_LIST`：字符串，目标 CUDA 架构列表，例如 `"9.0"`
    - `DISABLE_SM90_FEATURES`：`0` 或 `1`，为 legacy 方法禁用 SM90 特性，默认 `0`
    - `DISABLE_AGGRESSIVE_PTX_INSTRS`：`0` 或 `1`，为 legacy 方法禁用激进的 load/store 指令，默认 `0`

部分环境变量是**持久化**的：它们会在构建时被捕获，并作为默认值写入安装后的 package。导入时，如果当前环境变量没有覆盖这些值，就会自动应用构建时保存的默认值。持久化变量包括：`EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_NUM_TOPK_IDX_BITS`、`EP_NCCL_ROOT_DIR`。

更多细节请参考[测试代码](tests/elastic/test_ep.py)，或查看对应的 Python 文档。

## 网络配置

DeepEP 已经在 InfiniBand 网络上完成充分测试。不过从理论上讲，它也兼容 RDMA over Converged Ethernet（RoCE）。

### 流量隔离

InfiniBand 通过 Virtual Lanes（VL）支持流量隔离。

为了避免不同类型流量互相干扰，建议将工作负载隔离到不同的 virtual lanes：

- expert-parallel workloads
- other workloads

对于 DeepEP V2，可以通过设置 `sl_idx` 参数或 `EP_OVERRIDE_RDMA_SL` 环境变量来控制 virtual lane 分配。

### 自适应路由

自适应路由是 InfiniBand 交换机提供的高级路由能力，可以把流量均匀分布到多条路径上。虽然自适应路由会引入额外延迟，但仍建议在所有网络负载条件下启用它。

### 拥塞控制

建议关闭拥塞控制，因为它会损害最大带宽。如果某些场景中无法避免拥塞，建议将这些 workload 分配到低优先级 virtual lanes。

### PCI atomic mode

如果硬件支持，建议使用以下命令设置 NIC 的 `PCI_ATOMIC_MODE`，以提升 RDMA atomic 操作性能：

```bash
sudo mlxconfig -y -d mlx5_$i set PCI_ATOMIC_MODE=4
```

## 实验分支

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
    - 移除 PyTorch tensors 和通信 buffers 之间的拷贝，从而显著降低普通 kernel 的 SM 使用量
    - 该 PR 由 **腾讯网络平台部** 贡献
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
    - 使用低延迟协议，移除 RDMA atomic OPs 引入的额外 RTT 延迟
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
    - 使用 TMA 指令实现新的后端，以实现最小 SM 使用量，并支持更大的 NVLink 域
    - 针对 single-batch 场景提供细粒度通信-计算 overlap
    - 支持非 NVLink 环境下的 PCIe kernel
    - 支持 NVFP4 数据类型
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
    - 该优化系列由 **蚂蚁集团网络平台部** 贡献
    - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347)：通过解耦通信 kernel 执行和 NIC token 传输，从 RDMA 路径中消除 SM 占用，把 SM 释放给计算
    - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483)：通过 signaling 机制将 Down GEMM 计算与 Combine Send 通信 overlap，以降低端到端延迟
    - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500)：通过 rail-optimized forwarding 和 data merging 优化跨节点 LL operator 通信，以降低延迟
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
    - 基于 [MORI](https://github.com/ROCm/mori) 后端提供 ROCm / AMD GPU 支持，当前面向 low-latency 模式

## 社区 Fork

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep)：支持在异构 GPU 和 NIC 上运行 DeepEP，例如 NVIDIA、AMD GPU，以及 EFA、Broadcom、CX7 等 NIC
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP)：在 IBRC transport 中增加 multi-QP 方案和双端口 NIC 支持
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace)：用于高效、精确定位 slow ranks 的诊断分析器
- [ROCm/mori](https://github.com/ROCm/mori)：AMD 面向性能关键 AI 工作负载的下一代通信库，例如 Wide EP、KVCache transfer、Collectives

## 致谢

DeepEP V2 构建在 [NCCL](https://github.com/nvidia/nccl) Gin 后端之上。感谢 @sjeaugey、@pakmarkthub、@sb17v、@xiaofanl-nvidia 以及 NCCL 团队的支持！

## 许可证

该代码仓库基于 [MIT License](LICENSE) 发布。

## 引用

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
