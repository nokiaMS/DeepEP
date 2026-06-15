# DeepEP 学习手册

本文基于本机源码目录 `E:\codex_home\code\DeepEP` 整理，目标是帮助读者从“怎么用”逐步过渡到“源码如何实现”。DeepEP 当前主线是 V2 `ElasticBuffer`，用于 MoE 专家并行通信；V1 `Buffer` 仍保留在 legacy 路径中，主要用于兼容旧接口和理解历史实现。

## 1. DeepEP 是什么

DeepEP，全称 DeepEveryParallel，是面向大模型训练和推理的高性能 GPU 通信库。它最核心的工作负载是 MoE（Mixture of Experts）里的专家并行通信，也就是把 token 按 gate/top-k 路由到不同 rank 上的 expert，再把 expert 计算结果规约回原始 token 所在 rank。

在 MoE 层中，典型流程是：

1. Router/gate 为每个 token 选择 top-k experts。
2. `dispatch` 根据 expert 所属 rank 把 token 发送到目标 rank。
3. 每个 rank 在本地 expert 上执行 GEMM。
4. `combine` 把 expert 输出按原始 token 和 top-k 权重规约回原 rank。

DeepEP 主要优化第 2 步和第 4 步。它不是通用 collective 库，而是为 MoE dispatch/combine 这种细粒度、稀疏、all-to-all 风格的数据搬运定制的通信运行时。

## 2. 当前源码版本的主线能力

从 `README.md` 和源码看，当前 DeepEP V2 的重要特征是：

- 统一 API：高吞吐和低延迟 EP 都收敛到 `ElasticBuffer`。
- 后端切换：V2 主线基于 NCCL Gin，legacy V1 基于 NVSHMEM/IBGDA。
- 运行时 JIT：V2 kernel 通过 `csrc/jit/` 在运行时编译，安装时不需要把所有 CUDA kernel 都预编译进去。
- 自动建模：V2 根据通信拓扑、专家数、top-k、带宽估计 SM/QP 数，不再主要依赖手工 auto-tuning。
- 支持多类实验能力：Engram 远程内存访问、PP send/recv、AGRS all-gather/reduce-scatter 相关原语。

需要注意：

- V2 buffer 占用通常比 V1 大。
- V2 不再支持 0 SM RDMA low-latency EP；legacy V1 仍保留 low-latency 接口。
- Engram、PP、AGRS/CP 类能力在 README 中明确属于实验性质。

## 3. 代码目录速览

核心目录如下：

```text
DeepEP/
├── deep_ep/                         # Python 包入口和用户 API 封装
│   ├── __init__.py                  # 导入 _C、检查 NCCL、初始化 JIT、导出 API
│   ├── buffers/
│   │   ├── elastic.py               # V2 ElasticBuffer Python 封装
│   │   └── legacy.py                # V1 Buffer Python 封装
│   ├── include/deep_ep/             # V2 JIT CUDA header-only kernel
│   │   ├── common/                  # 公共 CUDA/PTX/layout/comm 工具
│   │   └── impls/                   # dispatch/combine/barrier/engram/PP kernel 实现
│   └── utils/                       # 分布式初始化、NCCL handle、参考实现、测试工具
├── csrc/                            # C++/CUDA extension 源码
│   ├── python_api.cpp               # pybind11 模块入口
│   ├── elastic/buffer.hpp           # V2 C++ runtime 和 pybind 注册
│   ├── legacy/buffer.hpp            # V1 C++ runtime 和 pybind 注册
│   ├── kernels/backend/             # NCCL/NVSHMEM/CUDA driver 后端
│   ├── kernels/elastic/             # V2 JIT kernel launch runtime
│   ├── kernels/legacy/              # V1 预编译 CUDA kernel
│   └── jit/                         # JIT 编译、缓存、动态库加载
├── tests/elastic/                   # V2 测试与 benchmark
├── tests/legacy/                    # V1 测试
├── docs/                            # legacy 和 NVSHMEM 文档
├── setup.py                         # PyTorch CUDAExtension 构建入口
└── README.md                        # 官方说明
```

最重要的阅读入口：

- `deep_ep/buffers/elastic.py`
- `csrc/elastic/buffer.hpp`
- `csrc/kernels/elastic/dispatch.hpp`
- `csrc/kernels/elastic/combine.hpp`
- `deep_ep/include/deep_ep/impls/dispatch.cuh`
- `deep_ep/include/deep_ep/impls/combine.cuh`
- `tests/elastic/test_ep.py`

## 4. Python 包导入流程

`deep_ep/__init__.py` 是用户 `import deep_ep` 时的入口，做了几件关键事情：

1. 读取构建时持久化的环境变量。
2. 查找 CUDA home。
3. 检查当前进程加载的 NCCL so 是否和 DeepEP 链接的 NCCL 一致。
4. 调用 `_C.init_jit(library_root_path, cuda_home, nccl_root)` 初始化 JIT。
5. 导出 `Buffer`、`ElasticBuffer`、`EPHandle`、`EventOverlap`、`Config`、`topk_idx_t` 等 API。

调用链可以概括为：

```text
import deep_ep
  -> deep_ep/__init__.py
  -> check_nccl_so()
  -> init_jit()
  -> import deep_ep._C
  -> 导出 Python API
```

`csrc/python_api.cpp` 是 `_C` 扩展模块入口，注册内容包括：

- `is_sm90_compiled`
- `topk_idx_t`
- JIT API：`init_jit`
- legacy buffer API
- elastic buffer API

## 5. 构建与依赖

构建入口是 `setup.py`。它使用 `torch.utils.cpp_extension.CUDAExtension` 生成 `deep_ep._C`。

关键构建依赖：

- CUDA
- PyTorch
- NCCL
- NVSHMEM，主要服务 legacy V1 路径
- third-party/fmt

`setup.py` 的几个要点：

- `find_pkgs.py` 自动查找 NCCL/NVSHMEM 安装路径。
- `persistent_env_names` 会把部分环境变量写入构建产物的 `deep_ep/envs.py`。
- 默认 `TORCH_CUDA_ARCH_LIST=9.0`，面向 Hopper/SM90。
- V2 kernel 多数不是作为普通 `.cu` 编译进 extension，而是以 header 形式进入 package，运行时由 JIT 编译。
- legacy kernel 的 `layout.cu`、`intranode.cu`、`internode.cu`、`internode_ll.cu` 会参与 extension 构建。

常见命令：

```bash
python setup.py build
python setup.py install
```

README 推荐使用 pip 安装 NCCL，例如：

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

## 6. V2 核心对象：ElasticBuffer

`ElasticBuffer` 定义在 `deep_ep/buffers/elastic.py`，是 V2 的统一入口。它封装以下能力：

- EP dispatch/combine
- barrier
- Engram fetch/write
- PP send/recv
- AGRS all-gather 相关接口
- 通信 stream/event 管理
- SM/QP 数估算

构造函数核心参数：

```python
deep_ep.ElasticBuffer(
    group,
    num_bytes=None,
    num_cpu_bytes=0,
    num_max_tokens_per_rank=0,
    hidden=0,
    num_topk=0,
    use_fp8_dispatch=False,
    deterministic=False,
    allow_hybrid_mode=True,
    allow_multiple_reduction=True,
    prefer_overlap_with_compute=True,
    sl_idx=3,
    num_allocated_qps=0,
    explicitly_destroy=False,
)
```

两种 buffer 创建方式：

1. 通过 MoE 参数自动计算：

```python
buffer = deep_ep.ElasticBuffer(
    group,
    num_max_tokens_per_rank=num_tokens,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=True,
)
```

2. 通过显式字节数创建：

```python
num_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
    group, num_tokens, hidden, num_topk, use_fp8_dispatch=True
)
buffer = deep_ep.ElasticBuffer(group, num_bytes=num_bytes)
```

构造时的关键内部步骤：

```text
ElasticBuffer.__init__
  -> get_nccl_comm_handle(group)
  -> calculate_elastic_buffer_size(...)
  -> check_nvlink_connections(group)
  -> 自动决定 num_allocated_qps
  -> 可选创建 CPU symmetric memory handle
  -> _C.ElasticBuffer(...)
  -> get_logical_domain_size()
  -> get_physical_domain_size()
  -> cuda synchronize + group barrier
```

## 7. EPHandle 的作用

`EPHandle` 是 `dispatch` 返回的通信元数据容器，后续 `combine` 必须依赖它把数据规约回原始 token。

重要字段：

- `topk_idx`：dispatch 时使用的 expert 路由。
- `num_experts`：全局 expert 数。
- `expert_alignment`：每个 local expert 接收 token 数的对齐粒度。
- `num_max_tokens_per_rank`：每 rank 最大 token 数。
- `num_sms`：dispatch 使用的 SM 数，combine 默认复用。
- `psum_num_recv_tokens_per_scaleup_rank`：按 scale-up rank 统计的接收 token 前缀和。
- `psum_num_recv_tokens_per_expert`：按 local expert 统计的前缀和。
- `num_recv_tokens_per_expert_list`：CPU 侧每个 local expert 接收数量。
- `recv_src_metadata`：源 token 和 buffer slot 等 metadata。
- `dst_buffer_slot_idx`：dispatch 目标 buffer slot。
- `token_metadata_at_forward`、`channel_linked_list`：hybrid mode 使用。

它有两个核心用途：

1. 作为 `combine(handle=...)` 的必需参数。
2. 作为 cached dispatch 的输入，跳过 layout 重算和 CPU 同步。

## 8. Dispatch 数据流

`ElasticBuffer.dispatch` 的输入通常是：

- `x`：`[num_tokens, hidden]`，BF16；或 FP8 模式下 `(x_fp8, scale_factors)`。
- `topk_idx`：`[num_tokens, num_topk]`，dtype 为 `deep_ep.topk_idx_t`。
- `topk_weights`：`[num_tokens, num_topk]`，float。
- `num_experts`
- `num_max_tokens_per_rank`
- `expert_alignment`

普通调用：

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    expert_alignment=128,
    async_with_compute_stream=True,
)
event.current_stream_wait()
```

内部主要流程：

```text
ElasticBuffer.dispatch
  -> check_torch_deterministic()
  -> 自动确定 num_sms / num_qps
  -> 拆分 FP8 input: x, sf
  -> 如果传入 handle，进入 cached mode
  -> self.runtime.dispatch(...)
  -> C++ ElasticBuffer::dispatch
  -> launch_dispatch(...)
  -> 分配 recv_x / recv_topk_idx / recv_topk_weights / metadata
  -> dispatch copy epilogue
  -> 构造 EPHandle
  -> 返回 EventOverlap
```

dispatch 有三类常见模式：

- 普通模式：传入 `topk_idx/topk_weights`，计算 layout 并通信。
- cached 模式：传入旧 `handle`，复用 layout，要求不再传 `topk_idx/topk_weights`。
- expanded 模式：`do_expand=True`，把 token 按 expert 槽位展开，更贴近后续 expert GEMM 输入布局。

## 9. Combine 数据流

`ElasticBuffer.combine` 用于把 expert 输出规约回原始 token 所在 rank。

普通调用：

```python
combined_x, combined_topk_weights, event = buffer.combine(
    expert_output,
    handle=handle,
    topk_weights=recv_topk_weights,
    async_with_compute_stream=True,
)
event.current_stream_wait()
```

内部主要流程：

```text
ElasticBuffer.combine
  -> check_torch_deterministic()
  -> num_sms 默认复用 handle.num_sms
  -> 自动确定 num_qps
  -> 拆分 bias
  -> self.runtime.combine(...)
  -> C++ ElasticBuffer::combine
  -> launch_combine(...)
  -> combine reduce epilogue
  -> 返回 combined_x / combined_topk_weights / EventOverlap
```

从训练反向传播角度看：

- dispatch 的 backward 本质是 combine。
- combine 的 backward 本质是 dispatch。

这也是 README 示例里的组织方式。

## 10. SM 和 QP 的估算

V2 的一个关键变化是解析式估算 SM/QP 数。

`ElasticBuffer.get_theoretical_num_sms(num_experts, num_topk, ...)` 做的事情：

- 根据物理/逻辑通信域推导 scale-up 和 scale-out 流量。
- 使用 `get_nvlink_gbs()` 和 `get_rdma_gbs()` 获取链路带宽估计。
- 估算每个 token 的 SM read/write 压力。
- 找到通信瓶颈链路。
- 得出推荐 SM 数，至少 4 个且对齐到偶数。
- 如果 `prefer_overlap_with_compute=False`，倾向于使用更多 SM 提升纯通信性能。

`ElasticBuffer.get_theoretical_num_qps(num_sms)` 做的事情：

- direct mode 下倾向少量 QP，降低 doorbell/调度开销。
- hybrid mode 下按 channel 分配更多 QP，公式近似为 `num_sms * 16 + 1`。
- 最终不超过构造时的 `num_allocated_qps`。

理解方式：

- SM 数决定通信 kernel 占用多少 GPU 计算资源。
- QP 数决定 RDMA 并行通道数量。
- SM/QP 太少可能打不满链路，太多会影响计算 overlap 或引入调度开销。

## 11. 通信域：physical 与 logical

DeepEP 区分物理通信域和逻辑通信域。

物理通信域：

- `num_rdma_ranks`：RDMA 维度。
- `num_nvlink_ranks`：NVLink 维度。
- 由 `_C.get_physical_domain_size()` 获取。

逻辑通信域：

- `num_scaleout_ranks`：跨节点/scale-out 维度。
- `num_scaleup_ranks`：节点内/scale-up 维度。
- 由 `_C.get_logical_domain_size()` 获取。

hybrid mode 下，dispatch/combine 通常会把跨节点 RDMA 和节点内 NVLink 分层组织，以便更好利用多 rail/multi-plane 网络和节点内高带宽。

## 12. EventOverlap 与异步执行

`EventOverlap` 定义在 `deep_ep/utils/event.py`，用于协调 communication stream 与当前 compute stream。

常见用法：

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
    async_with_compute_stream=True,
)

# 这里可以执行与 recv_x 无关的计算

event.current_stream_wait()
# 这里开始安全使用 recv_x
```

相关参数：

- `previous_event`：让通信 kernel 等待某个之前的 event。
- `previous_event_before_epilogue`：让 copy/reduce epilogue 等待某个 event。
- `async_with_compute_stream=True`：当前 stream 不等待通信完成，用户通过 event 显式同步。
- `allocate_on_comm_stream=True`：输出 tensor 的 ownership 绑定到通信 stream。

测试文件 `tests/elastic/test_ep.py` 会枚举这些组合，验证正确性。

## 13. FP8 dispatch

DeepEP 支持 dispatch 阶段使用 FP8，以降低通信量。

Python 工具函数：

- `deep_ep.utils.math.per_token_cast_to_fp8`
- `deep_ep.utils.math.per_token_cast_back`

FP8 输入形式：

```python
x_fp8, x_scales = per_token_cast_to_fp8(x_bf16)
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    (x_fp8, x_scales),
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
)
```

FP8 dispatch 返回的 `recv_x` 也是 tuple：

```python
recv_x_fp8, recv_scales = recv_x
recv_x_bf16 = per_token_cast_back(recv_x_fp8, recv_scales)
```

在 expanded mode 中，还可以设置 `use_tma_aligned_col_major_sf=True`，让 scale factor layout 更适合 TMA/GEMM 侧使用。

## 14. Expanded dispatch

普通 dispatch 的输出按接收 token 排列；expanded dispatch 会进一步按 expert 槽位展开。

调用方式：

```python
expanded_recv_x, _, expanded_recv_topk_weights, expanded_handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    expert_alignment=128,
    do_expand=True,
)
```

用途：

- 为每个 expert 准备更直接的输入布局。
- 减少后续整理 expert GEMM 输入的成本。
- combine 时可使用 expanded handle 执行 reduced combine。

`tests/elastic/test_ep.py` 里用 `fold_expanded()` 把 expanded 输出折叠回普通布局，用于和参考实现比较。

## 15. Cached dispatch

cached dispatch 适合推理 decoding 中 routing layout 可复用的场景。

第一次：

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
)
```

后续复用：

```python
recv_x, _, _, cached_handle, event = buffer.dispatch(
    x,
    handle=handle,
)
```

限制：

- cached 模式不能再传 `topk_idx` 和 `topk_weights`。
- 不能启用 CPU sync。
- `num_experts`、`expert_alignment`、`num_max_tokens_per_rank` 必须和原 handle 一致。

收益：

- 跳过 layout 重新计算。
- 避免部分 CPU 同步。
- 对 decoding 场景更友好。

## 16. C++ runtime 边界

V2 的 C++ runtime 在 `csrc/elastic/buffer.hpp`。Python 的 `self.runtime` 就是 `_C.ElasticBuffer` 实例。

主要职责：

- 创建和管理 NCCL Gin context/window。
- 管理 GPU/CPU symmetric memory。
- 执行 barrier。
- 发起 dispatch/combine/engram/PP/AGRS kernel。
- 分配 PyTorch tensor 输出。
- 做 shape、dtype、contiguous、CUDA device 等断言。

Python 与 C++ 边界大致是：

```text
deep_ep/buffers/elastic.py
  -> deep_ep._C.ElasticBuffer
  -> csrc/elastic/buffer.hpp
  -> csrc/kernels/elastic/*.hpp
  -> deep_ep/include/deep_ep/impls/*.cuh
```

阅读 `csrc/elastic/buffer.hpp` 时建议重点看：

- 构造函数：通信上下文、buffer、QPs、CPU memory 初始化。
- `dispatch(...)`：metadata、prefix sum、recv tensor、JIT dispatch launch。
- `combine(...)`：根据 handle metadata 反向规约。
- `engram_write/fetch(...)`
- `pp_send/pp_recv(...)`
- `all_gather(...)`
- `register_apis(...)`

## 17. JIT 机制

V2 kernel 通过 JIT 编译。初始化入口：

```text
deep_ep/__init__.py
  -> init_jit()
  -> _C.init_jit(...)
  -> csrc/jit/api.hpp
  -> Compiler::prepare_init
  -> KernelRuntime::prepare_init
  -> IncludeParser::prepare_init
```

JIT 相关目录：

- `csrc/jit/compiler.hpp`：封装 NVCC 编译命令、源码生成、编译选项。
- `csrc/jit/cache.hpp`：编译产物缓存。
- `csrc/jit/handle.hpp`：动态库句柄。
- `csrc/jit/include_parser.hpp`：解析 header 依赖。
- `csrc/jit/kernel_runtime.hpp`：kernel runtime 基类。
- `csrc/jit/launch_runtime.hpp`：kernel launch 模板封装。

V2 kernel 的源码实际放在：

- `deep_ep/include/deep_ep/common/*.cuh`
- `deep_ep/include/deep_ep/impls/*.cuh`

`csrc/kernels/elastic/*.hpp` 负责构造 launch runtime，把参数传给 JIT 编译出来的 kernel。

## 18. CUDA kernel 实现分层

V2 kernel 文件可以按功能分：

公共组件：

- `common/compiled.cuh`：编译期常量、类型、top-k index 类型。
- `common/comm.cuh`：设备侧通信、barrier、notify、RDMA/NVLink 辅助。
- `common/layout.cuh`：buffer/workspace layout。
- `common/handle.cuh`：NCCL Gin 设备侧 handle。
- `common/ptx.cuh`：内联 PTX，包含 mbarrier、load/store 等底层操作。
- `common/math.cuh`：对齐、数学、类型转换。

EP dispatch：

- `impls/dispatch_deterministic_prologue.cuh`
- `impls/dispatch.cuh`
- `impls/dispatch_copy_epilogue.cuh`
- `impls/hybrid_dispatch.cuh`

EP combine：

- `impls/combine.cuh`
- `impls/combine_reduce_epilogue.cuh`
- `impls/combine_utils.cuh`
- `impls/hybrid_combine.cuh`

其他实验能力：

- `impls/barrier.cuh`
- `impls/engram_fetch.cuh`
- `impls/engram_fetch_wait.cuh`
- `impls/pp_send_recv.cuh`

阅读顺序建议先看 host runtime，再看 CUDA header。直接从 `.cuh` 看起容易被底层 layout 和 PTX 细节淹没。

## 19. Legacy V1 Buffer

V1 入口是 `deep_ep/buffers/legacy.py` 的 `Buffer`，底层是 `csrc/legacy/buffer.hpp` 和 `csrc/kernels/legacy/*.cu`。

它支持：

- intranode all-to-all：主要走 NVLink。
- internode all-to-all：RDMA + NVLink。
- low-latency dispatch/combine：基于 NVSHMEM/IBGDA。

和 V2 的差异：

- V1 有独立的 `Config` tuning 配置。
- V1 需要显式 layout 计算，如 `get_dispatch_layout()`。
- V1 high-throughput 和 low-latency API 分开。
- V1 legacy kernel 是构建时编译进 extension 的 `.cu` 文件。
- V1 依赖 NVSHMEM 更重。

V1 常见接口：

- `Buffer.get_dispatch_config(num_ranks)`
- `Buffer.get_combine_config(num_ranks)`
- `Buffer.get_dispatch_layout(...)`
- `Buffer.dispatch(...)`
- `Buffer.combine(...)`
- `Buffer.low_latency_dispatch(...)`
- `Buffer.low_latency_combine(...)`

建议把 V1 当作历史对照：先学 V2，再用 V1 理解 DeepEP 为什么从手工配置、NVSHMEM backend 迁移到统一 `ElasticBuffer` 和 NCCL Gin。

## 20. Engram

Engram 是实验性的远程内存访问能力，用于从远端 rank 拉取 KV/cache 类条目。

测试入口：`tests/elastic/test_engram.py`

典型流程：

```python
num_gpu_bytes, num_cpu_bytes = deep_ep.ElasticBuffer.get_engram_storage_size_hint(
    num_entries, hidden, num_tokens, torch.bfloat16
)

buffer = deep_ep.ElasticBuffer(
    group,
    num_bytes=num_gpu_bytes + num_cpu_bytes,
    num_cpu_bytes=num_cpu_bytes,
    explicitly_destroy=True,
    num_allocated_qps=num_qps,
)

buffer.engram_write(local_storage)

hook = buffer.engram_fetch(indices)
fetched = hook()
```

理解要点：

- `engram_write(storage)` 把本 rank 的 storage 写入 NCCL window/symmetric memory。
- `engram_fetch(indices)` 发起远程 fetch，返回一个 hook。
- 调用 hook 才等待数据到达并得到 fetched tensor。
- 测试中用 `dist.all_gather_into_tensor` 构造 reference，对比 fetch 结果。

## 21. PP send/recv

PP send/recv 是实验性的 pipeline parallel 点对点通信接口。

测试入口：`tests/elastic/test_pp.py`

典型流程：

```python
num_bytes = deep_ep.ElasticBuffer.get_pp_buffer_size_hint(
    num_max_tensor_bytes,
    num_max_inflight_tensors,
)

buffer = deep_ep.ElasticBuffer(
    group,
    num_bytes=num_bytes,
    explicitly_destroy=True,
    allow_hybrid_mode=False,
)

buffer.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)
buffer.pp_send(tensor, dst_rank_idx)
buffer.pp_recv(output_tensor, src_rank_idx)
```

测试逻辑会随机生成 send/recv 操作序列，并验证接收到的 tensor 与发送 tensor 完全一致。

## 22. AGRS all-gather

AGRS 相关接口在 `ElasticBuffer` 中包括：

- `get_agrs_num_max_session_bytes`
- `get_agrs_buffer_size_hint`
- `agrs_set_config`
- `create_agrs_session`
- `destroy_agrs_session`
- `agrs_new_session`
- `agrs_get_inplace_tensor`
- `all_gather`

测试入口：`tests/elastic/test_agrs.py`

典型用法：

```python
num_session_bytes = deep_ep.ElasticBuffer.get_agrs_num_max_session_bytes(
    group, shapes, torch.bfloat16
)
num_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(group, num_session_bytes)

buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_bytes)
buffer.agrs_set_config(num_bytes, num_max_inflight_agrs)

with buffer.agrs_new_session():
    gathered, handle = buffer.all_gather(tensor)
    handle()
```

`all_gather` 返回的 handle 是等待函数，调用后才能保证数据完成。

## 23. 测试代码怎么读

最重要的测试是 `tests/elastic/test_ep.py`。它覆盖：

- BF16 和 FP8 dispatch。
- 普通 dispatch、expanded dispatch、cached dispatch。
- 普通 combine 和 reduced combine。
- handle copy 与 handle 复用。
- `previous_event`、`async_with_compute_stream`、`allocate_on_comm_stream`。
- deterministic 模式。
- CPU sync 与 no CPU sync。
- unbalanced gate 分布。
- masked top-k。
- Kineto profiling。

核心函数：

- `enumerate_ep_modes()`：枚举测试模式组合。
- `launch()`：统一注入 previous event 和异步 wait。
- `fold_expanded()`：把 expanded 输出折叠回普通 token 布局。
- `test_dispatch_combine()`：EP 正确性和性能主流程。
- `test_loop()`：初始化分布式、创建 buffer、执行压力测试、销毁资源。

参考实现来自：

- `deep_ep/utils/refs.py::dispatch`
- `deep_ep/utils/refs.py::combine`
- `deep_ep/utils/refs.py::generate_pre_combine_data`
- `deep_ep/utils/refs.py::ordered_accumulate`

这些 PyTorch 参考实现非常适合用来理解 dispatch/combine 的语义，因为它们更直接、更少底层优化。

## 24. 分布式初始化

测试使用 `deep_ep/utils/envs.py::init_dist` 初始化分布式环境。

主要逻辑：

```text
读取 MASTER_ADDR，默认 127.0.0.1
读取 MASTER_PORT，默认 8361
读取 WORLD_SIZE，表示节点数，默认 1
读取 RANK，表示当前节点编号，默认 0
world_size = num_nodes * num_local_ranks
rank = node_rank * num_local_ranks + local_rank
dist.init_process_group(backend='nccl', init_method='tcp://...')
torch.cuda.set_device(local_rank)
dist.new_group(all ranks)
```

单机测试一般可以直接：

```bash
python tests/elastic/test_ep.py --num-processes 8
```

多机测试需要设置：

```bash
export MASTER_ADDR=<master-ip>
export MASTER_PORT=8361
export WORLD_SIZE=<num_nodes>
export RANK=<node_rank>
python tests/elastic/test_ep.py --num-processes <gpus_per_node>
```

## 25. 环境变量

常用环境变量按用途分组：

通用：

- `EP_BUFFER_DEBUG=1`：打印 buffer 初始化、SM 估算和 backend 调试信息。
- `EP_SUPPRESS_NCCL_CHECK=1`：跳过 NCCL so 一致性检查。
- `EP_AVOID_RECORD_STREAM=1`：避免对输出 tensor 调用 `record_stream`。
- `EP_NUM_TOPK_IDX_BITS=32|64`：指定 top-k index 位宽。

网络：

- `EP_NIC_NAME=mlx5_0`：用于查询 RDMA NIC 属性。
- `EP_OVERRIDE_RDMA_SL=<int>`：覆盖 RDMA service level，用于流量隔离。
- `EP_DISABLE_GIN=1`：禁用 NCCL Gin 路径。

JIT：

- `EP_JIT_DEBUG=1`
- `EP_JIT_CACHE_DIR=<path>`
- `EP_JIT_NVCC_COMPILER=<path-to-nvcc>`
- `EP_JIT_CPP_STANDARD=20`
- `EP_JIT_PRINT_COMPILER_COMMAND=1`
- `EP_JIT_PTXAS_VERBOSE=1`
- `EP_JIT_PTXAS_CHECK=1`
- `EP_JIT_WITH_LINEINFO=1`
- `EP_JIT_DUMP_ASM=1`
- `EP_JIT_DUMP_PTX=1`
- `EP_JIT_DUMP_SASS=1`

构建：

- `EP_NCCL_ROOT_DIR=<path>`
- `EP_NVSHMEM_ROOT_DIR=<path>`
- `TORCH_CUDA_ARCH_LIST=9.0`
- `DISABLE_SM90_FEATURES=1`
- `DISABLE_AGGRESSIVE_PTX_INSTRS=1`

调试建议：

```bash
export EP_BUFFER_DEBUG=1
export EP_JIT_PRINT_COMPILER_COMMAND=1
export EP_JIT_CACHE_DIR=/tmp/deepep_jit_cache
```

## 26. 推荐学习路线

第一阶段：理解 API 语义

1. 读 `README.md` 的 Interfaces and examples。
2. 读 `tests/elastic/test_ep.py` 的 `test_dispatch_combine()`。
3. 读 `deep_ep/utils/refs.py`，理解 PyTorch 参考 dispatch/combine。
4. 自己画出 token -> expert -> rank -> token 的流向。

第二阶段：理解 Python 封装

1. 读 `deep_ep/__init__.py`，理解导入和 JIT 初始化。
2. 读 `deep_ep/buffers/elastic.py::ElasticBuffer.__init__`。
3. 读 `ElasticBuffer.dispatch` 和 `ElasticBuffer.combine`。
4. 读 `EPHandle` 字段含义。
5. 读 `EventOverlap`。

第三阶段：理解 C++ runtime

1. 读 `csrc/python_api.cpp`。
2. 读 `csrc/elastic/buffer.hpp::register_apis`。
3. 读 `csrc/elastic/buffer.hpp` 构造函数。
4. 读 `dispatch(...)` 和 `combine(...)`。
5. 对照 Python 返回值，看 C++ 如何分配 tensor 和 metadata。

第四阶段：理解 JIT 与 kernel

1. 读 `csrc/jit/api.hpp`。
2. 读 `csrc/kernels/elastic/dispatch.hpp` 和 `combine.hpp`。
3. 读 `deep_ep/include/deep_ep/common/layout.cuh`。
4. 读 `deep_ep/include/deep_ep/impls/dispatch.cuh`。
5. 读 `deep_ep/include/deep_ep/impls/combine.cuh`。

第五阶段：理解扩展能力

1. Engram：`tests/elastic/test_engram.py` + `impls/engram_fetch*.cuh`。
2. PP：`tests/elastic/test_pp.py` + `impls/pp_send_recv.cuh`。
3. AGRS：`tests/elastic/test_agrs.py` + `csrc/elastic/buffer.hpp::all_gather`。
4. Legacy：`docs/legacy.md` + `deep_ep/buffers/legacy.py`。

## 27. 一条 dispatch/combine 源码跟踪样例

以 `tests/elastic/test_ep.py` 的普通 BF16 dispatch/combine 为例：

```text
test_loop()
  -> init_dist()
  -> deep_ep.ElasticBuffer(...)
  -> test_dispatch_combine()
  -> buffer.dispatch(...)
```

进入 Python：

```text
deep_ep/buffers/elastic.py::dispatch
  -> get_theoretical_num_sms()
  -> get_theoretical_num_qps()
  -> self.runtime.dispatch(...)
```

进入 C++：

```text
csrc/elastic/buffer.hpp::ElasticBuffer::dispatch
  -> 检查 x/topk_idx/topk_weights shape 和 dtype
  -> 准备 cached metadata 或新 metadata
  -> launch_dispatch(...)
  -> 根据 psum 分配 recv_x/recv_topk_idx/recv_topk_weights
  -> launch dispatch copy epilogue
  -> 返回 metadata 和 event
```

回到 Python：

```text
elastic.py::dispatch
  -> EPHandle(...)
  -> EventOverlap(event)
  -> 返回 recv_x, recv_topk_idx, recv_topk_weights, handle, event
```

combine：

```text
test_dispatch_combine()
  -> buffer.combine(input_for_combine, handle=handle)
  -> elastic.py::combine
  -> self.runtime.combine(...)
  -> csrc/elastic/buffer.hpp::ElasticBuffer::combine
  -> launch_combine(...)
  -> 返回 combined_x, combined_topk_weights, event
```

正确性对比：

```text
deep_ep/utils/refs.py::dispatch
deep_ep/utils/refs.py::combine
torch.equal(...)
```

## 28. 常见概念解释

EP：

专家并行。不同 expert 分布在不同 rank/GPU 上，token 根据 gate 结果跨 rank 发送到 expert。

Dispatch：

把原始 token 根据 top-k expert 路由发送到 expert 所在 rank。

Combine：

把 expert 输出按原 token 归属和 top-k 权重规约回原始 rank。

Scale-up：

通常指节点内或 NVLink 域内扩展。

Scale-out：

通常指跨节点或 RDMA 域扩展。

QP：

RDMA Queue Pair，可以近似理解为 RDMA 通信通道。QP 数影响并行度和调度成本。

SM：

GPU Streaming Multiprocessor。DeepEP 通信 kernel 占用的 SM 越多，纯通信可能越快，但留给模型计算的资源越少。

NCCL Gin：

NCCL 体系内更适合 GPU-initiated networking 的能力，DeepEP V2 用它构建 MoE 细粒度通信。

NVSHMEM/IBGDA：

V1 legacy low-latency 路径使用的通信基础。IBGDA 允许 GPU 侧发起 RDMA 操作。

JIT：

Just-In-Time 编译。V2 kernel 在运行时根据参数和源码 header 编译并缓存。

## 29. 调试与排查建议

NCCL so 不一致：

- 现象：`check_nccl_so()` 断言失败。
- 思路：确认 PyTorch 加载的 `libnccl.so` 和 `EP_NCCL_ROOT_DIR` 指向的是同一套 NCCL。
- 临时绕过：`EP_SUPPRESS_NCCL_CHECK=1`，但不建议长期使用。

JIT 编译失败：

- 打开 `EP_JIT_PRINT_COMPILER_COMMAND=1`。
- 设置 `EP_JIT_CACHE_DIR` 到可写目录。
- 确认 `CUDA_HOME` 或 `CUDA_PATH` 正确。
- 确认 `TORCH_CUDA_ARCH_LIST` 与机器 GPU 匹配。

RDMA 性能异常：

- 检查 `EP_NIC_NAME` 是否对应实际 NIC。
- 检查 `ibstat` 输出是否可用。
- 检查 `EP_OVERRIDE_RDMA_SL` 是否和集群网络策略一致。
- 检查 QP 数是否被 `num_allocated_qps` 限制。

dispatch/combine 正确性问题：

- 先用 `tests/elastic/test_ep.py --skip-perf-test --test-first-only` 缩小范围。
- 关闭 FP8，先验证 BF16。
- 关闭 cached/expanded 复杂路径，先验证普通 dispatch/combine。
- 对比 `deep_ep/utils/refs.py` 的 reference 语义。

异步 stream 问题：

- 如果开启 `async_with_compute_stream=True`，使用结果前必须调用 `event.current_stream_wait()`。
- 如果传 `previous_event`，通常需要配合 `allocate_on_comm_stream=True`。
- PyTorch deterministic algorithms 与 `fill_uninitialized_memory` 同时开启会被 `check_torch_deterministic()` 拒绝。

## 30. 最小 EP 使用模板

下面是一个去掉训练框架细节后的最小结构：

```python
import torch
import torch.distributed as dist
import deep_ep

rank = dist.get_rank()
group = dist.group.WORLD

num_max_tokens_per_rank = 4096
hidden = 7168
num_topk = 8
num_experts = 256

buffer = deep_ep.ElasticBuffer(
    group,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=False,
    explicitly_destroy=True,
)

num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk)

x = torch.randn((num_max_tokens_per_rank, hidden), dtype=torch.bfloat16, device="cuda")
topk_idx = torch.randint(
    0, num_experts,
    (num_max_tokens_per_rank, num_topk),
    dtype=deep_ep.topk_idx_t,
    device="cuda",
)
topk_weights = torch.randn(
    (num_max_tokens_per_rank, num_topk),
    dtype=torch.float,
    device="cuda",
)

recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    expert_alignment=128,
    num_sms=num_sms,
    async_with_compute_stream=True,
)
event.current_stream_wait()

# 这里执行本地 expert 计算，示例中直接复用 recv_x
expert_out = recv_x

combined_x, combined_topk_weights, event = buffer.combine(
    expert_out,
    handle=handle,
    topk_weights=recv_topk_weights,
    num_sms=num_sms,
    async_with_compute_stream=True,
)
event.current_stream_wait()

buffer.destroy()
```

真实模型接入时，`topk_idx/topk_weights` 来自 router，`expert_out` 来自本地 expert GEMM。

## 31. 学习时的重点问题

读完 API 后，建议围绕这些问题检查自己是否真正理解：

1. `dispatch` 为什么需要返回 `handle`？
2. `combine` 为什么不能只依赖 `topk_idx`，还需要 dispatch 产生的 metadata？
3. cached dispatch 省掉了哪些工作？
4. expanded dispatch 解决的是通信问题还是 GEMM layout 问题？
5. `num_sms` 增大为什么可能降低计算 overlap？
6. `num_qps` 为什么不是越多越好？
7. hybrid mode 如何把 scale-out RDMA 和 scale-up NVLink 分层？
8. FP8 dispatch 的 scale factors layout 为什么重要？
9. JIT kernel 和构建时编译的 legacy kernel 分别在哪里？
10. `EventOverlap` 如果不 wait，会出现什么数据依赖问题？

## 32. 总结

DeepEP 的学习主线可以压缩成一句话：先理解 MoE dispatch/combine 的数据语义，再理解 `ElasticBuffer` 如何把语义变成 NCCL Gin + JIT CUDA kernel 的高性能实现。

推荐优先掌握 V2：

```text
README.md
  -> tests/elastic/test_ep.py
  -> deep_ep/buffers/elastic.py
  -> csrc/elastic/buffer.hpp
  -> csrc/kernels/elastic/*.hpp
  -> deep_ep/include/deep_ep/impls/*.cuh
```

legacy V1、Engram、PP、AGRS 都可以在 V2 主线清楚后再按需展开。
