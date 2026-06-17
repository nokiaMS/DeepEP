# DeepEP 在 NVIDIA H100 上的源码编译与测试说明

本文档基于本地源码目录 `E:\codex_home\code\DeepEP` 和 GitHub 远端地址 `https://github.com/nokiaMS/DeepEP.git`，说明如何在 NVIDIA H100 上从源码编译 DeepEP，并给出编译后的最小测试程序。

## 1. 硬件与源码适配结论

NVIDIA H100 是 DeepEP 当前源码最匹配的目标平台。

- NVIDIA 官方 CUDA GPU 表显示 H100 的 compute capability 是 `9.0`，即 `SM90`：
  <https://developer.nvidia.com/cuda/gpus>
- NVIDIA H100 官方页面说明 H100 基于 Hopper 架构，支持 FP8、第四代 Tensor Core、NVLink 等能力：
  <https://www.nvidia.com/en-us/data-center/h100/>
- DeepEP README 要求 Hopper `SM90`，本地 `setup.py` 默认设置 `TORCH_CUDA_ARCH_LIST=9.0`。
- DeepEP 的安装阶段主要构建 PyTorch CUDA 扩展 `deep_ep._C`，V2 通信 kernel 主要由包内 JIT 在运行期编译。

源码中与 H100 相关的关键点：

- `setup.py` 默认走 SM90 路径：

```python
os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')
nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
```

- 不应设置 `DISABLE_SM90_FEATURES=1`。H100 正是 SM90，应使用默认 FP8/TMA 路径；且当前源码中该分支没有完整实现。

## 2. 基础环境

建议在 Linux H100 机器上编译。Windows 原生环境不适合该项目的 NCCL、NVSHMEM、CUDA 扩展链路。

克隆源码：

```bash
git clone https://github.com/nokiaMS/DeepEP.git
cd DeepEP
git submodule update --init --recursive
```

创建 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging pybind11
```

安装 PyTorch、NCCL、NVSHMEM。CUDA 主版本需要和 PyTorch wheel 匹配。以下示例按 CUDA 12 写：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "nvidia-nccl-cu12>=2.30.4" nvidia-nvshmem-cu12 --no-deps
pip install pynvml
```

如果使用 CUDA 13 / cu13 PyTorch wheel，则 NCCL 包也应换成对应的 `nvidia-nccl-cu13`。

DeepEP 的 `deep_ep/utils/find_pkgs.py` 会优先通过以下环境变量查找依赖：

- `EP_NCCL_ROOT_DIR`
- `NCCL_DIR`
- `EP_NVSHMEM_ROOT_DIR`
- `NVSHMEM_DIR`

如果没有设置这些环境变量，它也会扫描 Python 环境中的 NVIDIA pip wheel。

## 3. 源码编译

在 DeepEP 源码根目录执行：

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# H100 = SM90
export TORCH_CUDA_ARCH_LIST="9.0"

# 可选：固定 JIT cache，便于复用运行期编译结果
export EP_JIT_CACHE_DIR="$HOME/.cache/deepep_jit"

python setup.py build
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

如果 NCCL/NVSHMEM 不是通过 pip wheel 安装，而是系统路径安装：

```bash
export EP_NCCL_ROOT_DIR=/path/to/nccl
export EP_NVSHMEM_ROOT_DIR=/path/to/nvshmem
export LD_LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib:$LD_LIBRARY_PATH"

python setup.py build
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

编译时不要设置：

```bash
export DISABLE_SM90_FEATURES=1
```

原因是 H100 应使用 SM90 特性路径，且当前源码中的 `DISABLE_SM90_FEATURES=1` 分支会直接失败。

## 4. 编译后导入测试

保存为 `test_deepep_import.py`：

```python
import torch
import deep_ep
import deep_ep._C as C

assert torch.cuda.is_available(), "CUDA is not available"

dev = torch.cuda.current_device()
name = torch.cuda.get_device_name(dev)
cc = torch.cuda.get_device_capability(dev)

print("DeepEP version:", deep_ep.__version__)
print("GPU:", name)
print("Compute capability:", cc)
print("Torch CUDA:", torch.version.cuda)
print("SM90 features compiled:", C.is_sm90_compiled())

assert cc == (9, 0), f"Expected H100/SM90, got {cc}"
assert C.is_sm90_compiled(), "DeepEP was not compiled with SM90 features"

print("DeepEP import smoke test passed")
```

运行：

```bash
python test_deepep_import.py
```

预期结果：

- 能正常 `import deep_ep`
- GPU capability 输出为 `(9, 0)`
- `SM90 features compiled` 输出为 `True`
- 最后输出 `DeepEP import smoke test passed`

## 5. 单卡 ElasticBuffer 功能测试

该测试会初始化 `torch.distributed`，创建 `ElasticBuffer`，执行一次 barrier，并在单 rank 下跑一次 dispatch/combine。

保存为 `test_deepep_single_gpu.py`：

```python
import os
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "8361")
os.environ.setdefault("EP_JIT_PRINT_COMPILER_COMMAND", "1")

import torch
import torch.distributed as dist
import deep_ep


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:8361",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda:0"),
    )
    group = dist.new_group([0])

    hidden = 128
    num_tokens = 16
    num_topk = 1
    num_experts = 1

    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
        allow_hybrid_mode=False,
        num_allocated_qps=1,
        explicitly_destroy=True,
    )

    buffer.barrier(with_cpu_sync=True)

    x = torch.randn((num_tokens, hidden), device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.zeros((num_tokens, num_topk), device="cuda", dtype=deep_ep.topk_idx_t)
    topk_weights = torch.ones((num_tokens, num_topk), device="cuda", dtype=torch.float32)

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_tokens,
        expert_alignment=1,
        num_sms=4,
        num_qps=1,
        async_with_compute_stream=False,
    )

    combined_x, combined_topk_weights, event = buffer.combine(
        recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        num_sms=4,
        num_qps=1,
        async_with_compute_stream=False,
    )

    torch.cuda.synchronize()

    assert combined_x.shape == x.shape
    torch.testing.assert_close(combined_x, x, rtol=0, atol=0)

    print("ElasticBuffer barrier + dispatch/combine single-rank test passed")

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

运行：

```bash
python test_deepep_single_gpu.py
```

第一次运行会触发 JIT 编译，耗时会明显长一些。设置 `EP_JIT_CACHE_DIR` 后，后续运行会复用缓存。

## 6. 多卡和仓库测试

在多卡 H100 节点上，可以继续运行仓库自带测试：

```bash
python tests/elastic/test_barrier.py --num-processes 8
python tests/elastic/test_ep.py
```

跨节点测试需要额外配置：

- `MASTER_ADDR`
- `MASTER_PORT`
- `WORLD_SIZE`
- `RANK`
- RDMA/NIC 设备
- NVSHMEM IBGDA 或 GDRCopy
- NCCL 网络环境变量

NVSHMEM 安装和 IBGDA/GDRCopy 配置可参考仓库文档：

```bash
docs/nvshmem.md
```
