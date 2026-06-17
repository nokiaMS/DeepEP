# DeepEP 在 NVIDIA H800 上的源码编译与验证

本文档记录如何在 `gx-test-inst1-0` Pod 上人工检测环境、编译 DeepEP 源码，并运行最小 smoke test 验证编译产物。命令基于 2026-06-17 在 `elm-test/gx-test-inst1-0` 上的实际验证结果整理。

## 1. 连接方式

本地通过 Teleport 连接 Kubernetes master，再通过 `kubectl exec` 进入 Pod：

```bash
tsh status
tsh ssh --user=guoxu root@hd04-cci-k8s-master-1
```

在 master 上确认 Pod：

```bash
kubectl get pod -A -o wide | grep gx-test-inst1-0
kubectl get node hd04-gpul-0042 -o wide
```

实际验证到的目标信息：

```text
namespace: elm-test
pod:       gx-test-inst1-0
node:      hd04-gpul-0042
pod ip:    172.16.84.47
node ip:   172.31.41.42
```

进入 Pod：

```bash
kubectl -n elm-test exec -it gx-test-inst1-0 -- bash
```

如果从本地 Windows 直接执行远程命令，可以使用：

```powershell
tsh ssh --user=guoxu root@hd04-cci-k8s-master-1 "kubectl -n elm-test exec gx-test-inst1-0 -- bash -lc 'hostname; whoami; pwd'"
```

## 2. 环境检测

进入 Pod 后执行：

```bash
hostname
whoami
pwd

nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader

which nvcc || true
nvcc --version || true

which python3 || which python || true
python3 --version || python --version || true
python3 -m pip --version
```

检测 PyTorch：

```bash
python3 - <<'PY'
import sys
import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

检测 Python 依赖：

```bash
python3 -m pip show torch nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-cuda-nvrtc-cu12
```

检测 NCCL/NVSHMEM/CUDA 头文件和库：

```bash
find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 4 \
  \( -name 'libnccl.so*' -o -name 'libnvshmem_host.so*' -o -name 'libnvshmem_device.a' \
     -o -name 'nvrtc.h' -o -name 'cusparse.h' -o -name 'cusolverDn.h' \) | sort

find /usr -name 'libnccl.so*' 2>/dev/null | sort
find /usr -name 'nccl.h' 2>/dev/null | sort
grep -n 'NCCL_MAJOR\|NCCL_MINOR\|NCCL_PATCH' \
  /usr/include/nccl.h \
  /usr/local/lib/python3.12/dist-packages/nvidia/nccl/include/nccl.h 2>/dev/null
```

本次实际检测结果：

```text
GPU:                 8 x NVIDIA H800
compute capability:  9.0
driver:              580.126.09
GPU memory:          81559 MiB each
CUDA toolkit:        12.9.86
Python:              3.12.13
PyTorch:             2.10.0+cu129
PyTorch CUDA:        12.9
pip NCCL:            nvidia-nccl-cu12 2.27.5
system NCCL:         2.29.7
NVSHMEM:             nvidia-nvshmem-cu12 3.4.5
```

结论：

- H800 是 Hopper/SM90 目标，必须按 `TORCH_CUDA_ARCH_LIST=9.0` 编译。
- 当前 DeepEP V2 使用 NCCL Gin backend，README 要求 NCCL `>=2.30.4`。
- Pod 内默认 pip NCCL `2.27.5` 和系统 NCCL `2.29.7` 都不足以编译当前源码。
- 需要额外准备 NCCL `2.30.4+`，本次使用的是 `nvidia-nccl-cu12 2.30.7`。

## 3. 准备源码

建议把源码放在独立目录，避免影响 Pod 内其他文件：

```bash
mkdir -p /userdata/guoxu/deepep-build
cd /userdata/guoxu/deepep-build
```

本文档对应的源码压缩包已放在仓库内：

```text
docs/deepep-h800-source-20260617.tar.gz
```

本地绝对路径：

```text
E:\codex_home\code\DeepEP\docs\deepep-h800-source-20260617.tar.gz
```

方式一：Pod 内能访问 GitHub 时直接 clone：

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
git submodule update --init --recursive
```

方式二：从本地仓库打包后复制到 Pod：

```powershell
# 使用本文档对应的源码压缩包
tsh scp --user=guoxu .\docs\deepep-h800-source-20260617.tar.gz root@hd04-cci-k8s-master-1:/tmp/deepep-transfer/deepep-src.tar.gz
```

在 master 上复制进 Pod：

```bash
kubectl -n elm-test cp /tmp/deepep-transfer/deepep-src.tar.gz gx-test-inst1-0:/userdata/guoxu/deepep-build/deepep-src.tar.gz
kubectl -n elm-test exec -it gx-test-inst1-0 -- bash
cd /userdata/guoxu/deepep-build
tar -xzf deepep-src.tar.gz
cd DeepEP
```

## 4. 安装隔离 NCCL 2.30.7

不要直接覆盖 Pod 全局 PyTorch 依赖。建议安装到独立目录：

```bash
mkdir -p /userdata/guoxu/deepep-build/deps
python3 -m pip install --target /userdata/guoxu/deepep-build/deps 'nvidia-nccl-cu12>=2.30.4' --no-deps
```

确认版本：

```bash
PYTHONPATH=/userdata/guoxu/deepep-build/deps python3 -m pip show nvidia-nccl-cu12
```

本次验证版本：

```text
nvidia-nccl-cu12 2.30.7
```

## 5. 编译环境变量

在 Pod 的 DeepEP 源码根目录执行：

```bash
cd /userdata/guoxu/deepep-build/DeepEP

export PYTHONPATH=/userdata/guoxu/deepep-build/deps

export LD_LIBRARY_PATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:\
/usr/local/cuda/lib64:\
/usr/local/nvidia/lib64

export CPATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cublas/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/curand/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cufft/include

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0

export EP_JIT_CACHE_DIR=/userdata/guoxu/deepep-build/jit-cache
export EP_JIT_PRINT_COMPILER_COMMAND=1
```

说明：

- `PYTHONPATH` 让 `setup.py` 优先发现隔离安装的 NCCL 2.30.7。
- `LD_LIBRARY_PATH` 让链接和运行时优先使用 NCCL 2.30.7，并找到 NVSHMEM/CUDA 库。
- `CPATH` 补齐 PyTorch CUDA pip wheel 中的头文件路径，例如 `nvrtc.h`、`cusparse.h`、`cusolverDn.h`。
- 不要设置 `DISABLE_SM90_FEATURES=1`。H800 是 SM90，应使用默认 SM90/FP8/TMA 路径。

## 6. 编译和打包

```bash
cd /userdata/guoxu/deepep-build/DeepEP
rm -rf build dist *.egg-info

python3 setup.py build
python3 setup.py bdist_wheel
```

成功后应生成类似产物：

```bash
ls -lh dist
```

本次验证产物：

```text
dist/deep_ep-2.0.0+local-cp312-cp312-linux_x86_64.whl
```

## 7. 编译后 smoke test

仓库内提供了一个最小 H800 smoke test：

```bash
tests/test_h800_smoke.py
```

该测试覆盖：

- CUDA 可用性
- GPU compute capability 为 `(9, 0)`
- `deep_ep._C.is_sm90_compiled()` 为 `True`
- 单 rank NCCL process group
- `ElasticBuffer.barrier`
- 单 rank dispatch/combine 闭环
- JIT 编译 `sm_90a` kernel

不安装 wheel，直接使用 build 输出运行：

```bash
cd /userdata/guoxu/deepep-build/DeepEP

export PYTHONPATH=/userdata/guoxu/deepep-build/DeepEP/build/lib.linux-x86_64-cpython-312:\
/userdata/guoxu/deepep-build/deps

export LD_LIBRARY_PATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:\
/usr/local/cuda/lib64:\
/usr/local/nvidia/lib64

export CPATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cublas/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/curand/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cufft/include

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export EP_JIT_CACHE_DIR=/userdata/guoxu/deepep-build/jit-cache
export EP_JIT_PRINT_COMPILER_COMMAND=1

python3 tests/test_h800_smoke.py
```

本次验证通过输出：

```text
GPU: NVIDIA H800
Compute capability: (9, 0)
PyTorch: 2.10.0+cu129, CUDA: 12.9
DeepEP: 2.0.0
SM90 compiled: True
DeepEP H800 smoke test passed
```

首次运行会打印 JIT 的 `nvcc` 命令，并生成 `sm_90a` cubin。后续运行会复用 `EP_JIT_CACHE_DIR`。

## 8. 可选：安装 wheel 后测试

如果需要安装到当前 Python 环境：

```bash
cd /userdata/guoxu/deepep-build/DeepEP
python3 -m pip install --force-reinstall dist/deep_ep-2.0.0+local-cp312-cp312-linux_x86_64.whl
```

安装后运行时仍应保证 NCCL 2.30.7 优先：

```bash
export PYTHONPATH=/userdata/guoxu/deepep-build/deps
export LD_LIBRARY_PATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:\
/usr/local/cuda/lib64:\
/usr/local/nvidia/lib64

python3 tests/test_h800_smoke.py
```

## 9. 常见失败及处理

### 9.1 `NCCL_API_MAGIC` 未定义或 `ginTrafficClass` 缺失

现象：

```text
identifier "NCCL_API_MAGIC" is undefined
class "ncclDevCommRequirements" has no member "ginTrafficClass"
```

原因：

- 构建使用了旧版 pip NCCL `2.27.5`。
- 或者 `/usr/include/nccl_device` 与旧版 `nvidia/nccl/include/nccl.h` 混用。

处理：

```bash
python3 -m pip install --target /userdata/guoxu/deepep-build/deps 'nvidia-nccl-cu12>=2.30.4' --no-deps
export PYTHONPATH=/userdata/guoxu/deepep-build/deps
export LD_LIBRARY_PATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/lib:$LD_LIBRARY_PATH
export CPATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/include:$CPATH
```

### 9.2 `nvrtc.h` 缺失

现象：

```text
fatal error: nvrtc.h: No such file or directory
```

处理：

```bash
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/include:$CPATH
```

### 9.3 `cusparse.h` 或 `cusolverDn.h` 缺失

现象：

```text
fatal error: cusparse.h: No such file or directory
fatal error: cusolverDn.h: No such file or directory
```

处理：

```bash
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/include:\
/usr/local/lib/python3.12/dist-packages/nvidia/cublas/include:\
$CPATH
```

### 9.4 smoke test 中 `Invalid hidden`

现象：

```text
static assertion failed with "Invalid hidden"
```

原因：

- DeepEP combine JIT kernel 对 BF16 hidden 有对齐要求。
- 最初使用 `hidden=128` 会失败。

处理：

- 使用 `hidden=256` 或更贴近真实模型的 `7168`。
- 当前 `tests/test_h800_smoke.py` 已使用 `hidden=256`。

## 10. 最终可复制命令摘要

在 Pod 内从已存在源码目录编译并测试：

```bash
cd /userdata/guoxu/deepep-build/DeepEP

mkdir -p /userdata/guoxu/deepep-build/deps
python3 -m pip install --target /userdata/guoxu/deepep-build/deps 'nvidia-nccl-cu12>=2.30.4' --no-deps

export PYTHONPATH=/userdata/guoxu/deepep-build/deps
export LD_LIBRARY_PATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
export CPATH=/userdata/guoxu/deepep-build/deps/nvidia/nccl/include:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/include:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/include:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/include:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/include:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/include:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include:/usr/local/lib/python3.12/dist-packages/nvidia/curand/include:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/include
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export EP_JIT_CACHE_DIR=/userdata/guoxu/deepep-build/jit-cache
export EP_JIT_PRINT_COMPILER_COMMAND=1

rm -rf build dist *.egg-info
python3 setup.py build
python3 setup.py bdist_wheel

export PYTHONPATH=/userdata/guoxu/deepep-build/DeepEP/build/lib.linux-x86_64-cpython-312:/userdata/guoxu/deepep-build/deps
python3 tests/test_h800_smoke.py
```
