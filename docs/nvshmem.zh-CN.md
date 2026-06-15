# 安装 NVSHMEM

## 重要说明

**本项目既不是由 NVIDIA 赞助，也不受 NVIDIA 支持。**

**NVIDIA NVSHMEM 的使用受 [NVSHMEM Software License Agreement](https://docs.nvidia.com/nvshmem/api/sla.html) 条款约束。**

## 前置条件

硬件要求：
   - 单节点内的 GPU 需要通过 NVLink 连接
   - 跨节点 GPU 需要通过 RDMA 设备连接，见 [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
   - 支持 InfiniBand GPUDirect Async（IBGDA），见 [IBGDA Overview](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
   - 更详细的要求见 [NVSHMEM Hardware Specifications](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)

软件要求：
   - NVSHMEM v3.3.9 或更高版本

## 安装流程

### 1. 安装 NVSHMEM 二进制包

NVSHMEM 3.3.9 二进制包提供多种格式：
   - 面向 [x86_64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz) 和 [aarch64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-sbsa/libnvshmem-linux-sbsa-3.3.9_cuda12-archive.tar.xz) 的 tarball
   - RPM 和 deb 包：说明见 [NVSHMEM installer page](https://developer.nvidia.com/nvshmem-downloads?target_os=Linux)
   - 通过 conda-forge 提供的 Conda 包
   - 通过 PyPI 提供的 pip wheel：`pip install nvidia-nvshmem-cu12`

DeepEP 兼容 upstream NVSHMEM 3.3.9 及更高版本。

### 2. 启用 NVSHMEM IBGDA 支持

NVSHMEM 支持两种模式，它们有不同的要求。可以使用以下任意一种方式启用 IBGDA 支持。

#### 2.1 配置 NVIDIA 驱动

该配置会启用传统 IBGDA 支持。

修改 `/etc/modprobe.d/nvidia.conf`：

```bash
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

更新 kernel 配置：

```bash
sudo update-initramfs -u
sudo reboot
```

#### 2.2 安装 GDRCopy 并加载 gdrdrv kernel module

该配置通过 CPU 辅助的异步 post-send 操作启用 IBGDA。关于 CPU-assisted IBGDA 的更多信息见[这篇博客](https://developer.nvidia.com/blog/enhancing-application-portability-and-compatibility-across-new-platforms-using-nvidia-magnum-io-nvshmem-3-0/#cpu-assisted_infiniband_gpu_direct_async%C2%A0)。

这种方式会带来少量性能损失，但在无法修改驱动 regkeys 时可以使用。

下载 GDRCopy。GDRCopy 提供预构建的 deb 和 rpm 包，见[这里](https://developer.download.nvidia.com/compute/redist/gdrcopy/)，也可以从 [GDRCopy GitHub 仓库](https://github.com/NVIDIA/gdrcopy)获取源码。

按照 [GDRCopy GitHub 仓库](https://github.com/NVIDIA/gdrcopy?tab=readme-ov-file#build-and-installation)中的说明安装 GDRCopy。

## 安装后配置

如果不是通过 RPM 或 deb 包安装 NVSHMEM，需要在 shell 配置中设置以下环境变量：

```bash
export NVSHMEM_DIR=/path/to/your/dir/to/install  # DeepEP 安装时使用
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
```

## 验证

```bash
nvshmem-info -a # 应显示 nvshmem 的详细信息
```
