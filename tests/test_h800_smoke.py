import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "8361")
os.environ.setdefault("EP_JIT_PRINT_COMPILER_COMMAND", "1")

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as deep_ep_c


def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(0)

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {cc}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"DeepEP: {deep_ep.__version__}")
    print(f"SM90 compiled: {deep_ep_c.is_sm90_compiled()}")

    assert cc == (9, 0), f"Expected H800/H100 class SM90 GPU, got {cc}"
    assert deep_ep_c.is_sm90_compiled(), "DeepEP was not compiled with SM90 features"

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:8361",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda:0"),
    )
    group = dist.new_group([0])

    num_tokens = 8
    hidden = 256
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

    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
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

    combined_x, combined_topk_weights, _ = buffer.combine(
        recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        num_sms=4,
        num_qps=1,
        async_with_compute_stream=False,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(combined_x, x, rtol=0, atol=0)
    torch.testing.assert_close(combined_topk_weights, topk_weights, rtol=0, atol=0)

    buffer.destroy()
    dist.destroy_process_group()
    print("DeepEP H800 smoke test passed")


if __name__ == "__main__":
    main()
