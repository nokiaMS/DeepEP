# 中文文件注释：
# 本文件用于测试 DeepEP V2 `ElasticBuffer` 的专家并行（EP）dispatch/combine 功能。
# 覆盖内容包括：不同 EP 模式枚举、FP8/BF16 dispatch、expanded/cached dispatch、
# combine/reduced combine、正确性校验、性能 profiling、deterministic 模式和压力测试。
# 测试依赖多进程分布式环境，每个进程通常对应一个 GPU/rank。
import argparse
import os
import torch
import torch.distributed as dist
from typing import Union, Tuple, Optional

import deep_ep
from deep_ep.utils.math import (
    align, count_bytes, calc_diff,
    per_token_cast_back, per_token_cast_to_fp8,
    safe_div
)
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.envs import init_dist, init_seed, dist_print
from deep_ep.utils.refs import dispatch as ref_dispatch
from deep_ep.utils.refs import combine as ref_combine
from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate
from deep_ep.utils.testing import bench_kineto


# noinspection PyUnusedLocal,PyShadowingNames
# 中文函数注释：枚举 EP dispatch/combine 的测试组合，包括 handle copy、expert 对齐、FP8、bias、事件同步和 stream 分配模式。
def enumerate_ep_modes():
    for do_handle_copy in (1, 0):
        for expert_alignment in (128, 1):
            for use_fp8_dispatch in (1, 0):
                for num_bias in (0, 1, 2):
                    for with_previous_event in (0, 1):
                        for async_with_compute_stream in (0, 1):
                            for allocate_on_comm_stream in ((1, ) if with_previous_event else (0, 1)):
                                yield (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
                                       with_previous_event, async_with_compute_stream, allocate_on_comm_stream)


# 中文函数注释：统一调用 `ElasticBuffer` 上的 dispatch/combine 等接口，并按需注入 previous_event 和等待异步事件完成。
def launch(buffer: deep_ep.ElasticBuffer, name: str,
           with_previous_event: int, async_with_compute_stream: int,
           params: dict):
    if with_previous_event:
        params.update(previous_event=buffer.capture())
    values = getattr(buffer, name)(**params)
    values[-1].current_stream_wait() if async_with_compute_stream else ()
    return values


# 中文函数注释：将 expanded dispatch 输出按 metadata 中的索引折叠回普通 token 布局，并校验同一 token 的展开副本一致。
def fold_expanded(expanded: Union[Tuple[torch.Tensor], torch.Tensor],
                  indices: torch.Tensor, valid_mask: torch.Tensor):
    if not isinstance(expanded, torch.Tensor):
        return tuple(fold_expanded(t, indices, valid_mask) for t in expanded)

    gathered = expanded[indices]
    first_valid_idx = valid_mask.to(torch.int).argmax(dim=1)
    folded = gathered[torch.arange(gathered.shape[0], device='cuda'), first_valid_idx]
    result = (gathered == folded.unsqueeze(1)).all(dim=-1)
    result = result | (~valid_mask)
    assert result.all()
    return folded


# noinspection PyUnboundLocalVariable,PyShadowingNames
# 中文函数注释：执行 Elastic EP dispatch/combine 的主测试流程，包含测试数据构造、参考实现对比、性能测试和正确性断言。
# 中文函数注释：测试 ElasticBuffer 的 EP dispatch/combine 全流程，覆盖模式枚举、正确性校验、性能 profiling 和边界配置。
def test_dispatch_combine(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):
    """
    中文函数说明：
        该函数是 tests/elastic/test_ep.py 的核心测试函数。
        它会基于输入的 ElasticBuffer 和命令行参数生成 MoE 路由数据，
        遍历 dispatch/combine 的多种配置组合，并验证 DeepEP 输出与 PyTorch 参考实现一致。

    中文参数说明：
        buffer：DeepEP V2 的 ElasticBuffer 实例，负责执行 dispatch、combine、barrier、capture 等通信操作。
        args：命令行参数集合，包含 token 数、hidden 维度、expert 数、SM/QP 设置、校验和 profiling 开关等。
    """
    # Settings
    # 读取当前 buffer 推导出的逻辑通信域，并从命令行参数中整理本轮测试的 MoE 规模。
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
    num_max_tokens_per_rank, num_tokens, hidden = args.num_tokens, max(1, args.num_tokens - dist.get_rank()), args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_local_experts = num_experts // buffer.num_ranks
    # 如果命令行没有显式指定 SM/QP 数量，则使用 ElasticBuffer 的解析式估算值。
    num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk) if args.num_sms == 0 else args.num_sms
    num_qps = buffer.get_theoretical_num_qps(num_sms) if args.num_qps == 0 else args.num_qps
    # 只在每个节点打印一次本轮测试配置，避免多 rank 输出互相刷屏。
    dist_print(f'Config:\n'
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n'
               f' > Experts: {num_topk}/{num_experts}\n'
               f' > Tokens: {num_tokens} (max: {num_max_tokens_per_rank}), hidden: {hidden}\n'
               f' > #SM: {num_sms}, #QPs: {num_qps}/{buffer.num_allocated_qps}\n',
               once_in_node=True)

    # Construct expert selections first (may have an unbalanced ratio here)
    # 构造 router/gate 分数，并取 top-k expert 作为每个 token 的路由目标。
    scores = get_unbalanced_scores(num_tokens, num_experts, buffer.num_ranks, num_topk, args.unbalanced_ratio, args.precise_unbalanced_ratio)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    # masked_ratio 用来模拟部分 token/expert 选择无效的情况，-1 表示该路由槽位被屏蔽。
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # Run all tests
    dist_print('Running all test cases:', once_in_node=True)
    # 遍历 dispatch/combine 的不同模式组合，覆盖 FP8、expanded、cached、事件同步等路径。
    for (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
         with_previous_event, async_with_compute_stream, allocate_on_comm_stream) in enumerate_ep_modes():
        dist_print(f' > Testing with '
                   f'{do_handle_copy=}, {expert_alignment=}, {use_fp8_dispatch=}, {num_bias=}, '
                   f'{with_previous_event=}, {async_with_compute_stream=}, {allocate_on_comm_stream=} ...',
                   once_in_node=True)

        # Random data
        # TODO: support top-k groups
        # 构造本 rank 的输入 hidden states；开启 FP8 dispatch 时转换为 data + scale_factors 形式。
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        x = per_token_cast_to_fp8(x) if use_fp8_dispatch else x
        # num_bias 用来覆盖 combine 阶段无 bias、单 bias、多个 bias 的测试路径。
        bias = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') if num_bias == 1 else None
        if num_bias == 2:
            bias = tuple(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') for _ in range(num_bias))
            assert len(bias) == 2   # To prevent linter warning

        # Test correctness with NCCL reference
        if not args.skip_check:
            # 使用 PyTorch/NCCL 参考实现生成 dispatch 期望结果，用于后续和 DeepEP 输出做 bitwise 对比。
            ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, \
                ref_recv_src_token_idx, ref_num_recv_tokens_per_rank = \
                ref_dispatch(x, topk_idx, topk_weights, num_max_tokens_per_rank, num_experts)
            ref_recv_x_bf16 = per_token_cast_back(ref_recv_x[0], ref_recv_x[1]) if use_fp8_dispatch else ref_recv_x

            # 根据 multiple reduction / hybrid mode 选择参考 combine 的规约策略。
            if args.allow_multiple_reduction:
                # Should be the same as the trigger condition of DeepEP's hybrid combine, which performs intra-scaleup reduction first
                if args.allow_hybrid_mode and num_scaleout_ranks > 1:
                    reduced_combine_recipe = (True, True)
                    combine_recipe = (True, True)
                else:
                    reduced_combine_recipe = (True, False)
                    combine_recipe = (True, False)
            else:
                reduced_combine_recipe = (False, False)
                combine_recipe = (True, False)
            # 构造 combine 输入的参考数据，并分别生成普通 combine 与 reduced combine 的期望结果。
            ref_y = generate_pre_combine_data(
                dist.get_rank() * num_max_tokens_per_rank + torch.arange(num_tokens, device='cuda'),
                num_max_tokens_per_rank, num_topk, hidden)
            ref_y[topk_idx == -1] = 0
            ref_reduced_combined_y = ref_combine(
                ref_y, topk_idx,
                num_scaleout_ranks, num_scaleup_ranks, num_experts,
                bias,
                *reduced_combine_recipe
            )
            ref_combined_y = ref_combine(
                ref_y, topk_idx,
                num_scaleout_ranks, num_scaleup_ranks,
                num_experts, bias,
                *combine_recipe
            )  # Reduce within rank, then globally, for non-expand combine mode
            torch.cuda.synchronize()

        # Do dispatch
        # 组装普通 dispatch 参数，并调用 ElasticBuffer.dispatch 执行 token 到 expert/rank 的重排。
        dispatch_args = dict(
            x=x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_sms=num_sms, num_qps=num_qps,
            num_max_tokens_per_rank=num_max_tokens_per_rank, num_experts=num_experts,
            expert_alignment=expert_alignment,
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_handle_copy=do_handle_copy, do_cpu_sync=args.do_cpu_sync)
        recv_x, recv_topk_idx, recv_topk_weights, handle, dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)
        # 为了统一后续校验逻辑，FP8 dispatch 输出会还原为 BF16 版本。
        recv_x_bf16 = per_token_cast_back(recv_x[0], recv_x[1]) if use_fp8_dispatch else recv_x

        # Expanding mode
        # expanded dispatch 会把 token 按 expert 展开布局，便于后续按 expert/GEMM 组织输入。
        expanded_dispatch_args = dispatch_args | dict(do_expand=True, use_tma_aligned_col_major_sf=True)
        expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, expanded_handle, expanded_dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, expanded_dispatch_args)
        expanded_recv_x_bf16 = per_token_cast_back(expanded_recv_x[0], expanded_recv_x[1]) if use_fp8_dispatch else expanded_recv_x

        # Cached mode
        # cached dispatch 复用第一次 dispatch 产生的 handle，验证跳过 layout 重算后的结果一致性。
        cached_dispatch_args = dict(
            x=x,
            num_sms=num_sms, num_qps=num_qps,
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
            handle=handle)
        cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle, cached_dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, cached_dispatch_args)

        # Count the number of received tokens
        # 从 handle 的 prefix-sum metadata 中取出本 rank 实际收到的 token 数。
        num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        assert num_recv_tokens == expanded_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item(), \
               'Expand should not affect the number of received tokens.'
        num_expanded_tokens = expanded_handle.psum_num_recv_tokens_per_expert[-1].item()

        # Construction the input data for DeepEP combine
        # 根据 dispatch 返回的源 token metadata 构造普通 combine 的本地输入。
        src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
        if not args.skip_check:
            sorted_src_token_global_idx = torch.sort(src_token_global_idx).values
            assert torch.equal(ref_recv_src_token_idx, sorted_src_token_global_idx), \
                f'{ref_recv_src_token_idx=}, {sorted_src_token_global_idx=}'
        local_y = generate_pre_combine_data(src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)  # [num_recv_tokens, topk, hidden]
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
        local_reduced_y = ordered_accumulate(local_y)
        input_for_combine = torch.empty_like(recv_x_bf16, dtype=torch.bfloat16, device='cuda')
        input_for_combine[:num_recv_tokens] = local_reduced_y

        # expanded combine 使用 expanded metadata 中的展开索引，把每个 top-k expert 输出放回对应槽位。
        expanded_src_token_global_idx = expanded_handle.recv_src_metadata[:num_recv_tokens, 0]
        if not args.skip_check:
            sorted_expanded_src_token_global_idx = torch.sort(expanded_src_token_global_idx).values
            assert torch.equal(ref_recv_src_token_idx, sorted_expanded_src_token_global_idx), \
                f'{ref_recv_src_token_idx=}, {sorted_expanded_src_token_global_idx=}'
        local_y_expand = generate_pre_combine_data(expanded_src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)  # [num_recv_tokens, topk, hidden]
        # We put an extra row to conveniently handle the -1 index
        input_for_expand_combine = torch.empty((expanded_recv_x_bf16.shape[0] + 1, hidden), dtype=torch.bfloat16, device='cuda')
        input_for_expand_combine[expanded_handle.recv_src_metadata[:num_recv_tokens, 2:].flatten()] = local_y_expand.view(-1, hidden)
        input_for_expand_combine = input_for_expand_combine[:-1, ...]

        # Do combine
        # 普通 combine 将 expert 输出按原 token 顺序规约回原始 rank。
        combine_args = dict(
            x=input_for_combine, topk_weights=recv_topk_weights, bias=bias,
            handle=handle,
            num_sms=num_sms, num_qps=num_qps,
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        combined_x, combined_topk_weights, combine_event = \
            launch(buffer, 'combine', with_previous_event, async_with_compute_stream, combine_args)

        # Reduced combine
        # reduced combine 使用 expanded dispatch 的布局，测试展开布局下的 combine/reduction 路径。
        reduced_combine_args = dict(
            x=input_for_expand_combine, bias=bias,
            handle=expanded_handle,
            num_sms=num_sms, num_qps=num_qps,
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        reduced_combined_x, reduced_combined_topk_weights, reduced_combine_event = \
            launch(buffer, 'combine', with_previous_event, async_with_compute_stream, reduced_combine_args)

        assert not (args.dump_profile_traces and args.skip_perf_test), '`--skip-perf-test` should not be specified when `--dump-profile-traces` is provided'
        if not args.skip_perf_test:
            # Profiling
            # 中文函数注释：根据 profiling 前缀和当前 rank 生成 Kineto trace 文件路径；未开启 dump 时返回 None。
            def get_trace_path(prefix: str):
                return None if not args.dump_profile_traces else f'{args.dump_profile_traces}/{prefix}_rank{buffer.rank_idx}.json'

            # Calculate the number of tokens that are sent to the other scaleout peers
            # 统计 dispatch 中需要跨 scale-out rank 发送的 token 数，用于计算逻辑带宽。
            dst_scaleout_rank_idx = topk_idx // (num_experts // num_scaleout_ranks)
            num_scaleout_send_tokens = 0
            for i in range(num_scaleout_ranks if num_scaleout_ranks > 1 else 0):
                if args.ignore_local_traffic and i == dist.get_rank() // num_scaleup_ranks:
                    continue
                num_scaleout_send_tokens += (dst_scaleout_rank_idx == i).any(dim=1).sum().item()

            # Calculate the number of tokens that are received via the other scaleup peers
            # 统计 dispatch 中经 scale-up peer 接收的 token 数，可按参数忽略本地流量。
            num_scaleup_recv_tokens = num_recv_tokens
            if args.ignore_local_traffic:
                num_scaleup_recv_tokens -= (src_token_global_idx // num_max_tokens_per_rank % num_scaleup_ranks == dist.get_rank() % num_scaleup_ranks).sum().item()

            # Test dispatch performance
            # 对普通 dispatch 做 Kineto benchmark，并根据逻辑流量换算 SO/SU 带宽。
            num_bytes_per_dispatch_token = safe_div(count_bytes(recv_x, recv_topk_idx, recv_topk_weights), recv_topk_idx.size(0))
            num_scaleup_bytes = num_bytes_per_dispatch_token * num_scaleup_recv_tokens  # Received via scaleup
            num_scaleout_bytes = num_bytes_per_dispatch_token * num_scaleout_send_tokens    # Send via scaleout
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**dispatch_args),
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('dispatch'))
            dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'dispatch: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {2 * num_recv_tokens * num_bytes_per_dispatch_token / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test expanded dispatch performance
            # 对 expanded dispatch 做 benchmark，额外计入 expanded metadata 和展开 token 拷贝量。
            num_bytes_per_dispatch_token_meta = safe_div(count_bytes(expanded_handle.recv_src_metadata), expanded_handle.recv_src_metadata.size(0))
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**expanded_dispatch_args),
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('expanded_dispatch'))
            dist_print(f'   - EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'expanded dispatch: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {(num_recv_tokens * (num_bytes_per_dispatch_token_meta + num_bytes_per_dispatch_token) + num_expanded_tokens * num_bytes_per_dispatch_token) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test cached dispatch performance
            # 对 cached dispatch 做 benchmark，验证复用 handle 后的通信与 copy 性能。
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**cached_dispatch_args),
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('cached_dispatch'))
            dist_print(f'   # EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'cached dispatch: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {2 * num_scaleup_bytes / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test combine performance
            # 预先计算 combine 单 token 字节量、bias 字节量和 reduction 写出字节量。
            num_bytes_per_combine_token = safe_div(count_bytes(recv_x_bf16, recv_topk_weights), recv_x_bf16.size(0))
            num_bias_bytes = count_bytes(bias)
            num_reduction_write_bytes = count_bytes(combined_x, combined_topk_weights)

            # 中文函数注释：估算 combine 阶段 scale-out、scale-up 和 reduce 读流量，用于计算逻辑带宽。
            def get_combine_bytes(is_expand_mode: bool) -> Tuple[float, float, float]:
                # 根据 EP 拆分关系计算每个 rank/scaleout rank 覆盖的 expert 数量。
                num_experts_per_rank = num_experts // (num_scaleup_ranks * num_scaleout_ranks)
                num_experts_per_scaleout_rank = num_experts_per_rank * num_scaleup_ranks

                # 中文函数注释：统计目标 index 中有效且去重后的目的地数量，并可按范围忽略本地流量。
                def get_unique_and_valid_dst_count(dst_idx: torch.Tensor,
                                                   ignored_nums_l: Optional[int] = None, ignored_nums_r: Optional[int] = None,
                                                   max_num_in_dst_idx: int = num_experts - 1) -> int:
                    """
                    Get the number of valid destinations, with deduplication within each token and numbers within `[ignored_nums_l, ignored_nums_r)` being ignored
                    """
                    # clone 后在临时张量上处理无效目的地和本地流量屏蔽，不修改原始 topk_idx。
                    dst_idx = dst_idx.clone()
                    ignore_mask = dst_idx == -1
                    if args.ignore_local_traffic and ignored_nums_l is not None:
                        assert ignored_nums_r is not None
                        ignore_mask |= ((dst_idx >= ignored_nums_l) & (dst_idx < ignored_nums_r))
                    dst_idx = dst_idx + torch.arange(0, dst_idx.shape[0], dtype=dst_idx.dtype, device=dst_idx.device).unsqueeze(-1) * (max_num_in_dst_idx + 1)  # So that different rows will have different values
                    dst_idx[ignore_mask] = dst_idx[0][0].item()  # So that these `-1`s won't affect the count of unique numbers
                    return torch.unique(dst_idx, sorted=False).numel()

                # 不允许 multiple reduction 时，普通布局和 expanded 布局的通信 token 统计方式不同。
                if not args.allow_multiple_reduction:
                    # No multiple reduction
                    if not is_expand_mode:
                        num_scaleup_tokens = num_scaleup_recv_tokens
                        num_scaleout_tokens = get_unique_and_valid_dst_count(
                            topk_idx // num_experts_per_rank, buffer.scaleout_rank_idx * num_scaleup_ranks, (buffer.scaleout_rank_idx + 1) * num_scaleup_ranks)
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_rank)
                    else:
                        tokens_src_rank_idx = src_token_global_idx//num_max_tokens_per_rank
                        if args.ignore_local_traffic:
                            num_scaleup_tokens = (recv_topk_idx[:num_recv_tokens] != -1)[tokens_src_rank_idx % num_scaleup_ranks != buffer.scaleup_rank_idx].sum().item()
                        else:
                            num_scaleup_tokens = (recv_topk_idx[:num_recv_tokens] != -1).sum().item()
                        num_scaleout_tokens = get_unique_and_valid_dst_count(
                            topk_idx, buffer.scaleout_rank_idx * num_experts_per_scaleout_rank, (buffer.scaleout_rank_idx + 1) * num_experts_per_scaleout_rank)
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx)
                else:
                    # With `allow_multiple_reduction`, "combine" has exactly the same number of tokens as "dispatch"
                    # multiple reduction 模式下，combine 的通信规模与 dispatch 对齐。
                    num_scaleup_tokens = num_scaleup_recv_tokens
                    num_scaleout_tokens = num_scaleout_send_tokens
                    if args.allow_hybrid_mode:
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_scaleout_rank)
                    else:
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_rank)
                if not args.ignore_local_traffic and num_scaleout_ranks == 1:
                    num_scaleout_tokens = 0
                return num_scaleout_tokens * num_bytes_per_combine_token, num_scaleup_tokens * num_bytes_per_combine_token, num_reduction_read_tokens * num_bytes_per_combine_token

            # 对普通 combine 做 benchmark，并拆分通信耗时和 reduce epilogue 耗时。
            num_scaleout_bytes, num_scaleup_bytes, num_reduction_read_bytes = get_combine_bytes(False)
            t, copy_t = bench_kineto(lambda: buffer.combine(**combine_args),
                                    kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('combine'))
            dist_print(f'   @ EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'combine: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'reduce: {(num_bias_bytes + num_reduction_read_bytes + num_reduction_write_bytes) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test reduced combine performance
            # 对 reduced combine 做 benchmark，使用 expanded 布局对应的逻辑流量估算。
            num_scaleout_bytes, num_scaleup_bytes, num_reduction_read_bytes = get_combine_bytes(True)
            t, copy_t = bench_kineto(lambda: buffer.combine(**reduced_combine_args),
                                    kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('reduced_combine'))
            dist_print(f'   + EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'reduced combine: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'reduce: {(num_bias_bytes + num_reduction_read_bytes + num_reduction_write_bytes) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')
            dist_print(once_in_node=True)

        # Checks
        # NOTES: we do checks after the performance tests, as we may modify some tensors
        if not args.skip_check:
            # Handle copy checks
            # 验证 handle 是否按 do_handle_copy 设置复制 topk_idx，并确认 cached handle 复用同一份 metadata。
            assert (topk_idx.data_ptr() != handle.topk_idx.data_ptr()) == do_handle_copy
            assert (topk_idx.data_ptr() != cached_handle.topk_idx.data_ptr()) == do_handle_copy
            assert handle.topk_idx.data_ptr() == cached_handle.topk_idx.data_ptr()

            # Make the valid part of the whole tensor for no CPU sync mode
            # 关闭 CPU sync 时输出 tensor 可能按最大容量分配，这里裁剪到真实有效 token 区间再校验。
            if not args.do_cpu_sync:
                if use_fp8_dispatch:
                    recv_x = (recv_x[0][:num_recv_tokens], recv_x[1][:num_recv_tokens])
                    cached_recv_x = (cached_recv_x[0][:num_recv_tokens], cached_recv_x[1][:num_recv_tokens])
                else:
                    recv_x = recv_x[:num_recv_tokens]
                    cached_recv_x = cached_recv_x[:num_recv_tokens]
                recv_x_bf16 = recv_x_bf16[:num_recv_tokens]
                recv_topk_idx = recv_topk_idx[:num_recv_tokens]
                recv_topk_weights = recv_topk_weights[:num_recv_tokens]
                cached_recv_topk_idx = cached_recv_topk_idx[:num_recv_tokens]
                handle.recv_src_metadata = handle.recv_src_metadata[:num_recv_tokens]
                expanded_handle.recv_src_metadata = expanded_handle.recv_src_metadata[:num_recv_tokens]

            # Make sure deterministic mode works by doing the dispatch twice
            # deterministic 模式下重复 dispatch，检查源 token metadata 是否稳定一致。
            if args.deterministic:
                recv_x_twice, recv_topk_idx_twice, recv_topk_weights_twice, handle_twice, dispatch_event_twice = \
                    launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)
                if not args.do_cpu_sync:
                    assert num_recv_tokens == handle_twice.psum_num_recv_tokens_per_scaleup_rank[-1].item()
                    handle_twice.recv_src_metadata = handle_twice.recv_src_metadata[:num_recv_tokens]
                assert torch.equal(handle.recv_src_metadata[:, :2], handle_twice.recv_src_metadata[:, :2])

            # Test cumulative stats counter
            # 验证 dispatch 可累计每个 local expert 接收 token 的统计计数。
            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
            dispatch_args['cumulative_local_expert_recv_stats'] = cumulative_local_expert_recv_stats
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)

            # Expanded checks
            # 检查 expanded dispatch 的 metadata 形状，并折叠 expanded 输出用于和普通参考结果比较。
            assert expanded_recv_topk_idx is None
            assert expanded_handle.recv_src_metadata.size(0) == num_recv_tokens
            expanded_indices = expanded_handle.recv_src_metadata[:, 2:]
            expanded_mask = expanded_indices >= 0
            expanded_safe_indices = expanded_indices.clone()
            expanded_safe_indices[~expanded_mask] = 0
            expanded_recv_x = fold_expanded(expanded_recv_x, expanded_safe_indices, expanded_mask)
            expanded_recv_topk_weights = expanded_recv_topk_weights[expanded_safe_indices]

            # Cached checks
            # cached dispatch 应与首次 dispatch 的数据和 routing metadata 完全一致。
            if use_fp8_dispatch:
                assert torch.equal(recv_x[0], cached_recv_x[0])
                assert torch.equal(recv_x[1], cached_recv_x[1])
            else:
                assert torch.equal(recv_x, cached_recv_x)
            assert torch.equal(recv_topk_idx, cached_recv_topk_idx)
            assert torch.equal(handle.dst_buffer_slot_idx, cached_handle.dst_buffer_slot_idx)
            assert torch.equal(handle.psum_num_recv_tokens_per_scaleup_rank, cached_handle.psum_num_recv_tokens_per_scaleup_rank)
            assert handle.num_recv_tokens_per_expert_list == cached_handle.num_recv_tokens_per_expert_list

            # Check dispatch expert count
            # 校验每个 local expert 接收到的 token 数及其 alignment 后的 prefix-sum metadata。
            assert recv_x_bf16.size() == ref_recv_x_bf16.size(), f'{recv_x_bf16.size()=}, {ref_recv_x_bf16.size()=}'
            assert recv_x_bf16.size(0) == num_recv_tokens
            for i in range(num_local_experts if args.do_cpu_sync else 0):
                ref_count = (ref_recv_topk_idx == i).sum().item()
                aligned_ref_count = align(ref_count, expert_alignment)
                assert ref_count == cumulative_local_expert_recv_stats[i].item(),\
                    f'{i}, {ref_count}, {cumulative_local_expert_recv_stats[i].item()}'
                assert aligned_ref_count == handle.num_recv_tokens_per_expert_list[i]
            psum_num_recv_tokens_per_expert_list = [0] + handle.psum_num_recv_tokens_per_expert.tolist()
            expanded_psum_num_recv_tokens_per_expert_list = [0] + expanded_handle.psum_num_recv_tokens_per_expert.tolist()
            for i in range(num_local_experts):
                ref_count = (ref_recv_topk_idx == i).sum().item()
                count = psum_num_recv_tokens_per_expert_list[i + 1] - psum_num_recv_tokens_per_expert_list[i]
                expanded_count = (expanded_psum_num_recv_tokens_per_expert_list[i + 1] -
                                  align(expanded_psum_num_recv_tokens_per_expert_list[i], expert_alignment))
                assert align(ref_count, expert_alignment) == count, f'{buffer.rank_idx=}, {i=}, {ref_count=}, {count=}'
                assert ref_count == expanded_count, f'{ref_count=}, {expanded_count=}'

            # Check dispatch scale-up received token psum
            # 校验按 scale-up rank 聚合后的接收 token prefix-sum 是否等于参考实现统计。
            psum_num_recv_tokens_per_scaleup_rank_list = [0] + handle.psum_num_recv_tokens_per_scaleup_rank.tolist()
            for i in range(num_scaleup_ranks):
                count = psum_num_recv_tokens_per_scaleup_rank_list[i + 1] - psum_num_recv_tokens_per_scaleup_rank_list[i]
                ref_count = sum(ref_num_recv_tokens_per_rank[i::num_scaleup_ranks])
                assert count == ref_count, f'{ref_count=}, {count=}'

            # Check dispatch data
            # 对 expanded 和 unexpanded 两种 dispatch 输出逐 rank 排序后与参考结果做 bitwise 对比。
            for check_recv_x, check_recv_topk_idx, check_recv_topk_weights, check_handle in (
                (expanded_recv_x, None, expanded_recv_topk_weights, expanded_handle),  # Expanded
                (recv_x, recv_topk_idx, recv_topk_weights, handle),  # Unexpanded
            ):
                for i in range(buffer.num_ranks):
                    rank_start_idx = sum(ref_num_recv_tokens_per_rank[:i])
                    rank_end_idx = rank_start_idx + ref_num_recv_tokens_per_rank[i]
                    sorted_metadata = torch.sort(check_handle.recv_src_metadata[:, 0])
                    sorted_indices = sorted_metadata.indices[rank_start_idx:rank_end_idx]
                    sorted_values = sorted_metadata.values[rank_start_idx:rank_end_idx]
                    assert torch.equal(ref_recv_src_token_idx[rank_start_idx:rank_end_idx], sorted_values)

                    # Data should be bitwise identical
                    # 分别检查 topk_weights、topk_idx 和 hidden states；masked expert 的权重先置零再比较。
                    check_list = [(ref_recv_topk_weights, check_recv_topk_weights, True)]
                    if check_recv_topk_idx is not None:
                        check_list.append((ref_recv_topk_idx, check_recv_topk_idx, False))
                    if use_fp8_dispatch:
                        check_list.append((ref_recv_x[0], check_recv_x[0], False))
                        check_list.append((ref_recv_x[1], check_recv_x[1], False))
                    else:
                        check_list.append((ref_recv_x, check_recv_x, False))
                    ref_mask = ref_recv_topk_idx[rank_start_idx:rank_end_idx] < 0
                    for ref_t, t, do_mask in check_list:
                        ref_t = ref_t[rank_start_idx:rank_end_idx]
                        t = t[sorted_indices]
                        if do_mask:
                            ref_t = ref_t.masked_fill(ref_mask, 0)
                            t = t.masked_fill(ref_mask, 0)
                        assert torch.equal(ref_t, t), f'{ref_t=}, {t=}'

            # Combined data should also be bitwise-identical
            # combine 输出必须与参考 combine 结果 bitwise 一致，topk_weights 也应恢复为原始权重。
            assert torch.equal(combined_x, ref_combined_y), \
                f'Diff: {calc_diff(combined_x, ref_combined_y)}'
            assert torch.equal(reduced_combined_x, ref_reduced_combined_y), \
                f'Diff: {calc_diff(reduced_combined_x, ref_reduced_combined_y)}'
            assert torch.equal(combined_topk_weights, topk_weights), \
                f'{calc_diff(combined_topk_weights, topk_weights)}'

        # Break on the first test case
        # 调试时可只跑第一组模式组合，避免完整组合测试耗时过长。
        if args.test_first_only:
            break
    dist_print('', once_in_node=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
# 中文函数注释：每个 spawned 进程执行的测试入口；初始化分布式环境、构造 ElasticBuffer、运行主测试和压力测试并清理资源。
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """
    中文参数说明：
    local_rank：当前进程在本机内部的 rank 编号，通常对应本机上的某一张 GPU。
    num_local_ranks：本机启动的进程数量，通常等于本机参与测试的 GPU 数量。
    args：命令行参数集合，包含 token 数、hidden 大小、expert 数、SM/QP 配置、测试开关等。
    """
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)  # 初始化分布式环境，返回全局 rank、总 rank 数和通信 group。
    # 中文函数注释：基于当前命令行参数和分布式 group 创建一个新的 `ElasticBuffer` 实例。
    def construct_elastic_buffer():
        return deep_ep.ElasticBuffer(group,  # 创建 ElasticBuffer，绑定当前分布式通信 group。
                                     num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,  # 设置每个 rank 最大 token 数和 hidden 维度。
                                     deterministic=args.deterministic,  # 设置是否启用 deterministic 算法。
                                     allow_hybrid_mode=args.allow_hybrid_mode,  # 设置是否允许 hybrid mode。
                                     allow_multiple_reduction=args.allow_multiple_reduction,  # 设置 combine 是否允许多阶段/多次 reduction。
                                     prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),  # 设置是否优先让通信与计算 overlap。
                                     sl_idx=args.sl_idx,  # 设置 RDMA service level / virtual lane 相关索引。
                                     num_allocated_qps=max(args.num_allocated_qps, args.num_qps),  # 设置实际分配的 QP 数，至少覆盖显式指定的 QP 数。
                                     explicitly_destroy=True,  # 设置需要显式调用 destroy 释放运行时资源。
                                     num_gpu_timeout_secs=args.num_gpu_timeout_secs,  # 设置 GPU 侧通信超时时间。
                                     num_cpu_timeout_secs=args.num_cpu_timeout_secs)  # 设置 CPU 侧同步/等待超时时间。

    buffer = construct_elastic_buffer()  # 构造本轮测试使用的 ElasticBuffer。

    # Warning in case of precise unbalanced ratio
    if args.precise_unbalanced_ratio:  # 如果启用了精确不均衡比例模式，则打印提示信息。
        dist_print('\033[33mWarning: Using precise unbalanced ratio mode. '  # 打印黄色 warning 前半段。
                   'Test data is manually constructed and may differ from real world distribution.\033[0m',  # 打印 warning 后半段并恢复终端颜色。
                   once_in_node=True)  # 只在每个节点打印一次，避免多 rank 重复刷屏。

    # Test MoE kernels
    test_dispatch_combine(buffer, args)  # 执行一次完整的 dispatch/combine 正确性和性能测试。

    # Pressure tests
    for seed in range(int(1e9) if args.do_pressure_test else 0):  # 如果开启压力测试，则用大量 seed 循环重复测试；否则循环次数为 0。
        if not args.reuse_elastic_buffer:  # 如果不复用 buffer，则每轮压力测试都重新创建 ElasticBuffer。
            # Recreate elastic buffer
            buffer.destroy()  # 销毁上一轮测试使用的 ElasticBuffer 和相关通信资源。
            buffer = construct_elastic_buffer()  # 为当前 seed 重新构造 ElasticBuffer。

        assert not args.skip_check  # 压力测试必须开启正确性检查，避免只跑性能而漏掉错误。
        dist_print(f'Testing with {seed=} ...', once_in_node=True)  # 打印当前压力测试使用的 seed。
        init_seed(seed)  # 使用当前 seed 初始化随机数，生成不同测试数据。
        test_dispatch_combine(buffer, args)  # 使用当前 seed 对 dispatch/combine 再执行一轮完整测试。

    # Destroy the runtime and communication group
    buffer.destroy()  # 测试结束后销毁 ElasticBuffer，释放 DeepEP 运行时资源。
    dist.destroy_process_group()  # 销毁 PyTorch 分布式通信进程组。


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test elastic EP kernels')  # 创建命令行参数解析器，用于配置 Elastic EP kernel 测试

    # 资源相关参数设置
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')  # 设置要启动的测试进程数量
    parser.add_argument('--num-sms', type=int, default=0, help='Number of SMs to use (0 means auto)')  # 设置使用的 GPU SM 数量，0 表示自动选择
    parser.add_argument('--num-qps', type=int, default=0, help='Number of QPs to use (0 means auto)')  # 设置通信使用的 QP 数量，0 表示自动选择
    parser.add_argument('--num-allocated-qps', type=int, default=0, help='Number of QPs to allocate (0 means auto)')  # 设置预分配的 QP 数量，0 表示自动选择
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100, help='Timeout in seconds (GPU side)')  # 设置 GPU 侧超时时间，单位为秒
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100, help='Timeout in seconds (CPU side)')  # 设置 CPU 侧超时时间，单位为秒
    parser.add_argument('--sl-idx', type=int, default=0, help='SL index')  # 设置通信使用的 service level 索引

    # 模型相关参数设置
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens')  # 设置每个 rank 参与测试的 token 数量
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')  # 设置 token hidden 维度大小
    parser.add_argument('--num-topk', type=int, default=6, help='Number of top-k experts')  # 设置每个 token 选择的 top-k 专家数量
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')  # 设置 MoE 专家总数

    # 测试场景相关参数设置
    parser.add_argument('--do-cpu-sync', type=int, default=1, help='Whether to do CPU sync')  # 控制测试过程中是否执行 CPU 同步
    parser.add_argument('--allow-hybrid-mode', type=int, default=1, help='Whether to allow hybrid mode')  # 控制是否允许 hybrid dispatch/combine 模式
    parser.add_argument('--allow-multiple-reduction', type=int, default=1, help='Whether to allow multiple reductions')  # 控制是否允许同一目标发生多次 reduction
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=0, help='Whether to prefer overlap with compute')  # 控制是否优先选择与计算重叠的执行方式
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithm')  # 启用确定性算法，便于稳定复现测试结果

    # 测试行为相关参数设置
    parser.add_argument('--seed', type=int, default=0, help='Default seed for pressure tests')  # 设置压力测试使用的随机种子
    parser.add_argument('--skip-check', action='store_true', help='Whether to skip correctness checks')  # 跳过结果正确性校验
    parser.add_argument('--skip-perf-test', action='store_true', help='Whether to skip performance tests')  # 跳过性能测试和 profiling
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')  # 启用压力测试流程
    parser.add_argument('--reuse-elastic-buffer', action='store_true', help='Whether to reuse elastic buffer for each test')  # 在多个测试用例之间复用 ElasticBuffer
    parser.add_argument('--test-first-only', action='store_true', help='Only test the first case')  # 只运行枚举出的第一个测试用例
    parser.add_argument('--unbalanced-ratio', type=float, default=1.0, help='The MoE unbalanced ratio')  # 设置 MoE 路由不均衡比例
    parser.add_argument('--precise-unbalanced-ratio', action='store_true', help='Generate topk index with precise unbalanced ratio')  # 按精确不均衡比例生成 top-k 索引
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')  # 设置需要 mask 掉的专家选择比例
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')  # 设置 profiling trace JSON 的导出目录
    parser.add_argument('--ignore-local-traffic', action='store_true', help='Whether to ignore local traffic during bandwidth calculation')  # 计算带宽时忽略本地通信流量
    args = parser.parse_args()  # 解析命令行参数并保存到 args

    # 创建 profiling trace 导出目录
    if args.dump_profile_traces:  # 只有用户指定导出目录时才创建目录
        os.makedirs(args.dump_profile_traces, exist_ok=True)  # 创建 trace 目录，目录已存在时不报错

    # 启动多进程测试
    num_processes = args.num_processes  # 从命令行参数中读取要启动的进程数量
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)  # 按进程数量启动 test_loop，每个进程通常对应一个本地 rank (一般来说，一个本地rank对应一个GPU。)
