import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_tp import (
    TPContext,
    TPMiniMindForCausalLM,
    shard_state_dict_for_tp,
)


COLUMN_PARALLEL_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
)

ROW_PARALLEL_SUFFIXES = (
    "o_proj.weight",
    "down_proj.weight",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def global_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def benchmark(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    benchmark_iters: int,
    backward: bool,
) -> float:
    def run_once() -> None:
        model.zero_grad(set_to_none=True)
        if backward:
            output = model(input_ids)
            output.logits.float().mean().backward()
        else:
            with torch.no_grad():
                model(input_ids)

    for _ in range(warmup_iters):
        run_once()
    synchronize(device)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(benchmark_iters):
        run_once()
    synchronize(device)
    elapsed = time.perf_counter() - start

    return global_max(elapsed / benchmark_iters * 1000, device)


TP_GRAD_SUFFIXES = COLUMN_PARALLEL_SUFFIXES + ROW_PARALLEL_SUFFIXES + (
    "q_norm.weight",
    "k_norm.weight",
)


def compare_tp_gradients(
    dense_model: MiniMindForCausalLM,
    tp_model: TPMiniMindForCausalLM,
    tp_context: TPContext,
) -> tuple[float, list[tuple[str, float, float, float]]]:
    dense_params = dict(dense_model.named_parameters())
    metrics = []
    overall_max_diff = 0.0

    for name, tp_param in tp_model.named_parameters():
        if not name.endswith(TP_GRAD_SUFFIXES):
            continue

        dense_param = dense_params[name]
        assert tp_param.grad is not None, f"missing TP gradient: {name}"
        assert dense_param.grad is not None, f"missing dense gradient: {name}"

        expected_grad = dense_param.grad
        if name.endswith(COLUMN_PARALLEL_SUFFIXES):
            expected_grad = expected_grad.chunk(tp_context.world_size, dim=0)[tp_context.rank]
        elif name.endswith(ROW_PARALLEL_SUFFIXES):
            expected_grad = expected_grad.chunk(tp_context.world_size, dim=1)[tp_context.rank]

        error = (tp_param.grad - expected_grad).float()
        expected = expected_grad.float()
        error_sum = error.abs().sum().to(torch.float64)
        error_count = torch.tensor(error.numel(), device=error.device, dtype=torch.float64)
        error_sq_sum = error.square().sum().to(torch.float64)
        expected_sq_sum = expected.square().sum().to(torch.float64)
        max_error = error.abs().max().to(torch.float64)

        dist.all_reduce(error_sum, op=dist.ReduceOp.SUM, group=tp_context.group)
        dist.all_reduce(error_count, op=dist.ReduceOp.SUM, group=tp_context.group)
        dist.all_reduce(error_sq_sum, op=dist.ReduceOp.SUM, group=tp_context.group)
        dist.all_reduce(expected_sq_sum, op=dist.ReduceOp.SUM, group=tp_context.group)
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX, group=tp_context.group)

        mean_error = (error_sum / error_count).item()
        relative_l2 = (
            torch.sqrt(error_sq_sum) / (torch.sqrt(expected_sq_sum) + 1e-12)
        ).item()
        max_error_value = max_error.item()
        overall_max_diff = max(overall_max_diff, max_error_value)
        metrics.append((name, mean_error, max_error_value, relative_l2))

    return overall_max_diff, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MiniMind tensor parallelism")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--check_backward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--benchmark_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    assert world_size == args.tp_size, "TP v1 requires world_size == tp_size"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU validation only supports float32")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tp_context = TPContext(
        group=dist.group.WORLD,
        world_size=world_size,
        rank=rank,
    )
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.seq_len, 32),
        dropout=0.0,
        flash_attn=False,
        use_moe=False,
    )

    dense_model = MiniMindForCausalLM(config).to(device=device, dtype=dtype)
    tp_model = TPMiniMindForCausalLM(tp_context, config).to(device=device, dtype=dtype)
    tp_state_dict = shard_state_dict_for_tp(dense_model.state_dict(), tp_context)
    tp_model.load_state_dict(tp_state_dict, strict=True)

    input_ids = torch.empty(
        args.batch_size,
        args.seq_len,
        device=device,
        dtype=torch.long,
    )
    if rank == 0:
        input_ids.random_(0, args.vocab_size)
    dist.broadcast(input_ids, src=0)
    labels = input_ids.clone()

    dense_model.eval()
    tp_model.eval()
    with torch.no_grad():
        dense_output = dense_model(input_ids, labels=labels)
        tp_output = tp_model(input_ids, labels=labels)

    logits_diff = global_max(
        (dense_output.logits - tp_output.logits).abs().max().item(),
        device,
    )
    loss_diff = global_max(
        abs(dense_output.loss.item() - tp_output.loss.item()),
        device,
    )
    atol = args.atol if args.atol is not None else (1e-4 if dtype == torch.float32 else 5e-2)

    grad_diff = None
    grad_metrics = []
    if args.check_backward:
        dense_model.zero_grad(set_to_none=True)
        tp_model.zero_grad(set_to_none=True)
        dense_model(input_ids, labels=labels).loss.backward()
        tp_model(input_ids, labels=labels).loss.backward()
        grad_diff, grad_metrics = compare_tp_gradients(
            dense_model,
            tp_model,
            tp_context,
        )

    dense_forward_ms = tp_forward_ms = None
    dense_train_ms = tp_train_ms = None
    if args.benchmark_iters > 0:
        dense_forward_ms = benchmark(
            dense_model,
            input_ids,
            device,
            args.warmup_iters,
            args.benchmark_iters,
            backward=False,
        )
        tp_forward_ms = benchmark(
            tp_model,
            input_ids,
            device,
            args.warmup_iters,
            args.benchmark_iters,
            backward=False,
        )
        if args.check_backward:
            dense_train_ms = benchmark(
                dense_model,
                input_ids,
                device,
                args.warmup_iters,
                args.benchmark_iters,
                backward=True,
            )
            tp_train_ms = benchmark(
                tp_model,
                input_ids,
                device,
                args.warmup_iters,
                args.benchmark_iters,
                backward=True,
            )

    if rank == 0:
        print(f"forward max logits diff: {logits_diff:.6e}")
        print(f"forward loss diff:       {loss_diff:.6e}")
        if grad_diff is not None:
            print(f"backward max grad diff:  {grad_diff:.6e}")
            current_layer = None
            for name, mean_error, max_error, relative_l2 in grad_metrics:
                layer_id = name.split(".")[2]
                if layer_id != current_layer:
                    current_layer = layer_id
                    print(f"layer {layer_id}:")
                short_name = name.split(f"model.layers.{layer_id}.", 1)[1]
                print(
                    f"  {short_name:<31} "
                    f"mean={mean_error:.3e} max={max_error:.3e} rel_l2={relative_l2:.3e}"
                )
        if dense_forward_ms is not None and tp_forward_ms is not None:
            print(f"dense forward:           {dense_forward_ms:.3f} ms")
            print(f"TP forward:              {tp_forward_ms:.3f} ms")
        if dense_train_ms is not None and tp_train_ms is not None:
            print(f"dense forward+backward:  {dense_train_ms:.3f} ms")
            print(f"TP forward+backward:     {tp_train_ms:.3f} ms")

    passed = logits_diff <= atol and loss_diff <= atol
    if grad_diff is not None:
        passed = passed and grad_diff <= atol

    passed_tensor = torch.tensor(int(passed), device=device)
    dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()

    if passed_tensor.item() != 1:
        raise RuntimeError(f"TP parity check failed with atol={atol}")


if __name__ == "__main__":
    main()
