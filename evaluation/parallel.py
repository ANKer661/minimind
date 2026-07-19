from __future__ import annotations

import argparse
import sys

from evaluation.launch import (
    module_command,
    python_command,
    run,
    torchrun_command,
    torchrun_module_command,
)


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def add_parallel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument(
        "--cp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cp_size", type=int, default=1)


def parallel_args(
    args: argparse.Namespace,
    *,
    include_cp: bool = True,
) -> list[str]:
    result = [
        "--pp_size",
        str(args.pp_size),
        "--tp_size",
        str(args.tp_size),
    ]
    if not include_cp:
        return result
    result.extend(["--cp_size", str(args.cp_size if args.cp else 1)])
    if args.cp:
        result.append("--cp")
    return result


def validate_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    if args.target == "tp":
        return torchrun_command(
            "demo_tensor_parallel.py",
            args.tp_size,
            ["--tp_size", str(args.tp_size), *passthrough],
        )

    cp_size = args.cp_size if args.cp else 1
    return torchrun_module_command(
        "evaluation.validate",
        args.pp_size * cp_size * args.tp_size,
        [*parallel_args(args), *passthrough],
    )


def benchmark_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    if args.target == "tp":
        if args.kind != "memory":
            raise ValueError("TP benchmark currently supports only --kind memory")
        return python_command("benchmark_tensor_parallel_memory.py", passthrough)

    if args.kind == "throughput":
        if args.target != "pp":
            raise ValueError("throughput benchmark currently supports only --target pp")
        return module_command(
            "evaluation.throughput",
            [*parallel_args(args, include_cp=False), *passthrough],
        )

    benchmark_args = [*parallel_args(args), *passthrough]
    if args.target == "parallel" and "--modes" not in passthrough:
        benchmark_args.extend(["--modes", "parallel"])
    return module_command("evaluation.benchmark", benchmark_args)


def profile_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    if args.target != "tp":
        raise ValueError("profile currently supports only --target tp")

    command_args = [
        "--profile_only",
        "--profile_variant",
        args.variant,
    ]
    if args.layers is not None:
        command_args.extend(["--profile_layers", str(args.layers)])
    command_args.extend(passthrough)
    return python_command("benchmark_tensor_parallel_memory.py", command_args)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser_argv, passthrough = split_passthrough(argv)
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for MiniMind parallel evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="run numerical parity checks")
    validate.add_argument(
        "--target",
        choices=("tp", "pp", "parallel"),
        required=True,
    )
    add_parallel_args(validate)

    benchmark = subparsers.add_parser("benchmark", help="run memory or throughput benchmark")
    benchmark.add_argument(
        "--target",
        choices=("tp", "pp", "parallel"),
        required=True,
    )
    benchmark.add_argument("--kind", choices=("memory", "throughput"), required=True)
    add_parallel_args(benchmark)

    profile = subparsers.add_parser("profile", help="collect profiler traces")
    profile.add_argument("--target", choices=("tp",), required=True)
    profile.add_argument("--variant", required=True)
    profile.add_argument("--layers", type=int, default=None)

    return parser.parse_args(parser_argv), passthrough


def main() -> None:
    args, passthrough = parse_args(sys.argv[1:])
    if args.command == "validate":
        command = validate_command(args, passthrough)
    elif args.command == "benchmark":
        command = benchmark_command(args, passthrough)
    else:
        command = profile_command(args, passthrough)

    raise SystemExit(run(command))


if __name__ == "__main__":
    main()
