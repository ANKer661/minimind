from __future__ import annotations

import argparse
import sys

from evaluation.launch import (
    module_command,
    run,
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


def parallel_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--pp_size",
        str(args.pp_size),
        "--tp_size",
        str(args.tp_size),
    ]
    result.extend(["--cp_size", str(args.cp_size if args.cp else 1)])
    if args.cp:
        result.append("--cp")
    return result


def validate_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    cp_size = args.cp_size if args.cp else 1
    return torchrun_module_command(
        "evaluation.validate",
        args.pp_size * cp_size * args.tp_size,
        [*parallel_args(args), *passthrough],
    )


def benchmark_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    return module_command(
        "evaluation.benchmark",
        ["--kind", args.kind, *parallel_args(args), *passthrough],
    )


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser_argv, passthrough = split_passthrough(argv)
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for MiniMind parallel evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="run numerical parity checks")
    add_parallel_args(validate)

    benchmark = subparsers.add_parser("benchmark", help="run memory or throughput benchmark")
    benchmark.add_argument("--kind", choices=("memory", "throughput"), required=True)
    add_parallel_args(benchmark)

    return parser.parse_args(parser_argv), passthrough


def main() -> None:
    args, passthrough = parse_args(sys.argv[1:])
    if args.command == "validate":
        command = validate_command(args, passthrough)
    else:
        command = benchmark_command(args, passthrough)

    raise SystemExit(run(command))


if __name__ == "__main__":
    main()
