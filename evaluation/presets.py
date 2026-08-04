from dataclasses import dataclass, replace
from typing import Literal


Runner = Literal["ddp", "tp", "general"]
Axis = Literal["tp", "cp", "pp"]


@dataclass(frozen=True)
class FeatureFlags:
    sequence_parallel: bool = False
    async_communication: bool = False
    vocab_parallel: bool = False


@dataclass(frozen=True)
class ModePreset:
    label: str
    runner: Runner
    axes: frozenset[Axis] | None
    features: FeatureFlags | None = None


@dataclass(frozen=True)
class ResolvedMode:
    name: str
    label: str
    runner: Runner
    pp_size: int
    tp_size: int
    cp_size: int
    cp_enabled: bool
    features: FeatureFlags
    world_size: int

    @property
    def sequence_parallel(self) -> bool:
        return self.features.sequence_parallel

    @property
    def async_communication(self) -> bool:
        return self.features.async_communication

    @property
    def vocab_parallel(self) -> bool:
        return self.features.vocab_parallel


NO_FEATURES = FeatureFlags()
MODE_PRESETS = {
    "ddp": ModePreset("DDP", "ddp", frozenset(), NO_FEATURES),
    "tp": ModePreset("TP", "tp", frozenset({"tp"})),
    "tp_sp": ModePreset(
        "TP + SP",
        "tp",
        frozenset({"tp"}),
        FeatureFlags(sequence_parallel=True),
    ),
    "tp_sp_async": ModePreset(
        "TP + SP + Async",
        "tp",
        frozenset({"tp"}),
        FeatureFlags(sequence_parallel=True, async_communication=True),
    ),
    "tp_vp": ModePreset(
        "TP + VP",
        "tp",
        frozenset({"tp"}),
        FeatureFlags(vocab_parallel=True),
    ),
    "tp_sp_async_vp": ModePreset(
        "TP + SP + Async + VP",
        "tp",
        frozenset({"tp"}),
        FeatureFlags(
            sequence_parallel=True,
            async_communication=True,
            vocab_parallel=True,
        ),
    ),
    "cp": ModePreset("CP", "general", frozenset({"cp"})),
    "pp": ModePreset("PP", "general", frozenset({"pp"})),
    "tp_cp": ModePreset("TP x CP", "general", frozenset({"tp", "cp"})),
    "tp_pp": ModePreset("TP x PP", "general", frozenset({"tp", "pp"})),
    "pp_cp": ModePreset("PP x CP", "general", frozenset({"pp", "cp"})),
    "tp_cp_pp": ModePreset(
        "TP x CP x PP",
        "general",
        frozenset({"tp", "cp", "pp"}),
    ),
    "config": ModePreset("Configured Parallelism", "general", None),
}
DEFAULT_MODES = ("ddp", "tp")


def _resolve_mode(
    name: str,
    *,
    pp_size: int,
    cp_size: int,
    tp_size: int,
    cp_enabled: bool,
    cli_features: FeatureFlags,
) -> ResolvedMode:
    preset = MODE_PRESETS[name]
    if preset.axes is None:
        use_tp = tp_size > 1
        use_cp = cp_enabled
        use_pp = pp_size > 1
    else:
        use_tp = "tp" in preset.axes
        use_cp = "cp" in preset.axes
        use_pp = "pp" in preset.axes

    if use_cp and not cp_enabled:
        raise ValueError(f"mode {name} requires --cp")
    if cp_enabled and cp_size < 1:
        raise ValueError("cp_size must be positive")

    effective_tp_size = tp_size if use_tp else 1
    effective_cp_size = cp_size if use_cp else 1
    effective_pp_size = pp_size if use_pp else 1
    features = preset.features
    if features is None:
        features = cli_features if use_tp else NO_FEATURES

    return ResolvedMode(
        name=name,
        label=preset.label,
        runner=preset.runner,
        pp_size=effective_pp_size,
        tp_size=effective_tp_size,
        cp_size=effective_cp_size,
        cp_enabled=use_cp,
        features=features,
        world_size=effective_pp_size * effective_cp_size * effective_tp_size,
    )


def resolve_modes(
    names: list[str],
    *,
    pp_size: int,
    cp_size: int,
    tp_size: int,
    cp_enabled: bool,
    cli_features: FeatureFlags,
) -> list[ResolvedMode]:
    resolved = [
        _resolve_mode(
            name,
            pp_size=pp_size,
            cp_size=cp_size,
            tp_size=tp_size,
            cp_enabled=cp_enabled,
            cli_features=cli_features,
        )
        for name in names
    ]
    comparison_world_sizes = {
        mode.world_size for mode in resolved if mode.runner != "ddp"
    }
    if len(comparison_world_sizes) > 1 and any(mode.runner == "ddp" for mode in resolved):
        details = ", ".join(
            f"{mode.name}={mode.world_size}"
            for mode in resolved
            if mode.runner != "ddp"
        )
        raise ValueError(
            "DDP cannot provide one fair baseline for modes with different "
            f"world sizes: {details}. Run them in separate benchmarks."
        )

    ddp_world_size = (
        next(iter(comparison_world_sizes))
        if comparison_world_sizes
        else pp_size * (cp_size if cp_enabled else 1) * tp_size
    )
    return [
        replace(mode, world_size=ddp_world_size)
        if mode.runner == "ddp"
        else mode
        for mode in resolved
    ]
