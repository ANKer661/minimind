# Parallel Evaluation

`evaluation` 提供两个统一入口：

- `validate`：比较并行模型与 Dense 模型的数值一致性；
- `benchmark`：测量显存或吞吐，并支持扫描模型层数与序列长度。

所有任务均面向 CUDA/NCCL 环境。

## 命令结构

```bash
python -m evaluation.parallel <validate|benchmark> [并行规模] -- [任务参数]
```

`--` 前描述 TP、CP、PP 的规模：

```text
--tp_size <size>
--pp_size <size>
--cp --cp_size <size>
```

`--` 后的参数交给 validate 或 benchmark。没有任务参数时可以省略 `--`。

### 任务参数

Validate 和 benchmark 都可以在 `--` 后控制模型结构、输入规模和并行算法：

| 参数 | 含义 |
|---|---|
| `--hidden_size` | Transformer hidden dimension |
| `--num_hidden_layers` | Transformer 层数；benchmark 可传多个值进行扫描 |
| `--num_attention_heads` | Query head 数量 |
| `--num_key_value_heads` | Key/Value head 数量 |
| `--vocab_size` | Embedding 和 LM head 的词表大小 |
| `--seq_len` | 序列长度；benchmark 可传多个值进行扫描 |
| `--micro_batch_size` | 每个 microbatch 的样本数 |
| `--num_microbatches` | 一个训练 step 累积的 microbatch 数量 |
| `--dtype` | 模型和计算使用的数据类型 |
| `--learning_rate` | AdamW learning rate |

并行行为通过下面的任务参数控制：

| 参数 | 含义 |
|---|---|
| `--cp_comm_type all_gather\|a2a` | CP Attention 的通信实现 |
| `--pp_schedule gpipe\|1f1b` | Pipeline schedule |
| `--sequence_parallel` | 启用 TP Sequence Parallelism |
| `--async_communication` | 启用 TP 异步通信/重计算实现 |
| `--vocab_parallel` | 启用 Vocab Parallelism |

Validate 另外提供 `--optimizer_steps`、`--check_backward` 和 `--atol` 等数值检查参数；
benchmark 另外提供 `--modes`、`--batch_policy`、`--warmup_iters`、
`--benchmark_iters`、`--flash_attn`、`--output_csv` 和 `--output_plot`。

完整参数及默认值以具体任务的帮助信息为准：

```bash
python -m evaluation.validate --help
python -m evaluation.benchmark --help
```

## Validate

Validate 会构造一份 Dense 模型作为数值基线，并比较并行模型的 logits、loss、参数梯度
和多步 AdamW 更新结果。

例如验证 2 TP × 2 CP × 2 PP，共使用 8 张 GPU：

```bash
python -m evaluation.parallel validate \
  --tp_size 2 \
  --pp_size 2 \
  --cp \
  --cp_size 2 \
  -- \
  --cp_comm_type a2a \
  --sequence_parallel \
  --async_communication \
  --vocab_parallel \
  --pp_schedule 1f1b \
  --optimizer_steps 10
```

Validate 一次只运行一个显式配置，实际启动进程数为：

```text
world_size = tp_size × cp_size × pp_size
```

未启用 CP 时，`cp_size` 按 1 计算。

## Benchmark

Benchmark 测量不同并行策略配置的显存或吞吐，并支持扫描模型层数或序列长度。

例如使用两张 GPU，依次运行 DDP、基础 TP 和四个 TP 功能变体，并
扫描 8、16、32 层模型的峰值显存：

```bash
python -m evaluation.parallel benchmark \
  --kind memory \
  --tp_size 2 \
  --pp_size 1 \
  -- \
  --modes ddp tp tp_sp tp_sp_async tp_vp tp_sp_async_vp \
  --batch_policy fixed_per_rank \
  --num_hidden_layers 8 16 32 \
  --seq_len 2048 \
  --micro_batch_size 4 \
  --num_microbatches 1
```

Benchmark 支持两种输出：

```text
--kind memory      峰值 GPU 显存、step time 和 tokens/s
--kind throughput  全局 tokens/s 和峰值 GPU 显存
```

### Mode、Preset 与 TP feature flags

`--modes` 后面的每个名字都是一个 preset，表示一套预先定义好的 benchmark 配置。
Benchmark 会按顺序为每个 mode 启动一次 worker；例如：

```text
--modes ddp tp tp_sp
```

表示依次运行 DDP、基础 TP 和 TP + SP 三组实验。`ddp`、`tp` 等 topology preset 决定
使用哪些并行维度，而 TP feature preset 进一步固定 TP 内部启用哪些功能。

下面四个名字是用于比较 TP 内部变体的 feature preset：

| Mode preset | 固定功能 |
|---|---|
| `tp_sp` | TP + SP |
| `tp_sp_async` | TP + SP + Async/Recompute |
| `tp_vp` | TP + VP |
| `tp_sp_async_vp` | TP + SP + Async/Recompute + VP |

这些 preset 的功能组合由名称固定，不受命令行中的 TP feature flags 覆盖。基础 `tp`
没有固定功能；未传 feature flags 时，它就是普通 TP。

除了使用固定 preset，也可以临时给 TP 添加 **TP feature flags**：

例如：

```bash
--modes tp --sequence_parallel --async_communication --vocab_parallel
```

表示使用 TP，并在其中开启 SP 和 VP 和 Async/Recompute，等价于 

```bash
--modes tp_sp_async_vp
```

### Memory 与 Throughput

`fixed_per_rank` 适合观察模型切分带来的单卡显存变化，但此时 DDP 与模型并行配置的
global batch 不同，不应直接比较整体吞吐。需要相同 global batch 时使用
`--batch_policy fixed_global`。

Throughput benchmark 固定使用 `fixed_global`，保证不同 mode 处理相同数量的 token。
将完整示例中的 kind 改成 throughput 即可运行吞吐测试：

```bash
python -m evaluation.parallel benchmark --kind throughput ...
```

### 扫描层数或序列长度

给 `--num_hidden_layers` 传入多个值，会扫描模型规模：

```text
--num_hidden_layers 2 4 8 16 32
--seq_len 2048
```

给 `--seq_len` 传入多个值，会扫描序列长度：

```text
--num_hidden_layers 16
--seq_len 2048 4096 8192 16384
```

一次只能扫描一个维度，不能同时为层数和序列长度传入多个值。

结果默认保存为 CSV 和 PNG，也可以显式指定路径：

```text
--output_csv results.csv
--output_plot results.png
```

## 多维并行、Preset 与 Config

`--tp_size`、`--cp_size`、`--pp_size` 给出各并行维度的候选规模，`--modes` 决定每次
benchmark 实际启用哪些维度。

### Topology preset

Topology preset 固定启用哪些并行轴，维度大小读取命令行：

| Mode | 并行拓扑 |
|---|---|
| `ddp` | DistributedDataParallel 基线 |
| `tp` | TP |
| `cp` | CP |
| `pp` | PP |
| `tp_cp` | TP × CP |
| `tp_pp` | TP × PP |
| `pp_cp` | PP × CP |
| `tp_cp_pp` | TP × CP × PP |

包含 TP 的 topology preset 会继承命令行中的 TP feature flags来控制 TP 内部的功能组合：

```text
--sequence_parallel
--async_communication
--vocab_parallel
```

包含 CP 的 preset 必须在统一入口中传入
`--cp` 和对应的 `--cp_size`。

### Config

`config` 不固定并行拓扑，完全读取入口给出的 TP、CP、PP size 和 TP feature flags，适合
运行一个临时的任意组合：

```bash
python -m evaluation.parallel benchmark \
  --kind memory \
  --tp_size 2 \
  --pp_size 2 \
  --cp \
  --cp_size 2 \
  -- \
  --modes config \
  --sequence_parallel \
  --async_communication \
  --vocab_parallel \
  --pp_schedule 1f1b \
  --num_hidden_layers 16 \
  --seq_len 4096
```

当 modes 中包含 DDP 时，DDP 会使用与非 DDP 对照组相同的 GPU 数量。如果同一次
benchmark 中的非 DDP modes 使用不同 world size，则需要拆成多次运行，避免用一份
DDP 结果同时比较不同 GPU 数量。

统一入口中的 TP、CP、PP 参数可以通过下面的帮助信息查看：

```bash
python -m evaluation.parallel validate --help
python -m evaluation.parallel benchmark --help
```
