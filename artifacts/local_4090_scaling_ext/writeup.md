# Local 4090 Version of Problem 3.3

This is a local reproduction of the scaling-law leaderboard problem. It is not
an official Stanford API submission because I do not have Stanford API access.

## Machine

I attempted to use `netbd`, but it was not usable for this run:

- `netbd` has free RTX 4090 GPUs, but no `torch`, `numpy`, `pip`, or `uv`.
- Its root/overlay filesystem is full.
- I initially used `netbd:/share` by mistake. The persistent mount assigned to
  this account is `/home/liuzirui`; the failed `/share` installs were removed
  where the filesystem allowed it. Some corrupted remnants still require a
  container or ZFS restart because deleting them enters uninterruptible I/O.

All future `netbd` work must stay under these paths:

```text
/home/liuzirui/cs336-a3-work
/home/liuzirui/cs336-a3-env
/home/liuzirui/cs336-a3-cache
/home/liuzirui/cs336-a3-tmp
```

I therefore ran the local reproduction on `cafe`, which has RTX 4090 GPUs and
an existing CUDA PyTorch environment at `/share/miniconda3`.

## Data And Model

The local experiment uses a byte-level Transformer language model on the first
64 MB of:

```text
/share/models_lzw/data/raw/pretrain_hq.jsonl
```

The final `2^18` bytes are used for validation. Training is deterministic and
uses a fixed contiguous token order.

For each architecture, I estimate non-embedding parameters using the assignment
formula:

```text
N = 12 * n_layer * d_model^2
```

The searched model specs were:

```text
L1_D16_H2, L1_D24_H4, L1_D32_H4, L1_D48_H4,
L1_D64_H4, L2_D64_H4, L2_D96_H6, L2_D128_H8
```

## IsoFLOPs Design

For each compute budget `C`, each model is trained for:

```text
D = C / (6N)
```

rounded down to an integer number of optimizer steps. I used:

```text
C in {3e9, 1e10, 3e10, 1e11, 3e11, 1e12}
```

For each `C`, I selected the run with the lowest validation loss.

## Best Runs

| C | best model | N | D | val loss |
| ---: | --- | ---: | ---: | ---: |
| 3e9 | L1_D16_H2 | 3.072e3 | 1.6179e5 | 5.468694 |
| 1e10 | L1_D16_H2 | 3.072e3 | 5.4170e5 | 4.503718 |
| 3e10 | L1_D16_H2 | 3.072e3 | 1.6271e6 | 3.775930 |
| 1e11 | L1_D16_H2 | 3.072e3 | 5.4252e6 | 3.100122 |
| 3e11 | L1_D16_H2 | 3.072e3 | 1.6275e7 | 2.817517 |
| 1e12 | L1_D16_H2 | 3.072e3 | 5.4253e7 | 2.341096 |

## Scaling-Law Fit

The fitted power laws are:

```text
N_opt(C) = 10^3.487421 * C^0.000000
D_opt(C) = 10^-4.275331 * C^1.000856
```

The model-size fit is flat because the best model for every tested compute
budget is the smallest searched model. This means the local search space is
boundary-limited: within this byte-level setup, spending extra compute on more
tokens beat increasing model size.

For the target local budget:

```text
C = 1e12 FLOPs
```

the prediction is:

```text
N_opt = 3.072e3 non-embedding parameters
D_opt = 5.4317e7 train tokens
predicted validation loss = 2.132611
```

The best actually observed `1e12` run was:

```text
model = L1_D16_H2
N = 3.072e3
D = 5.4253e7
validation loss = 2.341096
```

## Original Cafe Reproduction Commands

These commands record the completed `cafe` run. Their `/share` paths belong to
`cafe` and must not be reused on `netbd`.

```sh
cd /share/cs336-a3-work/assignment3-scaling
CUDA_VISIBLE_DEVICES=0 /share/miniconda3/bin/python scripts/local_4090_scaling.py run-grid \
  --device cuda:0 \
  --data /share/models_lzw/data/raw/pretrain_hq.jsonl \
  --max-bytes 64000000 \
  --output artifacts/local_4090_scaling/results_bigdata_ext.jsonl

/share/miniconda3/bin/python scripts/local_4090_scaling.py fit \
  --results artifacts/local_4090_scaling/results_bigdata_ext.jsonl \
  --out-dir artifacts/local_4090_scaling_ext \
  --target-compute 1e12
```
