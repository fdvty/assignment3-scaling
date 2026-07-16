# Local 4090 Scaling-Law Results

This is a local reproduction of Assignment 3 Problem 3.3, not an official Stanford API submission.
Fitted result file: `artifacts/local_4090_scaling/results_bigdata_ext.jsonl`.

## Best IsoFLOPs Runs

| target C | model | N non-emb params | train tokens | val loss | seconds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3.0000e+09 | L1_D16_H2 | 3.0720e+03 | 1.6179e+05 | 5.468694 | 0.77 |
| 1.0000e+10 | L1_D16_H2 | 3.0720e+03 | 5.4170e+05 | 4.503718 | 1.52 |
| 3.0000e+10 | L1_D16_H2 | 3.0720e+03 | 1.6271e+06 | 3.775930 | 5.05 |
| 1.0000e+11 | L1_D16_H2 | 3.0720e+03 | 5.4252e+06 | 3.100122 | 15.93 |
| 3.0000e+11 | L1_D16_H2 | 3.0720e+03 | 1.6275e+07 | 2.817517 | 47.86 |
| 1.0000e+12 | L1_D16_H2 | 3.0720e+03 | 5.4253e+07 | 2.341096 | 159.84 |

## Fits

- N_opt(C) = 10^3.487421 * C^0.000000; R2(log10) = 0.000000.
- D_opt(C) = 10^-4.275331 * C^1.000856; R2(log10) = 1.000000.
- Loss fit: L(C) = 16.737218 + -1.217051 log10(C); R2 = 0.966452.

## Prediction

For target C = 1.0000e+12, predicted N_opt = 3.0720e+03, predicted D_opt = 5.4317e+07, predicted validation loss = 2.132611.
