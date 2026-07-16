# Chinchilla IsoFLOPs Results

For each compute budget, the optimum is the run with the lowest final loss.

## IsoFLOPs Optima

| C FLOPs | N_opt parameters | D_opt tokens | final loss |
| ---: | ---: | ---: | ---: |
| 6.0000e+18 | 7.6209e+08 | 1.3122e+09 | 5.899930 |
| 1.0000e+19 | 8.0665e+08 | 2.0662e+09 | 5.617943 |
| 3.0000e+19 | 1.5369e+09 | 3.2534e+09 | 5.107177 |
| 6.0000e+19 | 1.9520e+09 | 5.1228e+09 | 4.830586 |
| 1.0000e+20 | 3.2534e+09 | 5.1228e+09 | 4.652893 |
| 3.0000e+20 | 5.9038e+09 | 8.4691e+09 | 4.311219 |
| 6.0000e+20 | 6.9711e+09 | 1.4345e+10 | 4.121241 |
| 1.0000e+21 | 6.8593e+09 | 2.4298e+10 | 4.002835 |
| 3.0000e+21 | 1.2149e+10 | 4.1156e+10 | 3.773188 |

## Fits

- Model size: N_opt(C) = 10^0.065733 * C^0.468683; R2(log10) = 0.978704.
- Dataset size: D_opt(C) = 10^-0.843884 * C^0.531317; R2(log10) = 0.983351.

## Predictions

| C FLOPs | predicted N_opt parameters | predicted D_opt tokens |
| ---: | ---: | ---: |
| 1.0000e+23 | 7.0054e+10 | 2.3791e+11 |
| 1.0000e+24 | 2.0612e+11 | 8.0860e+11 |
