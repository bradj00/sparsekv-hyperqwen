| config | context | samples | tokens | mean KLD | median | p95 | p99 | max | top-1 agree |
|---|---|---|---|---|---|---|---|---|---|
| noise floor: dense vs dense, other card + other server | 32K | 15 | 3825 | 0.02300 | 0.000261 | 0.0994 | 0.2748 | 3.119 | 96.39% |
| noise floor: dense vs dense, other card + other server | 64K | 15 | 3825 | 0.01672 | 0.000277 | 0.0868 | 0.2069 | 0.862 | 96.76% |
| noise floor: dense vs dense, other card + other server | 128K | 15 | 3825 | 0.01982 | 0.000536 | 0.1048 | 0.2248 | 0.796 | 95.87% |
| noise floor: dense vs dense, other card + other server | 192K | 15 | 3825 | 0.02057 | 0.000443 | 0.1036 | 0.2662 | 1.982 | 95.90% |
| noise floor: dense vs dense, other card + other server | **all** | 60 | 15300 | 0.02003 | 0.000374 | 0.0991 | 0.2414 | 3.119 | 96.23% |
| noise floor: dense vs dense, same server repeat | 128K | 3 | 765 | 0.00000 | 0.000000 | 0.0000 | 0.0000 | 0.000 | 100.00% |
| noise floor: dense vs dense, same server repeat | 192K | 3 | 765 | 0.00000 | 0.000000 | 0.0000 | 0.0000 | 0.000 | 100.00% |
| noise floor: dense vs dense, same server repeat | **all** | 6 | 1530 | 0.00000 | 0.000000 | 0.0000 | 0.0000 | 0.000 | 100.00% |
| sparse GPU-only vs dense, same card + same server | 32K | 15 | 3825 | 0.01845 | 0.000110 | 0.0819 | 0.2091 | 3.908 | 96.94% |
| sparse GPU-only vs dense, same card + same server | 64K | 15 | 3825 | 0.01516 | 0.000159 | 0.0847 | 0.1971 | 0.468 | 96.86% |
| sparse GPU-only vs dense, same card + same server | 128K | 15 | 3825 | 0.01870 | 0.000405 | 0.1010 | 0.2026 | 0.741 | 96.08% |
| sparse GPU-only vs dense, same card + same server | 192K | 15 | 3825 | 0.02281 | 0.000495 | 0.1277 | 0.2930 | 0.795 | 95.84% |
| sparse GPU-only vs dense, same card + same server | **all** | 60 | 15300 | 0.01878 | 0.000277 | 0.0984 | 0.2245 | 3.908 | 96.43% |
| sparse + RAM tier int8 vs dense, same card | 32K | 9 | 2295 | 0.01797 | 0.000199 | 0.0974 | 0.2348 | 1.144 | 96.38% |
| sparse + RAM tier int8 vs dense, same card | 64K | 9 | 2295 | 0.02425 | 0.000452 | 0.1290 | 0.3061 | 1.506 | 96.30% |
| sparse + RAM tier int8 vs dense, same card | 128K | 9 | 2295 | 0.02850 | 0.001172 | 0.1443 | 0.3380 | 1.062 | 94.86% |
| sparse + RAM tier int8 vs dense, same card | 192K | 9 | 2295 | 0.02896 | 0.000691 | 0.1605 | 0.3782 | 1.187 | 95.73% |
| sparse + RAM tier int8 vs dense, same card | **all** | 36 | 9180 | 0.02492 | 0.000572 | 0.1354 | 0.3179 | 1.506 | 95.82% |
| sparse + RAM tier int4 (production) vs dense, same card | 32K | 15 | 3825 | 0.02410 | 0.000630 | 0.1167 | 0.3293 | 4.497 | 96.13% |
| sparse + RAM tier int4 (production) vs dense, same card | 64K | 15 | 3825 | 0.02212 | 0.000607 | 0.1169 | 0.2655 | 0.920 | 96.13% |
| sparse + RAM tier int4 (production) vs dense, same card | 128K | 15 | 3825 | 0.02788 | 0.001581 | 0.1409 | 0.3327 | 1.960 | 95.08% |
| sparse + RAM tier int4 (production) vs dense, same card | 192K | 15 | 3825 | 0.03334 | 0.001026 | 0.1740 | 0.4188 | 2.349 | 95.16% |
| sparse + RAM tier int4 (production) vs dense, same card | **all** | 60 | 15300 | 0.02686 | 0.000914 | 0.1385 | 0.3331 | 4.497 | 95.63% |
| sparse GPU-only vs dense, other card | 32K | 15 | 3825 | 0.02282 | 0.000271 | 0.1145 | 0.2630 | 1.467 | 96.10% |
| sparse GPU-only vs dense, other card | 64K | 15 | 3825 | 0.01936 | 0.000329 | 0.1098 | 0.2377 | 0.848 | 96.37% |
| sparse GPU-only vs dense, other card | 128K | 15 | 3825 | 0.02379 | 0.000651 | 0.1230 | 0.2642 | 1.352 | 95.32% |
| sparse GPU-only vs dense, other card | 192K | 15 | 3825 | 0.02862 | 0.000738 | 0.1559 | 0.3431 | 2.516 | 95.11% |
| sparse GPU-only vs dense, other card | **all** | 60 | 15300 | 0.02365 | 0.000472 | 0.1254 | 0.2787 | 2.516 | 95.73% |
| sparse + RAM tier int8 vs dense, other card | 32K | 9 | 2295 | 0.01684 | 0.000232 | 0.0981 | 0.2218 | 0.745 | 96.64% |
| sparse + RAM tier int8 vs dense, other card | 64K | 9 | 2295 | 0.02132 | 0.000512 | 0.1146 | 0.2423 | 1.809 | 96.34% |
| sparse + RAM tier int8 vs dense, other card | 128K | 9 | 2295 | 0.02800 | 0.001281 | 0.1407 | 0.3237 | 1.993 | 94.20% |
| sparse + RAM tier int8 vs dense, other card | 192K | 9 | 2295 | 0.02973 | 0.000734 | 0.1677 | 0.3877 | 1.106 | 95.73% |
| sparse + RAM tier int8 vs dense, other card | **all** | 36 | 9180 | 0.02397 | 0.000629 | 0.1289 | 0.3025 | 1.993 | 95.73% |
| sparse + RAM tier int4 (production) vs dense, other card | 32K | 15 | 3825 | 0.02240 | 0.000564 | 0.1073 | 0.2984 | 2.089 | 96.42% |
| sparse + RAM tier int4 (production) vs dense, other card | 64K | 15 | 3825 | 0.02311 | 0.000550 | 0.1230 | 0.2900 | 1.083 | 96.26% |
| sparse + RAM tier int4 (production) vs dense, other card | 128K | 15 | 3825 | 0.02632 | 0.001437 | 0.1319 | 0.3286 | 1.458 | 94.82% |
| sparse + RAM tier int4 (production) vs dense, other card | 192K | 15 | 3825 | 0.03406 | 0.001148 | 0.1807 | 0.4326 | 2.186 | 94.93% |
| sparse + RAM tier int4 (production) vs dense, other card | **all** | 60 | 15300 | 0.02647 | 0.000892 | 0.1353 | 0.3382 | 2.186 | 95.61% |

| config | slice | samples | mean KLD | p99 | top-1 agree |
|---|---|---|---|---|---|
| noise floor: dense vs dense, other card + other server | doc:code | 36 | 0.01450 | 0.2034 | 97.37% |
| noise floor: dense vs dense, other card + other server | doc:prose | 24 | 0.02832 | 0.2940 | 94.51% |
| noise floor: dense vs dense, other card + other server | task:continue | 20 | 0.03113 | 0.3141 | 94.53% |
| noise floor: dense vs dense, other card + other server | task:explain | 20 | 0.02610 | 0.2570 | 94.41% |
| noise floor: dense vs dense, other card + other server | task:recall | 20 | 0.00285 | 0.0458 | 99.75% |
| noise floor: dense vs dense, same server repeat | doc:prose | 6 | 0.00000 | 0.0000 | 100.00% |
| noise floor: dense vs dense, same server repeat | task:continue | 2 | 0.00000 | 0.0000 | 100.00% |
| noise floor: dense vs dense, same server repeat | task:explain | 2 | 0.00000 | 0.0000 | 100.00% |
| noise floor: dense vs dense, same server repeat | task:recall | 2 | 0.00000 | 0.0000 | 100.00% |
| sparse GPU-only vs dense, same card + same server | doc:code | 36 | 0.01351 | 0.1927 | 97.70% |
| sparse GPU-only vs dense, same card + same server | doc:prose | 24 | 0.02669 | 0.2576 | 94.53% |
| sparse GPU-only vs dense, same card + same server | task:continue | 20 | 0.03148 | 0.3402 | 94.84% |
| sparse GPU-only vs dense, same card + same server | task:explain | 20 | 0.02326 | 0.2205 | 94.57% |
| sparse GPU-only vs dense, same card + same server | task:recall | 20 | 0.00160 | 0.0275 | 99.88% |
| sparse + RAM tier int8 vs dense, same card | doc:code | 24 | 0.01837 | 0.2852 | 96.72% |
| sparse + RAM tier int8 vs dense, same card | doc:prose | 12 | 0.03803 | 0.3488 | 94.02% |
| sparse + RAM tier int8 vs dense, same card | task:continue | 12 | 0.03274 | 0.3463 | 94.54% |
| sparse + RAM tier int8 vs dense, same card | task:explain | 12 | 0.03613 | 0.3511 | 93.30% |
| sparse + RAM tier int8 vs dense, same card | task:recall | 12 | 0.00589 | 0.1322 | 99.61% |
| sparse + RAM tier int4 (production) vs dense, same card | doc:code | 36 | 0.01809 | 0.2625 | 97.03% |
| sparse + RAM tier int4 (production) vs dense, same card | doc:prose | 24 | 0.04002 | 0.3746 | 93.53% |
| sparse + RAM tier int4 (production) vs dense, same card | task:continue | 20 | 0.04022 | 0.3972 | 93.24% |
| sparse + RAM tier int4 (production) vs dense, same card | task:explain | 20 | 0.03303 | 0.3221 | 94.06% |
| sparse + RAM tier int4 (production) vs dense, same card | task:recall | 20 | 0.00734 | 0.1510 | 99.59% |
| sparse GPU-only vs dense, other card | doc:code | 36 | 0.01628 | 0.2361 | 97.09% |
| sparse GPU-only vs dense, other card | doc:prose | 24 | 0.03470 | 0.3328 | 93.68% |
| sparse GPU-only vs dense, other card | task:continue | 20 | 0.03740 | 0.3705 | 93.53% |
| sparse GPU-only vs dense, other card | task:explain | 20 | 0.03049 | 0.2912 | 93.84% |
| sparse GPU-only vs dense, other card | task:recall | 20 | 0.00305 | 0.0553 | 99.80% |
| sparse + RAM tier int8 vs dense, other card | doc:code | 24 | 0.01687 | 0.2578 | 96.91% |
| sparse + RAM tier int8 vs dense, other card | doc:prose | 12 | 0.03817 | 0.3683 | 93.37% |
| sparse + RAM tier int8 vs dense, other card | task:continue | 12 | 0.03187 | 0.3537 | 93.92% |
| sparse + RAM tier int8 vs dense, other card | task:explain | 12 | 0.03341 | 0.3124 | 93.76% |
| sparse + RAM tier int8 vs dense, other card | task:recall | 12 | 0.00663 | 0.1238 | 99.51% |
| sparse + RAM tier int4 (production) vs dense, other card | doc:code | 36 | 0.01887 | 0.3086 | 97.12% |
| sparse + RAM tier int4 (production) vs dense, other card | doc:prose | 24 | 0.03788 | 0.3876 | 93.33% |
| sparse + RAM tier int4 (production) vs dense, other card | task:continue | 20 | 0.03877 | 0.4040 | 93.78% |
| sparse + RAM tier int4 (production) vs dense, other card | task:explain | 20 | 0.03410 | 0.3507 | 93.41% |
| sparse + RAM tier int4 (production) vs dense, other card | task:recall | 20 | 0.00655 | 0.1193 | 99.63% |

Compact: mean KLD (nats/token) and top-1 agreement per context length

| comparison | 32K | 64K | 128K | 192K | all |
|---|---|---|---|---|---|
| noise floor: dense vs dense, other card + other server | 0.0230 / 96.4% | 0.0167 / 96.8% | 0.0198 / 95.9% | 0.0206 / 95.9% | 0.0200 / 96.2% |
| noise floor: dense vs dense, same server repeat | n/a | n/a | 0.0000 / 100.0% | 0.0000 / 100.0% | 0.0000 / 100.0% |
| sparse GPU-only vs dense, same card + same server | 0.0185 / 96.9% | 0.0152 / 96.9% | 0.0187 / 96.1% | 0.0228 / 95.8% | 0.0188 / 96.4% |
| sparse + RAM tier int8 vs dense, same card | 0.0180 / 96.4% | 0.0242 / 96.3% | 0.0285 / 94.9% | 0.0290 / 95.7% | 0.0249 / 95.8% |
| sparse + RAM tier int4 (production) vs dense, same card | 0.0241 / 96.1% | 0.0221 / 96.1% | 0.0279 / 95.1% | 0.0333 / 95.2% | 0.0269 / 95.6% |
| sparse GPU-only vs dense, other card | 0.0228 / 96.1% | 0.0194 / 96.4% | 0.0238 / 95.3% | 0.0286 / 95.1% | 0.0236 / 95.7% |
| sparse + RAM tier int8 vs dense, other card | 0.0168 / 96.6% | 0.0213 / 96.3% | 0.0280 / 94.2% | 0.0297 / 95.7% | 0.0240 / 95.7% |
| sparse + RAM tier int4 (production) vs dense, other card | 0.0224 / 96.4% | 0.0231 / 96.3% | 0.0263 / 94.8% | 0.0341 / 94.9% | 0.0265 / 95.6% |
