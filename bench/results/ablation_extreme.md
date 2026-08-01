# Ablation study

All suites, balanced mode, n=156 instances per row.
Δ columns are relative to the full system.

| ablation | ratio % | Δratio | answerability % | Δans | contradiction % | identifiers % | Δident | integrity % | latency ms |
|---|---|---|---|---|---|---|---|---|---|
| full system | 81.2 | +0.0 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 41.2 |
| no_ladder | 76.5 | -4.6 | 97.3 | +0.0 | 0.0 | 100.0 | +2.4 | 100.0 | 41.4 |
| no_attention | 81.1 | -0.1 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 37.9 |
| no_lexical | 81.1 | -0.0 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 38.5 |
| no_dedup | 79.8 | -1.4 | 97.0 | -0.3 | 0.0 | 97.6 | +0.0 | 100.0 | 28.5 |
| no_closure | 81.7 | +0.5 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 40.6 |
| no_order | 81.2 | +0.0 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 40.9 |
| no_repair | 82.8 | +1.6 | 60.4 | -36.9 | 0.0 | 97.6 | +0.0 | 100.0 | 39.7 |
| no_coverage | 80.6 | -0.6 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 40.4 |
| no_cov+no_attn | 80.7 | -0.5 | 97.3 | +0.0 | 0.0 | 97.6 | +0.0 | 100.0 | 37.5 |

## Reading the table

A mechanism earns its place if removing it *either* costs compression at equal quality *or* costs quality at equal compression. A row that matches the full system on both is a mechanism we should delete.
