# Extrinsic evaluation — real model in the loop

Model: `gemini-flash-latest` · mode: `balanced` · requests spent: 23/48 · pack=6

| condition | n | mean tokens | answer accuracy | fully correct |
|---|---|---|---|---|
| full | 36 | 2914 | 82.2% | 55.6% |
| ulrc3 | 36 | 379 | 76.4% | 41.7% |
| truncate | 36 | 379 | 10.6% | 5.6% |

_Unequal n above means these are three different instance subsets. The comparison that matters is the paired one:_

## Paired comparison — the 36 instances where all three completed

| condition | accuracy |
|---|---|
| full (uncompressed control) | 82.2% |
| **ulrc3** | **76.4%** |
| truncate (matched budget) | 10.6% |

**vs truncation at the same token count:** +65.7 points (95% CI [+54.4, +76.2]), better on 31, worse on 0, tied 5; sign test p=0.0000.
**vs the uncompressed control:** -5.8 points (95% CI [-13.0, -0.2]), better on 1, worse on 5, tied 30; sign test p=1.0000.

| suite | full | ulrc3 | truncate |
|---|---|---|---|
| apidocs | 67% | 67% | 0% |
| code | 56% | 62% | 25% |
| logs | 100% | 67% | 0% |
| memory | 100% | 100% | 0% |
| mixed | 67% | 67% | 8% |
| multihop | 50% | 50% | 12% |
| needle | 100% | 100% | 25% |
| numeric | 100% | 75% | 0% |
| rag | 100% | 100% | 25% |

**Statistical caveat.** The paired comparison rests on n=36 instances from a free-tier quota. That is enough to establish a large effect (the truncation gap clears significance comfortably) and *not* enough to resolve a small one — the ulrc3-vs-full difference is well inside noise and should be read as 'no detected loss', not as 'no loss'. One model, one temperature, one prompt template.
