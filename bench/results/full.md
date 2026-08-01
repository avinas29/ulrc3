# ULRC3 benchmark report

## apidocs

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  52.2 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 47.9 |
| ulrc3:balanced |  84.5 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 46.4 |
| ulrc3:aggressive |  85.2 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 45.0 |
| ulrc3:extreme |  86.0 |  73.8 |   0.0 |  85.7 | 100.0 | 100.0 | 0.00 | 43.8 |
| truncate_head |  83.4 |  42.9 |   0.0 |  62.2 |  71.4 | 100.0 | 0.14 | 0.6 |
| truncate_tail |  83.4 |  52.4 |   0.0 |  65.3 |  85.7 | 100.0 | 0.00 | 1.4 |
| truncate_middle |  83.3 |  42.9 |   0.0 |  60.2 |  71.4 | 100.0 | 0.00 | 1.7 |
| random_sentences |  83.5 |  47.6 |   0.0 |  60.2 |  78.6 | 100.0 | 0.00 | 0.7 |
| tfidf_mmr |  83.6 |  66.7 |   0.0 |  57.1 | 100.0 | 100.0 | 0.00 | 4.9 |
| selective_context |  83.5 | 100.0 |   0.0 |  57.1 | 100.0 | 100.0 | 0.00 | 1.3 |
| llmlingua_style |  83.4 |  90.5 |   0.0 |  81.6 | 100.0 | 100.0 | 0.00 | 2.2 |

## code

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  43.2 | 100.0 |   0.0 | 100.0 |  98.4 | 100.0 | 0.00 | 40.0 |
| ulrc3:balanced |  60.5 | 100.0 |   0.0 | 100.0 |  94.2 | 100.0 | 0.00 | 36.2 |
| ulrc3:aggressive |  65.2 | 100.0 |   0.0 | 100.0 |  76.9 | 100.0 | 0.00 | 35.8 |
| ulrc3:extreme |  65.5 | 100.0 |   0.0 | 100.0 |  76.9 | 100.0 | 0.00 | 35.4 |
| truncate_head |  59.8 |  53.1 |   0.0 |  64.2 |  69.5 | 100.0 | 0.00 | 0.7 |
| truncate_tail |  59.9 |  78.1 |   0.0 |  68.2 |  78.5 | 100.0 | 0.00 | 2.3 |
| truncate_middle |  59.8 |  65.6 |   0.0 |  56.9 |  80.3 | 100.0 | 0.12 | 2.1 |
| random_sentences |  60.3 |  46.9 |   0.0 |  49.0 |  71.6 | 100.0 | 0.00 | 0.5 |
| tfidf_mmr |  61.7 | 100.0 |   0.0 |  56.2 |  80.7 | 100.0 | 0.00 | 1.5 |
| selective_context |  60.2 | 100.0 |   0.0 |  51.5 |  60.7 | 100.0 | 0.00 | 0.6 |
| llmlingua_style |  59.8 |  75.0 |   0.0 |  91.1 |  87.7 | 100.0 | 0.00 | 1.6 |

## logs

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  98.7 | 100.0 |   0.0 |  19.7 | 100.0 | 100.0 | 0.00 | 12.4 |
| ulrc3:balanced |  98.7 | 100.0 |   0.0 |  19.7 | 100.0 | 100.0 | 0.00 | 12.2 |
| ulrc3:aggressive |  98.7 | 100.0 |   0.0 |  19.7 | 100.0 | 100.0 | 0.00 | 12.0 |
| ulrc3:extreme |  98.7 | 100.0 |   0.0 |  19.7 | 100.0 | 100.0 | 0.00 | 12.3 |
| truncate_head |  98.6 |   0.0 |   0.0 |  22.6 |  33.3 | 100.0 | 0.00 | 4.0 |
| truncate_tail |  98.6 |   0.0 |   0.0 |  23.4 |  33.3 | 100.0 | 0.00 | 4.4 |
| truncate_middle |  98.6 |   0.0 |   0.0 |  23.3 |  33.3 | 100.0 | 1.00 | 8.2 |
| random_sentences |  99.9 |   0.0 |   0.0 |   0.0 |   0.0 | 100.0 | 0.00 | 4.4 |
| tfidf_mmr |  99.9 |   0.0 |   0.0 |   0.0 |   0.0 | 100.0 | 0.00 | 7.0 |
| selective_context |  99.9 |   0.0 |   0.0 |   0.0 |   0.0 | 100.0 | 0.00 | 5.6 |
| llmlingua_style |  99.9 |   0.0 |   0.0 |   0.0 |   0.0 | 100.0 | 0.00 | 8.2 |

## memory

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  75.4 | 100.0 |   0.0 |  66.7 | 100.0 | 100.0 | 0.00 | 34.7 |
| ulrc3:balanced |  77.9 | 100.0 |   0.0 |  66.7 | 100.0 | 100.0 | 0.00 | 34.3 |
| ulrc3:aggressive |  85.0 | 100.0 |   0.0 |  66.7 | 100.0 | 100.0 | 0.00 | 34.6 |
| ulrc3:extreme |  86.4 | 100.0 |   0.0 |  66.7 | 100.0 | 100.0 | 0.00 | 34.3 |
| truncate_head |  76.6 |   4.2 | 100.0 |  33.3 | 100.0 | 100.0 | 0.00 | 0.5 |
| truncate_tail |  76.6 |  33.3 |   0.0 |  33.3 | 100.0 | 100.0 | 0.00 | 1.2 |
| truncate_middle |  76.5 |   2.1 |  37.5 |  12.5 | 100.0 | 100.0 | 0.00 | 1.3 |
| random_sentences |  77.1 |  64.6 |   0.0 |  31.2 | 100.0 | 100.0 | 0.00 | 0.5 |
| tfidf_mmr |  77.3 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.00 | 1.4 |
| selective_context |  77.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.00 | 0.8 |
| llmlingua_style |  76.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.00 | 1.6 |

## mixed

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  46.3 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 7.6 |
| ulrc3:balanced |  53.6 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 6.8 |
| ulrc3:aggressive |  53.6 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 6.8 |
| ulrc3:extreme |  53.6 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 6.8 |
| truncate_head |  50.2 |  63.9 |   0.0 |  29.6 |  20.8 | 100.0 | 0.17 | 0.1 |
| truncate_tail |  51.4 |  27.8 |   0.0 |  81.5 |  66.7 | 100.0 | 0.00 | 0.4 |
| truncate_middle |  49.7 |   0.0 |   0.0 |  16.7 |  12.5 | 100.0 | 0.00 | 0.3 |
| random_sentences |  52.0 |  38.9 |   0.0 |  18.5 |   8.3 | 100.0 | 0.00 | 0.1 |
| tfidf_mmr |  51.5 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 0.3 |
| selective_context |  51.7 |  72.2 |   0.0 |  63.0 |  58.3 | 100.0 | 0.00 | 0.2 |
| llmlingua_style |  50.2 |  58.3 |   0.0 |  86.1 |  29.2 | 100.0 | 0.00 | 0.4 |

## multihop

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  58.7 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 23.0 |
| ulrc3:balanced |  61.3 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 22.4 |
| ulrc3:aggressive |  65.5 |  97.5 |   0.0 |  99.5 | 100.0 | 100.0 | 0.00 | 21.4 |
| ulrc3:extreme |  66.1 |  97.5 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 21.7 |
| truncate_head |  59.8 |  40.0 |   0.0 |  55.4 | 100.0 | 100.0 | 0.15 | 0.3 |
| truncate_tail |  59.8 |  55.0 |   0.0 |  50.1 | 100.0 | 100.0 | 0.00 | 1.1 |
| truncate_middle |  59.6 |  52.5 |   0.0 |  52.8 | 100.0 | 100.0 | 0.00 | 1.0 |
| random_sentences |  60.0 |  45.0 |   0.0 |  57.7 | 100.0 | 100.0 | 0.00 | 0.4 |
| tfidf_mmr |  60.2 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 2.5 |
| selective_context |  60.1 | 100.0 |   0.0 |  63.9 | 100.0 | 100.0 | 0.00 | 0.7 |
| llmlingua_style |  59.7 |  50.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 1.3 |

## needle

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  78.7 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 115.5 |
| ulrc3:balanced |  78.7 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 117.4 |
| ulrc3:aggressive |  88.0 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 114.3 |
| ulrc3:extreme |  97.1 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 111.7 |
| truncate_head |  78.4 |  25.0 |   0.0 |  25.0 | 100.0 | 100.0 | 0.04 | 1.5 |
| truncate_tail |  78.4 |  20.8 |   0.0 |  20.8 | 100.0 | 100.0 | 0.00 | 4.2 |
| truncate_middle |  78.4 |  20.8 |   0.0 |  20.8 | 100.0 | 100.0 | 0.00 | 4.3 |
| random_sentences |  78.5 |  20.8 |   0.0 |  20.8 | 100.0 | 100.0 | 0.00 | 1.6 |
| tfidf_mmr |  78.6 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 22.8 |
| selective_context |  78.5 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 3.2 |
| llmlingua_style |  78.4 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 5.9 |

## numeric

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  59.0 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 43.4 |
| ulrc3:balanced |  88.9 | 100.0 |   0.0 |  98.8 | 100.0 | 100.0 | 0.00 | 42.1 |
| ulrc3:aggressive |  91.4 | 100.0 |   0.0 |  97.5 | 100.0 | 100.0 | 0.00 | 41.1 |
| ulrc3:extreme |  91.8 | 100.0 |   0.0 |  96.2 | 100.0 | 100.0 | 0.00 | 40.4 |
| truncate_head |  88.3 |  15.0 |   0.0 |  13.8 | 100.0 | 100.0 | 0.05 | 0.5 |
| truncate_tail |  88.3 |  17.5 |   0.0 |  17.5 | 100.0 | 100.0 | 0.00 | 1.0 |
| truncate_middle |  88.2 |  25.0 |   0.0 |  23.8 | 100.0 | 100.0 | 0.00 | 1.3 |
| random_sentences |  88.4 |  33.8 |   0.0 |  30.0 | 100.0 | 100.0 | 0.00 | 0.6 |
| tfidf_mmr |  88.6 |  80.0 |   0.0 |  80.0 | 100.0 | 100.0 | 0.00 | 3.1 |
| selective_context |  88.4 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 1.2 |
| llmlingua_style |  88.3 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 2.0 |

## rag

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | integrity% | halluc. | latency ms |
|---|---|---|---|---|---|---|---|---|
| ulrc3:conservative |  49.2 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 33.2 |
| ulrc3:balanced |  73.3 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 31.5 |
| ulrc3:aggressive |  75.1 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 30.8 |
| ulrc3:extreme |  75.5 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 31.1 |
| truncate_head |  72.2 |  55.0 |   0.0 |  71.0 | 100.0 | 100.0 | 0.00 | 0.4 |
| truncate_tail |  72.2 |  53.3 |   0.0 |  69.0 | 100.0 | 100.0 | 0.00 | 1.3 |
| truncate_middle |  72.0 |  58.3 |   0.0 |  72.0 | 100.0 | 100.0 | 0.05 | 1.3 |
| random_sentences |  72.3 |  65.0 |   0.0 |  75.0 | 100.0 | 100.0 | 0.00 | 0.5 |
| tfidf_mmr |  72.5 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 2.9 |
| selective_context |  72.2 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 0.9 |
| llmlingua_style |  72.1 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 | 0.00 | 1.7 |


## Cross-suite means

| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | halluc. |
|---|---|---|---|---|---|---|
| ulrc3:conservative |  62.4 | 100.0 |   0.0 |  87.4 |  99.8 | 0.00 |
| ulrc3:balanced |  75.3 | 100.0 |   0.0 |  87.2 |  99.4 | 0.00 |
| ulrc3:aggressive |  78.6 |  99.7 |   0.0 |  87.0 |  97.4 | 0.00 |
| ulrc3:extreme |  80.1 |  96.8 |   0.0 |  85.4 |  97.4 | 0.00 |
| truncate_head |  74.1 |  33.2 |  11.1 |  41.9 |  77.2 | 0.06 |
| truncate_tail |  74.3 |  37.6 |   0.0 |  47.7 |  84.9 | 0.00 |
| truncate_middle |  74.0 |  29.7 |   4.2 |  37.7 |  77.5 | 0.13 |
| random_sentences |  74.7 |  40.3 |   0.0 |  38.1 |  73.2 | 0.00 |
| tfidf_mmr |  74.9 |  83.0 |  11.1 |  77.0 |  86.7 | 0.00 |
| selective_context |  74.6 |  85.8 |  11.1 |  70.6 |  79.9 | 0.00 |
| llmlingua_style |  74.3 |  74.9 |  11.1 |  84.3 |  79.6 | 0.00 |
