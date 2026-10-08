"""LM Studio model performance benchmark.

Measures how fast the local models serve a request — time to first token and
tokens/second — and nothing else. Correctness of the answers is the eval's job
(tests/eval/). See tests/perf/README.md.
"""
