#!/usr/bin/env python3
"""Benchmark llama3:8b served by a locally running Ollama instance.

This script is intended to run on the GitHub Actions Ubuntu runner configured
by .github/workflows/practical1.yml.  It never pulls or substitutes a model.
"""

import csv
import platform
import shutil
import subprocess
import sys
import time

import ollama
import psutil


STUDENT_ID = "RBT23AR001"
MODEL = "llama3:8b"
RESULTS_FILE = "practical1_results.csv"
NANOSECONDS_PER_SECOND = 1_000_000_000
BYTES_PER_GIB = 1024 ** 3

# Deterministic settings and bounded answers for a practical-length run.
GENERATION_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 256,
}

TESTS = [
    (
        "TEST 1",
        "Short",
        "What is Generative AI? Answer in two sentences.",
    ),
    (
        "TEST 2",
        "Programming",
        "Write a Python program to calculate the factorial of a number and explain the code step by step.",
    ),
    (
        "TEST 3",
        "Explanation",
        "Explain how a transformer-based large language model processes a user prompt and generates a response. Discuss tokenization, embeddings, attention, and autoregressive generation in a clear structured explanation.",
    ),
]

CSV_FIELDS = [
    "test",
    "prompt_type",
    "execution_time_s",
    "generated_tokens",
    "generation_time_s",
    "tokens_per_second",
    "prompt_tokens",
    "prompt_evaluation_time_s",
    "total_duration_s",
    "load_duration_s",
    "ram_before_percent",
    "ram_after_percent",
    "ram_used_before_gb",
    "ram_used_after_gb",
    "ram_available_before_gb",
    "ram_available_after_gb",
]


def command_output(command):
    """Return a command's combined output without making diagnostics fatal."""
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.stdout.strip() or "(no output)"
    except FileNotFoundError:
        return "(command not available)"


def response_value(response, name):
    """Read a metadata field from either Ollama's object or dict response."""
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


def response_text(response):
    """Read assistant content from either Ollama response representation."""
    message = response_value(response, "message")
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def seconds_from_nanoseconds(value):
    if value is None:
        return None
    return value / NANOSECONDS_PER_SECOND


def as_float(value):
    return None if value is None else float(value)


def display(value, decimals=3):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def print_environment():
    memory = psutil.virtual_memory()
    print("\n=== Environment Information ===")
    print(f"Operating system: {platform.platform()}")
    print(f"CPU architecture: {platform.machine()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Total RAM: {memory.total / BYTES_PER_GIB:.2f} GB")
    print(f"Available RAM: {memory.available / BYTES_PER_GIB:.2f} GB")
    print("Ollama version:")
    print(command_output(["ollama", "--version"]))
    print("Ollama list:")
    print(command_output(["ollama", "list"]))
    print("Ollama ps:")
    print(command_output(["ollama", "ps"]))

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("GPU acceleration: Not available on this GitHub-hosted runner.")
    else:
        gpu_info = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if gpu_info.returncode == 0 and gpu_info.stdout.strip():
            print("NVIDIA GPU information:")
            print(gpu_info.stdout.strip())
        else:
            print("GPU acceleration: Not available on this GitHub-hosted runner.")


def benchmark(test_name, prompt_type, prompt):
    memory_before = psutil.virtual_memory()
    started = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=GENERATION_OPTIONS,
    )
    wall_clock_seconds = time.perf_counter() - started
    memory_after = psutil.virtual_memory()

    eval_count = response_value(response, "eval_count")
    eval_duration_ns = response_value(response, "eval_duration")
    prompt_eval_count = response_value(response, "prompt_eval_count")
    prompt_eval_duration_ns = response_value(response, "prompt_eval_duration")
    total_duration_ns = response_value(response, "total_duration")
    load_duration_ns = response_value(response, "load_duration")

    generation_seconds = seconds_from_nanoseconds(eval_duration_ns)
    tokens_per_second = None
    if eval_count is not None and generation_seconds is not None and generation_seconds > 0:
        tokens_per_second = eval_count / generation_seconds

    result = {
        "test": test_name,
        "prompt_type": prompt_type,
        "execution_time_s": wall_clock_seconds,
        "generated_tokens": eval_count,
        "generation_time_s": generation_seconds,
        "tokens_per_second": tokens_per_second,
        "prompt_tokens": prompt_eval_count,
        "prompt_evaluation_time_s": seconds_from_nanoseconds(prompt_eval_duration_ns),
        "total_duration_s": seconds_from_nanoseconds(total_duration_ns),
        "load_duration_s": seconds_from_nanoseconds(load_duration_ns),
        "ram_before_percent": memory_before.percent,
        "ram_after_percent": memory_after.percent,
        "ram_used_before_gb": memory_before.used / BYTES_PER_GIB,
        "ram_used_after_gb": memory_after.used / BYTES_PER_GIB,
        "ram_available_before_gb": memory_before.available / BYTES_PER_GIB,
        "ram_available_after_gb": memory_after.available / BYTES_PER_GIB,
    }

    print(f"\n{'=' * 72}\n{test_name} — {prompt_type}\n{'=' * 72}")
    print(f"Prompt: {prompt}")
    print("\nGenerated response:")
    print(response_text(response))
    print("\nBenchmark summary:")
    print(f"Wall-clock execution time: {display(wall_clock_seconds)} s")
    print(f"Generated tokens (eval_count): {display(eval_count)}")
    print(f"Generation time (eval_duration): {display(generation_seconds)} s")
    print(f"Generation throughput: {display(tokens_per_second)} tokens/s")
    print(f"Prompt tokens (prompt_eval_count): {display(prompt_eval_count)}")
    print(
        "Prompt evaluation time (prompt_eval_duration): "
        f"{display(result['prompt_evaluation_time_s'])} s"
    )
    print(f"Total duration: {display(result['total_duration_s'])} s")
    print(f"Load duration: {display(result['load_duration_s'])} s")
    print(f"RAM before: {memory_before.percent:.1f}% ({result['ram_used_before_gb']:.2f} GB used)")
    print(f"RAM after: {memory_after.percent:.1f}% ({result['ram_used_after_gb']:.2f} GB used)")
    print(f"Available RAM before/after: {result['ram_available_before_gb']:.2f} / {result['ram_available_after_gb']:.2f} GB")
    return result


def write_csv(results):
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def print_results_table(results):
    columns = [
        ("Test", "test", 8),
        ("Prompt Type", "prompt_type", 14),
        ("Execution Time (s)", "execution_time_s", 18),
        ("Generated Tokens", "generated_tokens", 18),
        ("Generation Time (s)", "generation_time_s", 19),
        ("Tokens/sec", "tokens_per_second", 13),
        ("Prompt Tokens", "prompt_tokens", 14),
        ("RAM Before (%)", "ram_before_percent", 15),
        ("RAM After (%)", "ram_after_percent", 14),
        ("RAM Used Before (GB)", "ram_used_before_gb", 21),
        ("RAM Used After (GB)", "ram_used_after_gb", 20),
        ("Load Duration (s)", "load_duration_s", 18),
    ]
    print("\n=== Consolidated Benchmark Results ===")
    print(" | ".join(title.ljust(width) for title, _, width in columns))
    print("-|-".join("-" * width for _, _, width in columns))
    for result in results:
        row = []
        for _, key, width in columns:
            value = result[key]
            rendered = value if isinstance(value, str) else display(value)
            row.append(rendered.ljust(width))
        print(" | ".join(row))


def main():
    print(f"{STUDENT_ID} - Generative AI Practical 1")
    print(f"Model: {MODEL}")
    print(f"Generation options: {GENERATION_OPTIONS}")
    print_environment()

    results = []
    try:
        for test_name, prompt_type, prompt in TESTS:
            results.append(benchmark(test_name, prompt_type, prompt))
    finally:
        write_csv(results)

    print_results_table(results)
    print(f"\nCSV measurements saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
