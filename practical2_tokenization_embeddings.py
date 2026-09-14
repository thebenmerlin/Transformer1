#!/usr/bin/env python3
"""Generative AI Practical 2: tokenization, embeddings, and similarity.

The corresponding GitHub Actions workflow downloads all Hugging Face assets
inside an ephemeral Ubuntu runner.  This program does not substitute models.
"""

import csv
import platform
import sys

import numpy as np
import sklearn
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, GPT2Tokenizer


STUDENT_ID = "RBT23AR001"
BPE_MODEL = "gpt2"
WORDPIECE_MODEL = "bert-base-uncased"
SENTENCE_MODEL = "all-MiniLM-L6-v2"
RESULTS_FILE = "practical2_results.csv"
EMBEDDINGS_SUMMARY_FILE = "practical2_embeddings_summary.csv"

TOKENIZATION_SENTENCES = [
    ("Sentence 1", "Machine learning is very useful."),
    (
        "Sentence 2",
        "Generative artificial intelligence is transforming cybersecurity.",
    ),
]

SIMILARITY_PAIRS = [
    (
        "Pair 1",
        "Official",
        "I love machine learning.",
        "I enjoy studying artificial intelligence.",
    ),
    (
        "Pair 2",
        "High similarity",
        "The weather is very hot today.",
        "Today is an extremely warm day.",
    ),
    (
        "Pair 3",
        "Low similarity",
        "Machine learning is used to analyze data.",
        "I bought vegetables from the market.",
    ),
    (
        "Pair 4",
        "Paraphrase",
        "The student completed the assignment before the deadline.",
        "The assignment was finished by the student ahead of time.",
    ),
    (
        "Pair 5",
        "Related but not identical",
        "Python is widely used for machine learning.",
        "Java is commonly used for enterprise software.",
    ),
]

MATRIX_SENTENCES = [
    "Artificial intelligence is changing the technology industry.",
    "AI is transforming the world of technology.",
    "Deep learning is a branch of machine learning.",
    "The football team won the championship.",
    "I prepared pasta for dinner.",
]


def package_versions():
    print("\n=== Environment and Package Versions ===")
    print(f"Operating system: {platform.platform()}")
    print(f"CPU architecture: {platform.machine()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"transformers version: {transformers.__version__}")
    print(f"sentence-transformers version: {sentence_transformers.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"torch version: {torch.__version__}")


def token_details(tokenizer, text):
    """Obtain aligned token strings and real vocabulary IDs from a tokenizer."""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids


def print_tokenization_section(section_name, tokenizer_name, tokenizer, label):
    print(f"\n=== {section_name}: {tokenizer_name} ===")
    details = []
    for sentence_id, sentence in TOKENIZATION_SENTENCES:
        tokens, token_ids = token_details(tokenizer, sentence)
        print(f"\n{sentence_id}")
        print(f"Original Text: {sentence}")
        print(f"{label} Tokens: {tokens}")
        print(f"{label} Token IDs: {token_ids}")
        print(f"Number of {label} Tokens: {len(tokens)}")
        details.append(
            {
                "sentence_id": sentence_id,
                "text": sentence,
                "tokens": tokens,
                "token_ids": token_ids,
                "count": len(tokens),
            }
        )
    return details


def print_tokenization_comparison(bpe_details, wordpiece_details):
    print("\n=== Part C — BPE and WordPiece Tokenization Comparison ===")
    print(
        "Sentence | BPE token count | WordPiece token count | BPE tokens | WordPiece tokens"
    )
    print("-" * 130)
    for bpe, wordpiece in zip(bpe_details, wordpiece_details, strict=True):
        print(
            f"{bpe['sentence_id']} | {bpe['count']} | {wordpiece['count']} | "
            f"{bpe['tokens']} | {wordpiece['tokens']}"
        )


def embedding_summary(model):
    sentence = "Machine learning is a part of artificial intelligence."
    embedding = model.encode(sentence, convert_to_numpy=True)
    norm = float(np.linalg.norm(embedding))
    print("\n=== Part D — Sentence Embedding ===")
    print(f"Sentence: {sentence}")
    print(f"Embedding Dimension: {embedding.size}")
    print("First 10 Embedding Values: " + str(np.round(embedding[:10], 6).tolist()))
    print(f"L2 Norm: {norm:.6f}")
    return sentence, embedding, norm


def calculate_similarity_pairs(model):
    print("\n=== Parts E and F — Semantic Similarity ===")
    results = []
    for pair_id, category, sentence_1, sentence_2 in SIMILARITY_PAIRS:
        embeddings = model.encode([sentence_1, sentence_2], convert_to_numpy=True)
        score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        print(f"\n{pair_id} — {category}")
        print(f"Sentence 1: {sentence_1}")
        print(f"Sentence 2: {sentence_2}")
        print(f"Cosine Similarity: {score:.4f}")
        results.append(
            {
                "pair_id": pair_id,
                "category": category,
                "sentence_1": sentence_1,
                "sentence_2": sentence_2,
                "cosine_similarity": score,
            }
        )
    return results


def similarity_matrix(model):
    embeddings = model.encode(MATRIX_SENTENCES, convert_to_numpy=True)
    matrix = cosine_similarity(embeddings)
    labels = [f"S{number}" for number in range(1, len(MATRIX_SENTENCES) + 1)]
    print("\n=== Part G — 5 x 5 Cosine Similarity Matrix ===")
    for label, sentence in zip(labels, MATRIX_SENTENCES, strict=True):
        print(f"{label}: {sentence}")
    print("\n      " + "  ".join(f"{label:>7}" for label in labels))
    for label, row in zip(labels, matrix, strict=True):
        print(f"{label:>4}  " + "  ".join(f"{value:7.4f}" for value in row))
    return embeddings, matrix, labels


def write_results_csv(bpe_details, wordpiece_details, pair_results):
    fieldnames = [
        "record_type",
        "test_id",
        "category",
        "text_1",
        "text_2",
        "bpe_token_count",
        "wordpiece_token_count",
        "cosine_similarity",
    ]
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for bpe, wordpiece in zip(bpe_details, wordpiece_details, strict=True):
            writer.writerow(
                {
                    "record_type": "tokenization_comparison",
                    "test_id": bpe["sentence_id"],
                    "category": "BPE vs WordPiece",
                    "text_1": bpe["text"],
                    "text_2": "",
                    "bpe_token_count": bpe["count"],
                    "wordpiece_token_count": wordpiece["count"],
                    "cosine_similarity": "",
                }
            )
        for pair in pair_results:
            writer.writerow(
                {
                    "record_type": "semantic_similarity",
                    "test_id": pair["pair_id"],
                    "category": pair["category"],
                    "text_1": pair["sentence_1"],
                    "text_2": pair["sentence_2"],
                    "bpe_token_count": "",
                    "wordpiece_token_count": "",
                    "cosine_similarity": pair["cosine_similarity"],
                }
            )


def write_embeddings_summary(embedding_sentence, embedding, norm, matrix, labels):
    fieldnames = [
        "record_type",
        "record_id",
        "sentence",
        "embedding_dimension",
        "l2_norm",
        "first_10_embedding_values",
        *[f"similarity_{label}" for label in labels],
    ]
    with open(EMBEDDINGS_SUMMARY_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "record_type": "embedding_demo",
                "record_id": "Embedding 1",
                "sentence": embedding_sentence,
                "embedding_dimension": embedding.size,
                "l2_norm": norm,
                "first_10_embedding_values": " ".join(
                    f"{value:.8f}" for value in embedding[:10]
                ),
            }
        )
        for label, sentence, row in zip(labels, MATRIX_SENTENCES, matrix, strict=True):
            writer.writerow(
                {
                    "record_type": "similarity_matrix",
                    "record_id": label,
                    "sentence": sentence,
                    "embedding_dimension": "",
                    "l2_norm": "",
                    "first_10_embedding_values": "",
                    **{
                        f"similarity_{column_label}": value
                        for column_label, value in zip(labels, row, strict=True)
                    },
                }
            )


def final_summary(pair_results):
    print("\n=== Part H — Consolidated Semantic Similarity Summary ===")
    print("Pair | Category | Similarity Score")
    print("-" * 72)
    for pair in pair_results:
        print(f"{pair['pair_id']} | {pair['category']} | {pair['cosine_similarity']:.4f}")
    highest = max(pair_results, key=lambda item: item["cosine_similarity"])
    lowest = min(pair_results, key=lambda item: item["cosine_similarity"])
    print(
        f"Highest Similarity Pair: {highest['pair_id']} — {highest['category']} "
        f"({highest['cosine_similarity']:.4f})"
    )
    print(
        f"Lowest Similarity Pair: {lowest['pair_id']} — {lowest['category']} "
        f"({lowest['cosine_similarity']:.4f})"
    )


def main():
    print(f"{STUDENT_ID} - Generative AI Practical 2")
    print("Tokenization, Sentence Embeddings and Semantic Similarity")
    package_versions()

    print(f"\nLoading BPE tokenizer: {BPE_MODEL}")
    bpe_tokenizer = GPT2Tokenizer.from_pretrained(BPE_MODEL)
    bpe_details = print_tokenization_section(
        "Part A — BPE Tokenization", "GPT-2 BPE (gpt2)", bpe_tokenizer, "BPE"
    )

    print(f"\nLoading WordPiece tokenizer: {WORDPIECE_MODEL}")
    wordpiece_tokenizer = BertTokenizer.from_pretrained(WORDPIECE_MODEL)
    wordpiece_details = print_tokenization_section(
        "Part B — WordPiece Tokenization",
        "BERT WordPiece (bert-base-uncased)",
        wordpiece_tokenizer,
        "WordPiece",
    )
    print_tokenization_comparison(bpe_details, wordpiece_details)

    print(f"\nLoading sentence-transformer model: {SENTENCE_MODEL}")
    sentence_model = SentenceTransformer(SENTENCE_MODEL)
    embedding_sentence, embedding, norm = embedding_summary(sentence_model)
    pair_results = calculate_similarity_pairs(sentence_model)
    _, matrix, labels = similarity_matrix(sentence_model)

    write_results_csv(bpe_details, wordpiece_details, pair_results)
    write_embeddings_summary(embedding_sentence, embedding, norm, matrix, labels)
    final_summary(pair_results)
    print(f"\nStructured results saved to: {RESULTS_FILE}")
    print(f"Embedding and matrix summary saved to: {EMBEDDINGS_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
