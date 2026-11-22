#!/usr/bin/env python3
"""Tokenize a text file with the official Mistral-Large-4 tokenizer (HF format).

Usage: .venv/bin/python tools/py/tokenize.py tokenizer.json < text.txt > tokens.txt

Writes whitespace-separated token ids (one line per input line's tokens).
HF tokenizer.json ids match the GGUF tekken ids for Mistral models.
"""
import sys
from tokenizers import Tokenizer

def main():
    tk = Tokenizer.from_file(sys.argv[1])
    text = sys.stdin.read()
    ids = tk.encode(text, add_special_tokens=False).ids
    print(" ".join(map(str, ids)))

if __name__ == "__main__":
    main()
