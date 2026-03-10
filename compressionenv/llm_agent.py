# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM-based compression agent using Qwen 8B for data-dependent algorithm generation."""

import re
from typing import Any, Dict, Union

from .models import CompressionenvAction

# Lazy-loaded model and tokenizer
_tokenizer = None
_model = None
_MODEL_NAME = "Qwen/Qwen3-8B"


def _get_model():
    """Lazy load transformers model and tokenizer."""
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
    return _tokenizer, _model


def llm_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Wrapper around LLM call. Returns generated text."""
    tokenizer, model = _get_model()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(
        generated_ids[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True
    )


def _parse_code_blocks(response: str) -> tuple[str, str]:
    """Extract compress and decompress code from LLM response."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if len(blocks) >= 2:
        return blocks[0].strip(), blocks[1].strip()
    if len(blocks) == 1:
        code = blocks[0].strip()
        if "def decompress" in code:
            idx = code.index("def decompress")
            return code[:idx].strip(), code[idx:].strip()
        return code, "def decompress(data: bytes) -> str:\n    return data.decode('utf-8')"
    cc_match = re.search(
        r"(def compress\s*\([^)]*\)[^:]*:.*?)(?=def decompress)", response, re.DOTALL
    )
    dc_match = re.search(
        r"(def decompress\s*\([^)]*\)[^:]*:.*)", response, re.DOTALL
    )
    cc = (
        cc_match.group(1).strip()
        if cc_match
        else "def compress(text: str) -> bytes:\n    return text.encode('utf-8')"
    )
    dc = (
        dc_match.group(1).strip()
        if dc_match
        else "def decompress(data: bytes) -> str:\n    return data.decode('utf-8')"
    )
    return cc, dc


def _get_obs_field(obs: Union[Dict[str, Any], Any], key: str, default=None):
    """Get field from dict or object."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def generate_compression_action(
    prev_obs: Union[Dict[str, Any], Any], step: int, essay_text: str = ""
) -> CompressionenvAction:
    """
    Prompt LLM to generate/improve compression based on previous step result.

    Args:
        prev_obs: Previous observation (dict or object) with baselines_size_bytes, etc.
        step: Step number (1 = first action)
        essay_text: The data to compress (for data-dependent algorithms)

    Returns:
        CompressionenvAction with generated compression_code and decompression_code
    """
    bl = _get_obs_field(prev_obs, "baselines_size_bytes") or {}
    best_bl = _get_obs_field(prev_obs, "best_baseline_size_bytes")
    csz = _get_obs_field(prev_obs, "compressed_size_bytes")
    err = _get_obs_field(prev_obs, "error")
    valid = _get_obs_field(prev_obs, "valid")
    avg_prev = _get_obs_field(prev_obs, "avg_prev_compressed_size_bytes")

    essay_sample = (
        (essay_text[:16000] + "\n...[truncated]")
        if len(essay_text) > 16000
        else essay_text
    )

    forbidden = (
        "You MUST NOT use zlib, bz2, lzma, or gzip - only custom algorithms. "
        "Reward is 0 if you use them."
    )

    if step == 1:
        prompt = f"""Read the DATA below and design a CUSTOM compression algorithm tailored to this specific text.
Analyze patterns, repeated phrases, character frequencies, or structure. Your algorithm must depend on the data.

DATA to compress:
---
{essay_sample}
---

You must define exactly two functions:
1. compress(text: str) -> bytes  - takes a string, returns compressed bytes
2. decompress(data: bytes) -> str - takes bytes, returns original string (round-trip must match).

{forbidden}
Target sizes to beat: {bl}. Best: {best_bl} bytes.

Output ONLY two code blocks, no explanation:
```python
def compress(text: str) -> bytes:
    ...

def decompress(data: bytes) -> str:
    ...
```"""
    else:
        status = (
            f"Last attempt: valid={valid}, compressed_size={csz} bytes"
            if valid
            else f"Last attempt FAILED: {err}"
        )
        hint = f" (improve to get < {avg_prev:.0f} avg)" if avg_prev else ""
        prompt = f"""Improve your data-dependent compression. {status}
Baselines to beat: {bl}. Best: {best_bl} bytes.{hint}

DATA (same as before):
---
{essay_sample}
---

{forbidden}
Output improved Python code - ONLY two functions, no explanation:
```python
def compress(text: str) -> bytes:
    ...

def decompress(data: bytes) -> str:
    ...
```"""

    response = llm_generate(prompt, max_new_tokens=1024)
    cc, dc = _parse_code_blocks(response)
    return CompressionenvAction(
        compression_code=cc, decompression_code=dc, algo_name=f"llm_step{step}"
    )
