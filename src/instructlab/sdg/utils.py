# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Union
import io
import json
import os
import re

# Third Party
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_OVERLAP = 100

# When otherwise unknown, ilab uses this as the default family
DEFAULT_MODEL_FAMILY = "merlinite"

# Model families understood by ilab
MODEL_FAMILIES = set(("merlinite", "mixtral"))

# Map model names to their family
MODEL_FAMILY_MAPPINGS = {
    "granite": "merlinite",
}


class GenerateException(Exception):
    """An exception raised during generate step."""


def _make_w_io_base(f, mode: str):
    # pylint: disable=consider-using-with
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="utf-8")
    return f


def _make_r_io_base(f, mode: str):
    # pylint: disable=consider-using-with
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="utf-8")
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    with _make_w_io_base(f, mode) as f_:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f_, indent=indent, default=default)
        elif isinstance(obj, str):
            f_.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    with _make_r_io_base(f, mode) as f_:
        return json.load(f_)


def num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def num_tokens_from_chars(num_chars) -> int:
    return int(num_chars / 4)  # 1 token ~ 4 English character


def max_seed_example_tokens(server_ctx_size, prompt_num_chars) -> int:
    """
    Estimates the maximum number of tokens any seed example can have based
    on the server context size and number of characters in the selected prompt.

    A lot has to fit into the given server context size:
      - The prompt itself, which can vary in size a bit based on model family and knowledge vs skill
      - Two seed examples, which we append to the prompt template.
      - A knowledge document chunk, if this is a knowledge example.
      - The generated completion, which can vary substantially in length.

    This is an attempt to roughly estimate the maximum size any seed example
    (question + answer + context values from the yaml) should be to even have
    a hope of not often exceeding the server's maximum context size.

    NOTE: This does not take into account knowledge document chunks. It's meant
    to calculate the maximum size that any seed example should be, whether knowledge
    or skill. Knowledge seed examples will want to stay well below this limit.

    NOTE: This is a very simplistic calculation, and examples with lots of numbers
    or punctuation may have quite a different token count than the estimates here,
    depending on the model (and thus tokenizer) in use. That's ok, as it's only
    meant to be a rough estimate.

    Args:
        server_ctx_size (int): Size of the server context, in tokens.
        prompt_num_chars (int): Number of characters in the prompt (not including the examples)
    """
    # Ensure we have at least 1024 tokens available for a response.
    max_seed_tokens = server_ctx_size - 1024
    # Subtract the number of tokens in our prompt template
    max_seed_tokens = max_seed_tokens - num_tokens_from_chars(prompt_num_chars)
    # Divide number of characters by 2, since we insert 2 examples
    max_seed_tokens = int(max_seed_tokens / 2)
    return max_seed_tokens


def chunk_document(documents: List, server_ctx_size, chunk_word_count) -> List[str]:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (dict): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """
    no_tokens_per_doc = num_tokens_from_words(chunk_word_count)
    if no_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )
    content = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=num_chars_from_tokens(no_tokens_per_doc),
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    for docs in documents:
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])

    return content


def get_model_family(forced, model_path):
    forced = MODEL_FAMILY_MAPPINGS.get(forced, forced)
    if forced and forced.lower() not in MODEL_FAMILIES:
        raise GenerateException("Unknown model family: %s" % forced)

    # Try to guess the model family based on the model's filename
    guess = re.match(r"^\w*", os.path.basename(model_path)).group(0).lower()
    guess = MODEL_FAMILY_MAPPINGS.get(guess, guess)

    return guess if guess in MODEL_FAMILIES else DEFAULT_MODEL_FAMILY
