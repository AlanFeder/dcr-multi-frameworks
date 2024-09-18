import logging
import os

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)
load_dotenv()


def make_oai_client() -> OpenAI:
    """
    Make an OpenAI client.

    Returns:
        OpenAI: An OpenAI client.
    """
    oai_client = OpenAI(os.getenv("OPENAI_API_KEY"))

    return oai_client


def do_1_embed(lt: str, emb_client: OpenAI) -> np.ndarray:
    """
    Generate embeddings using the OpenAI API for a single text.

    Args:
        lt (str): A text to generate embeddings for.
        emb_client (OpenAI): The embedding API client (OpenAI).

    Returns:
        np.ndarray: The generated embeddings.
    """
    if isinstance(emb_client, OpenAI):
        # Generate embeddings using OpenAI API
        embed_response = emb_client.embeddings.create(
            input=lt,
            model="text-embedding-3-small",
        )
        here_embed = np.array(embed_response.data[0].embedding)
    else:
        logger.error("There is some problem with the embedding client")
        raise Exception("There is some problem with the embedding client")

    logger.info(f"Embedded {lt}")
    return here_embed


def calc_n_tokens(text_in: str) -> int:
    """
    Calculate the number of tokens in the input text using the 'o200k_base' encoding.

    Args:
        text_in (str): The input text.

    Returns:
        int: The number of tokens in the input text.
    """
    tok_model = tiktoken.get_encoding("o200k_base")
    token_ids = tok_model.encode(text=text_in)
    n_tokens = len(token_ids)

    logger.info(f"{n_tokens} counted")

    return n_tokens


def calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost in cents based on the number of prompt, completion, and embedding tokens.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.

    Returns:
        float: The cost in cents.
    """
    prompt_cost = prompt_tokens / 2000
    completion_cost = 3 * completion_tokens / 2000

    cost_cents = prompt_cost + completion_cost

    logger.info(
        f"Costs: Prompt: {prompt_cost}. Completion: {completion_cost}. Total Cost: {cost_cents}"
    )

    return cost_cents
