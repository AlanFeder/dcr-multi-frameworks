import logging
from typing import Any

import numpy as np

from rag_utils.rag_utils import OpenAI, do_1_embed

logger = logging.getLogger(__name__)


def do_sort(
    embed_q: np.ndarray, embed_talks: np.ndarray, list_talk_ids: list[str]
) -> list[dict[str, str | float]]:
    """
    Sort documents based on their cosine similarity to the query embedding.

    Args:
        embed_dict (dict[str, np.ndarray]): Dictionary containing document embeddings.
        arr_q (np.ndarray): Query embedding.

    Returns:
        pd.DataFrame: Sorted dataframe containing document IDs and similarity scores.
    """

    # Calculate cosine similarities between query embedding and document embeddings
    cos_sims = np.dot(embed_talks, embed_q)

    # Get the indices of the best matching video IDs
    best_match_video_ids = np.argsort(-cos_sims)

    # Get the sorted video IDs based on the best match indices
    sorted_vids = [
        {"id0": list_talk_ids[i], "score": -cs}
        for i, cs in zip(best_match_video_ids, np.sort(-cos_sims))
    ]

    return sorted_vids


def limit_docs(
    sorted_vids: list[dict[str, str | float]], talk_info: dict[str, str | int], n_results: int
) -> list[dict[str, Any]]:
    """
    Limit the retrieved documents based on a score threshold and return the top documents.

    Args:
        df_sorted (pd.DataFrame): Sorted dataframe containing document IDs and similarity scores.
        df_talks (pd.DataFrame): Dataframe containing talk information.
        n_results (int): Number of top documents to retrieve.
        transcript_dicts (dict[str, dict]): Dictionary containing transcript text for each document ID.

    Returns:
        dict[str, dict]: Dictionary containing the top documents with their IDs, scores, and text.
    """

    # Get the top n_results documents
    top_vids = sorted_vids[:n_results]

    # Get the top score and calculate the score threshold
    top_score = top_vids[0]["score"]
    score_thresh = max(min(0.6, top_score - 0.05), 0.2)

    # Filter the top documents based on the score threshold
    keep_texts = []
    for my_vid in top_vids:
        if my_vid["score"] >= score_thresh:
            vid_data = talk_info[my_vid["id0"]]
            vid_data = {**vid_data, **my_vid}
            keep_texts.append(vid_data)

    logger.info(f"{len(keep_texts)} videos kept")

    return keep_texts


def do_retrieval(
    query0: str,
    n_results: int,
    api_client: OpenAI,
    talk_ids: list[str],
    embeds: np.ndarray,
    talk_info: dict[str, str | int],
) -> list[dict[str, Any]]:
    """
    Retrieve relevant documents based on the user's query.

    Args:
        query0 (str): The user's query.
        n_results (int): The number of documents to retrieve.
        api_client (OpenAI): The API client (OpenAI) for generating embeddings.

    Returns:
        dict[str, dict]: The retrieved documents.
    """
    logger.info(f"Starting document retrieval for query: {query0}")
    try:
        # Generate embeddings for the query
        arr_q = do_1_embed(query0, api_client)

        # Sort documents based on their cosine similarity to the query embedding
        sorted_vids = do_sort(embed_q=arr_q, embed_talks=embeds, list_talk_ids=talk_ids)

        # Limit the retrieved documents based on a score threshold
        keep_texts = limit_docs(sorted_vids=sorted_vids, talk_info=talk_info, n_results=n_results)

        logger.info(f"Retrieved {len(keep_texts)} documents for query: {query0}")

        return keep_texts
    except Exception as e:
        logger.error(f"Error during document retrieval for query: {query0}, Error: {str(e)}")
        raise e
