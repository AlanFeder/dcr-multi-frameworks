import json
import pickle
import io

import numpy as np
import requests


def import_data() -> tuple[list[str], np.ndarray, dict]:
    """
    Import data from files.

    Returns:
        tuple[list[str], np.ndarray, dict]: A tuple containing the talks IDs, embeddings, and talk info.
    """

    target_file_url = "https://github.com/AlanFeder/dcr-multi-frameworks/raw/refs/heads/main/data/interim/embeds_talks_dcr.pkl"

    response = requests.get(target_file_url)
    response.raise_for_status()  # Ensure we got a successful response

    # Load the .pkl file into a Python object
    data2load = pickle.load(io.BytesIO(response.content))


    talk_ids = data2load["talk_ids"]
    embeds = data2load["embeds"]
    talk_info = data2load["talk_info"]

    return talk_ids, embeds, talk_info



def do_1_embed(lt: str, oai_api_key: str) -> np.ndarray:
    """
    Generate embeddings using the OpenAI API for a single text.

    Args:
        lt (str): A text to generate embeddings for.
        emb_client (OpenAI): The embedding API client (OpenAI).

    Returns:
        np.ndarray: The generated embeddings.
    """
    # OpenAI API endpoint for embeddings
    url = "https://api.openai.com/v1/embeddings"

    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {oai_api_key}",
    }

    # Request payload
    payload = {"input": lt, "model": "text-embedding-3-small"}

    # Make the API request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        embed_response = response.json()

        # Extract the embedding
        here_embed = np.array(embed_response["data"][0]["embedding"])

        return here_embed
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


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
    sorted_vids: list[dict[str, str | float]],
    talk_info: dict[str, str | int],
    n_results: int,
) -> list[dict]:
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

    return keep_texts


def do_retrieval(
    query0: str,
    n_results: int,
    talk_ids: list[str],
    oai_api_key: str,
    embeds: np.ndarray,
    talk_info: dict[str, str | int],
) -> list[dict]:
    """
    Retrieve relevant documents based on the user's query.

    Args:
        query0 (str): The user's query.
        n_results (int): The number of documents to retrieve.
        api_client (OpenAI): The API client (OpenAI) for generating embeddings.

    Returns:
        dict[str, dict]: The retrieved documents.
    """
    try:
        # Generate embeddings for the query
        arr_q = do_1_embed(query0, oai_api_key=oai_api_key)

        # Sort documents based on their cosine similarity to the query embedding
        sorted_vids = do_sort(embed_q=arr_q, embed_talks=embeds, list_talk_ids=talk_ids)

        # Limit the retrieved documents based on a score threshold
        keep_texts = limit_docs(
            sorted_vids=sorted_vids, talk_info=talk_info, n_results=n_results
        )

        return keep_texts
    except Exception as e:
        raise e


SYSTEM_PROMPT = """
You are an AI assistant that helps answer questions by searching through video transcripts. 
I have retrieved the transcripts most likely to answer the user's question.
Carefully read through the transcripts to find information that helps answer the question. 
Be brief - your response should not be more than two paragraphs.
Only use information directly stated in the provided transcripts to answer the question. 
Do not add any information or make any claims that are not explicitly supported by the transcripts.
If the transcripts do not contain enough information to answer the question, state that you do not have enough information to provide a complete answer.
Format the response clearly.  If only one of the transcripts answers the question, don't reference the other and don't explain why its content is irrelevant.
Do not speak in the first person. DO NOT write a letter, make an introduction, or salutation.
Reference the speaker's name when you say what they said.
"""


def set_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """
    Set the messages for the chat completion.

    Args:
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.

    Returns:
        tuple[list[dict[str, str]], int]: A tuple containing the messages and the total number of input tokens.
    """
    messages1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages1


def make_user_prompt(question: str, keep_texts: list[dict]) -> str:
    """
    Create the user prompt based on the question and the retrieved transcripts.

    Args:
        question (str): The user's question.
        keep_texts (dict[str, dict[str, str]]): The retrieved transcripts.

    Returns:
        str: The user prompt.
    """
    user_prompt = f"""
Question: {question}
==============================
"""
    if len(keep_texts) > 0:
        list_strs = []
        for i, tx_val in enumerate(keep_texts):
            text0 = tx_val["text"]
            speaker_name = tx_val["Speaker"]
            list_strs.append(
                f"Video Transcript {i+1}\nSpeaker: {speaker_name}\n{text0}"
            )
        user_prompt += "\n-------\n".join(list_strs)
        user_prompt += """
==============================
After analyzing the above video transcripts, please provide a helpful answer to my question. Remember to stay within two paragraphs
Address the response to me directly.  Do not use any information not explicitly supported by the transcripts. Remember to reference the speaker's name."""
    else:
        # If no relevant transcripts are found, generate a default response
        user_prompt += "No relevant video transcripts were found.  Please just return a result that says something like 'I'm sorry, but the answer to {Question} was not found in the transcripts from the R/Gov Conference'"
    # logger.info(f'User prompt: {user_prompt}')
    return user_prompt


def parse_1_query_stream(response):

    # Check if the request was successful
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            yield f"Error decoding JSON: {data}"
    else:
        yield f"Error: {response.status_code}\n{response.text}"


def parse_1_query_no_stream(response):
    if response.status_code == 200:
        try:
            response1 = response.json()
            completion = response1["choices"][0]["message"]["content"]
            return completion
        except json.JSONDecodeError:
            return f"Error decoding JSON: {response.text}"
    else:
        return f"Error: {response.status_code}\n{response.text}"


def do_1_query(messages1: list[dict[str, str]], oai_api_key: str, stream: bool):
    """
    Generate a response using the specified chat completion model.

    Args:
        messages1 (list[dict[str, str]]): The messages for the chat completion.
        gen_client (OpenAI): The generation client (OpenAI).
    """

    # OpenAI API endpoint for chat completions
    url = "https://api.openai.com/v1/chat/completions"

    # Your OpenAI API key
    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {oai_api_key}",
    }
    if stream:
        headers["Accept"] = "text/event-stream"  # Required for streaming

    # Model to use
    model1 = "gpt-4o"

    # Request payload
    payload = {
        "model": model1,
        "messages": messages1,
        "seed": 18,
        "temperature": 0,
        "stream": stream,
    }

    # Make the API request
    response = requests.post(
        url, headers=headers, data=json.dumps(payload), stream=stream
    )

    if stream:
        response1 = parse_1_query_stream(response)
    else:
        # Check if the request was successful
        response1 = parse_1_query_no_stream(response)

    return response1


def do_generation(query1: str, keep_texts: list[dict], oai_api_key: str, stream: bool):
    """
    Generate the chatbot response using the specified generation client.

    Args:
        query1 (str): The user's query.
        keep_texts (dict[str, dict[str, str]]): The retrieved relevant texts.
        gen_client (OpenAI): The generation client (OpenAI).

    Returns:
        tuple[Stream, int]: A tuple containing the generated response stream and the number of prompt tokens.
    """
    user_prompt = make_user_prompt(query1, keep_texts=keep_texts)
    messages1 = set_messages(SYSTEM_PROMPT, user_prompt)
    response = do_1_query(messages1, oai_api_key=oai_api_key, stream=stream)

    return response


def calc_cost(
    prompt_tokens: int, completion_tokens: int, embedding_tokens: int
) -> float:
    """
    Calculate the cost in cents based on the number of prompt, completion, and embedding tokens.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        embedding_tokens (int): The number of tokens in the embedding.

    Returns:
        float: The cost in cents.
    """
    prompt_cost = prompt_tokens / 2000
    completion_cost = 3 * completion_tokens / 2000
    embedding_cost = embedding_tokens / 500000

    cost_cents = prompt_cost + completion_cost + embedding_cost

    return cost_cents


def do_rag(user_input: str, oai_api_key: str, stream: bool = False, n_results: int = 3):
    # Load the data
    talk_ids, embeds, talk_info = import_data()
    # Load the model


    retrieved_docs = do_retrieval(
        query0=user_input,
        n_results=n_results,
        oai_api_key=oai_api_key,
        talk_ids=talk_ids,
        embeds=embeds,
        talk_info=talk_info,
    )

    response = do_generation(
        query1=user_input,
        keep_texts=retrieved_docs,
        oai_api_key=oai_api_key,
        stream=stream,
    )

    return response, retrieved_docs
