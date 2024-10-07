import os
import pickle

import numpy as np
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pyprojroot import here


def load_oai_model() -> OpenAI:
    """
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Create OpenAI API client
    openai_client = OpenAI(api_key=api_key)

    return openai_client


def import_data() -> tuple[list[str], np.ndarray, dict]:
    """
    Import data from files.

    Returns:
        tuple[pd.DataFrame, dict, dict]: A tuple containing the talks dataframe, transcript dictionaries, and full embeddings.
    """

    with open(here() / "data" / "interim" / "embeds_talks_dcr.pkl", "rb") as f:
        data2load = pickle.load(f)

    talk_ids = data2load["talk_ids"]
    embeds = data2load["embeds"]
    talk_info = data2load["talk_info"]

    return talk_ids, embeds, talk_info


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

    return here_embed


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
    api_client: OpenAI,
    talk_ids: list[str],
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
        arr_q = do_1_embed(query0, api_client)

        # Sort documents based on their cosine similarity to the query embedding
        sorted_vids = do_sort(embed_q=arr_q, embed_talks=embeds, list_talk_ids=talk_ids)

        # Limit the retrieved documents based on a score threshold
        keep_texts = limit_docs(sorted_vids=sorted_vids, talk_info=talk_info, n_results=n_results)

        return keep_texts
    except Exception as e:
        raise e


def load_oai_model() -> OpenAI:
    """
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Create OpenAI API client
    openai_client = OpenAI(api_key=api_key)

    return openai_client


def import_data() -> tuple[list[str], np.ndarray, dict]:
    """
    Import data from files.

    Returns:
        tuple[pd.DataFrame, dict, dict]: A tuple containing the talks dataframe, transcript dictionaries, and full embeddings.
    """

    with open(here() / "data" / "interim" / "embeds_talks_dcr.pkl", "rb") as f:
        data2load = pickle.load(f)

    talk_ids = data2load["talk_ids"]
    embeds = data2load["embeds"]
    talk_info = data2load["talk_info"]

    return talk_ids, embeds, talk_info


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


def set_messages(system_prompt: str, user_prompt: str) -> tuple[list[dict[str, str]], int]:
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
    n_system_tokens = calc_n_tokens(system_prompt)
    n_user_tokens = calc_n_tokens(user_prompt)
    n_input_tokens = n_system_tokens + n_user_tokens

    return messages1, n_input_tokens


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
            list_strs.append(f"Video Transcript {i+1}\nSpeaker: {speaker_name}\n{text0}")
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


def do_1_query(messages1: list[dict[str, str]], gen_client: OpenAI):
    """
    Generate a response using the specified chat completion model.

    Args:
        messages1 (list[dict[str, str]]): The messages for the chat completion.
        gen_client (OpenAI): The generation client (OpenAI).

    Returns:
        Stream: The generated response stream.
    """

    model1 = "gpt-4o"

    # Generate the response using the specified model
    response1 = gen_client.chat.completions.create(
        messages=messages1, model=model1, seed=18, temperature=0, stream=True
    )

    return response1


def do_generation(query1: str, keep_texts: list[dict], gen_client: OpenAI) -> tuple:
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
    messages1, prompt_tokens = set_messages(SYSTEM_PROMPT, user_prompt)
    response = do_1_query(messages1, gen_client)

    return response, prompt_tokens


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

    return cost_cents


def run_app():

    st.set_page_config(
        page_title="Streamlit RAG on R/Gov Talks",
        page_icon="favicon_io/favicon.ico",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title("Use Streamlit to Run RAG on the previous R/Gov Talks")
    # Load the data
    talk_ids, embeds, talk_info = import_data()
    # Load the model
    oai_client = load_oai_model()

    # How many results to bring to the generator
    n_results = 3

    # Get user input
    user_input = st.text_input("Enter your question:")
    if user_input:
        # Perform retrieval
        retrieved_docs = do_retrieval(
            query0=user_input,
            n_results=n_results,
            api_client=oai_client,
            talk_ids=talk_ids,
            embeds=embeds,
            talk_info=talk_info,
        )
        # Perform generation
        response, prompt_tokens = do_generation(
            query1=user_input, keep_texts=retrieved_docs, gen_client=oai_client
        )
        # Display the response
        text_out = st.write_stream(response)

        st.divider()
        st.subheader("RAG-identified relevant videos")
        n_vids = len(retrieved_docs)
        if n_vids == 0:
            st.markdown("No relevant videos identified")
        elif n_vids == 1:
            _, vid_c1, _ = st.columns(3)
            vid_containers = [vid_c1]
        elif n_vids == 2:
            _, vid_c1, vid_c2, _ = st.columns([1 / 6, 1 / 3, 1 / 3, 1 / 6])
            vid_containers = [vid_c1, vid_c2]
        elif n_vids > 2:
            vid_containers = st.columns(n_vids)
        for i, vid_info in enumerate(retrieved_docs):
            vid_container = vid_containers[i]
            with vid_container:
                vid_title = vid_info["Title"]
                vid_speaker = vid_info["Speaker"]
                sim_score = 100 * vid_info["score"]
                vid_url = vid_info["VideoURL"]
                st.markdown(f"**{vid_title}**\n\n*{vid_speaker}*\n\nYear: {vid_info['Year']}")
                st.caption(f"Similarity Score: {sim_score:.0f}/100")
                st.video(vid_url)
                with st.expander(label="Transcript", expanded=False):
                    st.markdown(vid_info["text"])
        # logger.info("Printing completed")
        completion_tokens = calc_n_tokens(text_out)

        cost_cents = calc_cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

        st.caption(f"This cost approximately {cost_cents:.01f}¢")
        # logger.info("Cost displayed")

        # st.divider()

        # st.caption('''This streamlit app was created for Alan Feder's [talk at the 10th Anniversary New York R Conference](https://rstats.ai/nyr.html). \n\n The slides used are [here](https://bit.ly/nyr-rag). \n\n The Github repository that houses all the code is [here](https://github.com/AlanFeder/nyr-rag-app) -- feel free to fork it and use it on your own!''')

        st.divider()

        st.subheader("Contact me!")
        st.image("AJF_Headshot.jpg", width=60)
        st.markdown(
            "[Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)"
        )


if __name__ == "__main__":
    run_app()
