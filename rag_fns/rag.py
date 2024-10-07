from rag_fns.generation import do_generation
from rag_fns.retrieval import do_retrieval
from rag_fns.setup_load import import_data, load_oai_model


def do_rag(user_input: str, stream: bool = False, n_results: int = 3):
    # Load the data
    talk_ids, embeds, talk_info = import_data()
    # Load the model
    oai_client = load_oai_model()

    retrieved_docs = do_retrieval(
        query0=user_input,
        n_results=n_results,
        api_client=oai_client,
        talk_ids=talk_ids,
        embeds=embeds,
        talk_info=talk_info,
    )

    response, prompt_tokens = do_generation(
        query1=user_input, keep_texts=retrieved_docs, gen_client=oai_client, stream=stream
    )

    return response, retrieved_docs, prompt_tokens
