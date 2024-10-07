import sys

import gradio as gr
from pyprojroot import here

sys.path.append(str(here()))


from rag_fns.rag import do_rag


def gr_ch_if(user_input: str, history):
    response, _, _ = do_rag(user_input, stream=False, n_results=3)
    return response


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        placeholder="Use Gradio to Run RAG on the previous R/Gov Talks - Chat Interface 1"
    )
    chatbot.like(None, None, None)
    gr.ChatInterface(
        fn=gr_ch_if,
        chatbot=chatbot,
        title="Use Gradio to Run RAG on the previous R/Gov Talks - Chat Interface 1",
    )

if __name__ == "__main__":
    demo.launch(share=True)
