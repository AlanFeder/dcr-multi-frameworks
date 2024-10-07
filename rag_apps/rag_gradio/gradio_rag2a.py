import sys

import gradio as gr
from pyprojroot import here

sys.path.append(str(here()))


from rag_fns.rag import do_rag
from rag_utils.rag_utils import calc_cost, calc_n_tokens

# for word in response:
#     yield word

# demo = gr.ChatInterface(fn=gr_ch_if, title="Use Gradio to Run RAG on the previous R/Gov Talks - Chat Interface 1")

# def vote(data: gr.LikeData):
#     if data.liked:
#         print("You upvoted this response: " + data.value)
#     else:
#         print("You downvoted this response: " + data.value)


# def create_video_html(video_info):
#     html = ""
#     for vid_info in video_info:
#         yt_id = vid_info["VideoURL"].split("/")[-1].split("=")[-1]
#         yt_url = f"https://www.youtube.com/embed/{yt_id}"
#         html += f"""
#         <div style="margin-bottom: 20px;">
#             <h3>{vid_info["Title"]}</h3>
#             <p><em>{vid_info["Speaker"]}</em></p>
#             <p>Year: {vid_info["Year"]}</p>
#             <p>Similarity Score: {100 * vid_info["score"]:.0f}/100</p>
#             <iframe width="100%" height="315" src="{yt_url}" frameborder="0" allowfullscreen></iframe>
#             <details>
#                 <summary>Transcript</summary>
#                 <p>{vid_info["text"]}</p>
#             </details>
#         </div>
#         """
#     return html


def create_video_html(video_info):
    htmls = []
    for vid_info in video_info:
        yt_id = vid_info["VideoURL"].split("/")[-1].split("=")[-1]
        yt_url = f"https://www.youtube.com/embed/{yt_id}"
        html = f"""
        <div style="margin-bottom: 20px;">
            <h3>{vid_info["Title"]}</h3>
            <p><em>{vid_info["Speaker"]}</em></p>
            <p>Year: {vid_info["Year"]}</p>
            <p>Similarity Score: {100 * vid_info["score"]:.0f}/100</p>
            <iframe width="100%" height="315" src="{yt_url}" frameborder="0" allowfullscreen></iframe>
            <details>
                <summary>Transcript</summary>
                <p>{vid_info["text"]}</p>
            </details>
        </div>
        """
        htmls.append(html)
    return htmls


def cost_words(response, prompt_tokens):
    completion_tokens = calc_n_tokens(response)
    cost_cents = calc_cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    text_out = f"This cost approximately {cost_cents:.01f}¢"

    return text_out


def gr_ch_if(user_input: str):
    response, retrieved_docs, prompt_tokens = do_rag(user_input, stream=False, n_results=3)

    video_html = create_video_html(retrieved_docs)

    text_out = cost_words(response, prompt_tokens)

    return response, video_html, text_out


# Create Gradio interface
# iface = gr.Interface(
#     fn=gr_ch_if,
#     inputs=gr.Textbox(label="Enter your question:"),
#     outputs=[
#         gr.Textbox(label="Response", interactive=False),
#         gr.Textbox(label="Cost", interactive=False),
#         gr.HTML(label="Relevant Videos")
#     ],
#     title="RAG on R/Gov Talks",
#     description="Use Gradio to Run RAG on the previous R/Gov Talks",
#     allow_flagging="never"
# )


# Create Gradio interface with single column layout
with gr.Blocks() as iface:
    gr.Markdown("# RAG on R/Gov Talks")
    gr.Markdown("Use Gradio to Run RAG on the previous R/Gov Talks")

    query_input = gr.Textbox(label="Enter your question:")
    response_output = gr.Textbox(label="Response", interactive=False)
    video_output = gr.HTML(label="Relevant Videos")
    cost_output = gr.Textbox(label="Cost", interactive=False)

    query_input.submit(
        fn=gr_ch_if, inputs=[query_input], outputs=[response_output, video_output, cost_output]
    )

# iface = gr.ChatInterface(
#     chat_function,
#     chatbot=gr.Chatbot(height=600),
#     textbox=gr.Textbox(placeholder="Ask a question about R/Gov Talks", container=False, scale=7),
#     title="RAG on R/Gov Talks",
#     description="Use this chatbot to ask questions about previous R/Gov Talks",
#     theme="soft",
#     examples=[
#         "What were some key topics discussed in recent talks?",
#         "Can you summarize the main points from talks about data science in government?",
#         "What insights were shared about using R in public policy analysis?"
#     ],
#     cache_examples=False,
#     retry_btn=None,
#     undo_btn="Delete Last",
#     clear_btn="Clear",
# )


# Launch the app
if __name__ == "__main__":
    iface.launch()
