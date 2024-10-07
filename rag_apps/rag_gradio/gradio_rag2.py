import sys

import gradio as gr
from pyprojroot import here

sys.path.append(str(here()))
from rag_fns.rag import do_rag
from rag_utils.rag_utils import calc_cost, calc_n_tokens


def create_video_html(video_info: list) -> str:
    html = """
    <style>
        .video-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .video-item {
            width: 30%;
            min-width: 300px;
            margin-bottom: 20px;
        }
        @media (max-width: 1200px) {
            .video-item {
                width: 45%;
            }
        }
        @media (max-width: 768px) {
            .video-item {
                width: 100%;
            }
        }
    </style>
    <div class="video-container">
    """

    for vid_info in video_info:
        yt_id = vid_info["VideoURL"].split("/")[-1].split("=")[-1]
        yt_url = f"https://www.youtube.com/embed/{yt_id}"
        html += f"""
        <div class="video-item">
            <h3>{vid_info["Title"]}</h3>
            <p><em>{vid_info["Speaker"]}</em></p>
            <p>Year: {vid_info["Year"]}</p>
            <p>Similarity Score: {100 * vid_info["score"]:.0f}/100</p>
            <iframe width="100%" height="215" src="{yt_url}" frameborder="0" allowfullscreen></iframe>
            <details>
                <summary>Transcript</summary>
                <p>{vid_info["text"]}</p>
            </details>
        </div>
        """
    html += "</div>"
    return html


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


# Launch the app
if __name__ == "__main__":
    iface.launch()
