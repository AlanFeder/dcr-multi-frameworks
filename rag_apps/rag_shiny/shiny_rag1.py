import sys

from pyprojroot import here
from shiny import reactive
from shiny.express import input, render, ui

sys.path.append(str(here()))
from rag_fns.rag import do_rag
from rag_utils.rag_utils import calc_cost, calc_n_tokens

ui.page_opts(
    title="Use Shiny to Run RAG on the previous R/Gov Talks",
    fillable=True,
    fillable_mobile=True,
)


ui.input_text(
    id="query1", label="What question do you want to ask?", placeholder="What is the tidyverse?"
)
ui.input_action_button("run_rag", "Submit!")


rag_answer = reactive.value("")
list_retrieved_docs = reactive.value([])
n_prompt_tokens = reactive.value(0)


@reactive.effect
@reactive.event(input.run_rag)
def do_rag_shiny():
    response, retrieved_docs, prompt_tokens = do_rag(
        user_input=input.query1(), n_results=3, stream=False
    )

    rag_answer.set(response)
    list_retrieved_docs.set(retrieved_docs)
    n_prompt_tokens.set(prompt_tokens)


# render.text(rag_answer())
# @reactive.event(input.action_button)
# @render.text
@render.ui
def render_response():
    ans = rag_answer()
    # return ans
    return ui.markdown(ans)


@render.ui
def create_video_html():
    video_info = list_retrieved_docs()
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
    return ui.HTML(html)


# @render.text
# @render.express
@render.ui
def cost_words():
    # return input.query1()
    completion_tokens = calc_n_tokens(rag_answer())
    cost_cents = calc_cost(
        prompt_tokens=int(n_prompt_tokens()), completion_tokens=completion_tokens
    )
    text_out = f"This cost approximately {cost_cents:.01f}¢"
    if cost_cents:
        # return text_out
        # return ui.help_text(text_out)
        return ui.div(text_out, style="color:grey; font-size:60%;")
