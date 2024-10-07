import logging
import sys

import streamlit as st
from pyprojroot import here

sys.path.append(str(here()))


from rag_fns.rag import do_rag
from rag_utils.rag_utils import calc_cost, calc_n_tokens

def show_response(response, to_stream: bool) -> str:
    if to_stream:
        text_out = st.write_stream(response)
    else:
        st.write(response)
        text_out = response

    return text_out

def run_app():

    st.set_page_config(
        page_title="Streamlit RAG on R/Gov Talks",
        page_icon="favicon_io/favicon.ico",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title("Use Streamlit to Run RAG on the previous R/Gov Talks")

    # Get user input
    user_input = st.text_input("Enter your question:")
    button1 = st.button("Submit")
    if user_input:
        if button1:
            # with st.spinner("Running RAG..."):
            if 1 == 1:
                to_stream = True
                n_results = 3
                response, retrieved_docs, prompt_tokens = do_rag(user_input, stream=to_stream, n_results=n_results)
                # Display the response
                text_out = show_response(response, to_stream)
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

                cost_cents = calc_cost(
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                )

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
