from shiny.express import ui, input


from shinylive_rag_fns import do_rag

ui.page_opts(
    title="Use Shiny to Run RAG on the previous R/Gov Talks",
    fillable=True,
    fillable_mobile=True,
)

# with ui.layout_sidebar():
with ui.sidebar():
# @render.code
# def get_oai_api_key():
    ui.input_password(id='oai_api_key', label="Put your OpenAI API Key here:")


# Create a chat instance and display it
chat = ui.Chat(id="chat")
chat.ui()


# Define a callback to run when the user submits a message
@chat.on_user_submit
async def _():
    user_message = chat.user_input()
    response, _ = do_rag(
        user_input=user_message, oai_api_key=input.oai_api_key(), n_results=3, stream=True
    )
    # Append the response into the chat
    await chat.append_message_stream(response)
