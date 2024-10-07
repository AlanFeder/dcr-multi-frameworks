import sys

from pyprojroot import here
from shiny.express import ui

sys.path.append(str(here()))
from rag_fns.rag import do_rag

ui.page_opts(
    title="Use Shiny to Run RAG on the previous R/Gov Talks",
    fillable=True,
    fillable_mobile=True,
)

# Create a chat instance and display it
chat = ui.Chat(id="chat")
chat.ui()


# Define a callback to run when the user submits a message
@chat.on_user_submit
async def _():
    user_message = chat.user_input()
    response, _, _ = do_rag(user_input=user_message, n_results=3, stream=True)
    # Append the response into the chat
    await chat.append_message_stream(response)
