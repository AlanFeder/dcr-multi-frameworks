import logging
import os
import pickle
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pyprojroot import here

logger = logging.getLogger(__name__)


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

    logger.info("OpenAI Client set up")

    return openai_client


def import_data() -> tuple[list[str], np.ndarray, dict[str, Any]]:
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
