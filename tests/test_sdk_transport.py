"""Verify application requests against the upgraded SDK without network access."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx2
import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_utils import rag_utils
from rag_fns import generation


class SDKTransportTests(unittest.TestCase):
    @patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
    def test_embedding_and_streaming_with_the_real_sdk(self):
        requests = []

        def respond(request):
            self.assertEqual("Bearer offline-placeholder", request.headers["authorization"])
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path == "/v1/embeddings":
                return httpx2.Response(200, json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                })
            self.assertEqual("/v1/chat/completions", request.url.path)
            self.assertTrue(body["stream"])
            event = {
                "id": "chatcmpl-offline", "object": "chat.completion.chunk",
                "created": 0, "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
            }
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"},
                content=("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode(),
            )

        with OpenAI(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        ) as client:
            client = client
            np.testing.assert_array_equal([1.0, 0.0], rag_utils.do_1_embed("question", client))
            messages = [{"role": "user", "content": "question"}]
            chunks = list(generation.do_1_query(messages, client, stream=True))
            self.assertEqual("answer", "".join(chunk.choices[0].delta.content or "" for chunk in chunks))
        self.assertEqual("question", requests[0][1]["input"])
        self.assertEqual(messages, requests[1][1]["messages"])

    def test_client_factory_initializes_sdk_with_keyword_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder"}):
            with rag_utils.make_oai_client() as client:
                self.assertEqual("offline-placeholder", client.api_key)


