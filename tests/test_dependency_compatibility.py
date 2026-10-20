"""Offline checks for the audited runtime and committed application data."""
import hashlib
import io
import os
from unittest.mock import patch
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENTRYPOINT = "rag_apps/rag_streamlit/streamlit_1_page.py"
FIXTURES = ['data/interim/embeds_talks_dcr.pkl']
USES_AUTH = False


class CompatibilityTests(unittest.TestCase):
    def test_parquet_and_numpy_round_trip(self):
        original = pd.DataFrame({"value": np.array([1.0, 2.0, 3.0])})
        restored = pd.read_parquet(io.BytesIO(original.to_parquet(index=False)), use_threads=False)
        pd.testing.assert_frame_equal(restored, original)

    def test_committed_data_still_loads(self):
        for filename in FIXTURES:
            with self.subTest(file=filename), (ROOT / filename).open("rb") as source:
                self.assertIsNotNone(pickle.load(source))

    def test_app_starts_without_external_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-placeholder"}), \
                patch("httpx.Client.send", side_effect=AssertionError("API calls are disabled in this test")):
            app = AppTest.from_file(str(ROOT / ENTRYPOINT)).run(timeout=30)
        self.assertEqual([], [error.message for error in app.exception])



if __name__ == "__main__":
    unittest.main()
