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
        restored = pd.read_parquet(io.BytesIO(original.to_parquet(index=False)))
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

    def test_authenticated_app_and_password_rotation(self):
        if not USES_AUTH:
            return
        from app_auth import authenticate_session, check_credentials, hash_password, session_authenticated
        stored_hash = hash_password("test-password-only")
        users = {"test-user": stored_hash}
        self.assertFalse(check_credentials("test-user", "wrong", users))
        self.assertFalse(check_credentials("test-user", "plaintext", {"test-user": "plaintext"}))
        state = {}
        self.assertTrue(authenticate_session(state, "test-user", "test-password-only", users))
        self.assertTrue(session_authenticated(state, users))
        self.assertFalse(session_authenticated(state, {"test-user": hash_password("replacement")}))
        self.assertFalse(session_authenticated(state, {}))
        app = AppTest.from_file(str(ROOT / ENTRYPOINT))
        app.secrets["auth"] = {"users": users}
        app.session_state["auth_user"] = "test-user"
        app.session_state["auth_stamp"] = hashlib.sha256(stored_hash.encode()).hexdigest()
        app.run(timeout=30)
        self.assertEqual([], [error.message for error in app.exception])


if __name__ == "__main__":
    unittest.main()
