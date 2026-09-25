from __future__ import annotations

import io
import unittest
from unittest import mock
from urllib import error

from dataset_tools.dataset_pipeline import OpenAICompatibleClient


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"ok": true}'


class ApiRetryTests(unittest.TestCase):
    def test_retries_transient_502_then_succeeds(self) -> None:
        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="test",
            model="gpt-5.5",
            max_http_retries=3,
        )
        transient = error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            502,
            "Bad Gateway",
            None,
            io.BytesIO(b'{"error":"upstream"}'),
        )
        with (
            mock.patch(
                "dataset_tools.dataset_pipeline.request.urlopen",
                side_effect=[transient, _Response()],
            ) as urlopen,
            mock.patch("dataset_tools.dataset_pipeline.time.sleep") as sleep,
        ):
            result = client._post_json("/chat/completions", {"model": "gpt-5.5"})

        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, urlopen.call_count)
        sleep.assert_called_once_with(1)

    def test_does_not_retry_non_transient_400(self) -> None:
        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="test",
            model="gpt-5.5",
            max_http_retries=3,
        )
        invalid = error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            400,
            "Bad Request",
            None,
            io.BytesIO(b'{"error":"invalid"}'),
        )
        with (
            mock.patch("dataset_tools.dataset_pipeline.request.urlopen", side_effect=invalid) as urlopen,
            mock.patch("dataset_tools.dataset_pipeline.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "API request failed 400"):
                client._post_json("/chat/completions", {"model": "gpt-5.5"})

        self.assertEqual(1, urlopen.call_count)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
