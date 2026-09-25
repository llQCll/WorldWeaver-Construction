from __future__ import annotations

import json
import unittest
from unittest import mock

from dataset_tools.dataset_pipeline import OpenAICompatibleClient


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'


class ApiLimitTests(unittest.TestCase):
    def test_chat_json_sends_reasoning_and_completion_limits(self) -> None:
        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="test",
            model="gpt-5.5",
            reasoning_effort="low",
            max_completion_tokens=2000,
        )
        with mock.patch(
            "dataset_tools.dataset_pipeline.request.urlopen", return_value=_Response()
        ) as urlopen:
            result = client.chat_json(
                system_prompt="Return JSON.",
                user_prompt="Return ok.",
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual({"ok": True}, result)
        self.assertEqual("low", payload["reasoning_effort"])
        self.assertEqual(2000, payload["max_completion_tokens"])


if __name__ == "__main__":
    unittest.main()
