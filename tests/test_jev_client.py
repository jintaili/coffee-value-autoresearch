"""The provider request uses an application identity accepted by TypeSafe's edge."""

import io
import os
import unittest
from unittest.mock import patch

from coffee_value.extraction.client import evaluate


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class JevClientTests(unittest.TestCase):
    def test_request_has_explicit_user_agent(self):
        requests = []

        def respond(request, timeout):
            requests.append(request)
            return Response(b'{"model":"jev-1.13.0","answers":{}}')

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
            "coffee_value.extraction.client.urllib.request.urlopen", side_effect=respond
        ):
            evaluate({"notes": "coffee"}, questions={}, attempts=1)

        self.assertEqual(requests[0].get_header("User-agent"), "coffee-value-shared/0.1")


if __name__ == "__main__":
    unittest.main()
