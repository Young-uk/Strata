"""Structured short tasks use the isolated lane without changing summary rules."""
import unittest
from serve.atomic_summary import request

class AuxiliaryRequestTests(unittest.TestCase):
    def test_structured_auxiliary_is_opt_in(self):
        body = dict(atomic_summary=True, think=False, messages=[dict(role="user", content="Title")],
                    response_format={"type": "json_object"})
        with self.assertRaises(ValueError): request(body)
        accepted = request({**body, "atomic_auxiliary": True})
        self.assertEqual(accepted["response_format"], body["response_format"])
        self.assertFalse(accepted["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("chat_template_kwargs", body)

    def test_auxiliary_cannot_enable_tools_or_thinking(self):
        body = dict(atomic_summary=True, atomic_auxiliary=True, think=False,
                    messages=[dict(role="user", content="Title")])
        for extra in ({"tools": [{"type": "function"}]}, {"think": True}, {"atomic_auxiliary": 1}):
            with self.assertRaises(ValueError): request({**body, **extra})

if __name__ == "__main__": unittest.main()
