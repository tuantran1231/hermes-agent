"""Regression tests for fail-closed deferred tool-call resolution.

These tests only exercise the parser boundary; they do not assert live gateway
registration or authorization behavior.
"""

from unittest import TestCase
from unittest.mock import patch

from agent import tool_executor


class DeferredResolutionErrorTests(TestCase):
    def test_bridge_parse_error_is_reported_before_hooks(self):
        with patch(
            "tools.tool_search.resolve_underlying_call",
            return_value=(None, {}, "Invalid tool_call arguments"),
        ):
            name, args, block = tool_executor._unwrap_tool_search_call(
                object(), "tool_call", {"calls": []}
            )
        self.assertEqual(name, "tool_call")
        self.assertEqual(args, {"calls": []})
        self.assertEqual(block, "Invalid tool_call arguments")

    def test_unexpected_resolver_exception_fails_closed(self):
        with patch(
            "tools.tool_search.resolve_underlying_call",
            side_effect=RuntimeError("unexpected failure"),
        ):
            name, _, block = tool_executor._unwrap_tool_search_call(
                object(), "tool_call", {"name": "lacvon_workspace", "arguments": {}}
            )
        self.assertEqual(name, "tool_call")
        self.assertIn("no tool was executed", block)

    def test_unresolved_call_without_error_fails_closed(self):
        with patch(
            "tools.tool_search.resolve_underlying_call",
            return_value=(None, {}, None),
        ):
            _, _, block = tool_executor._unwrap_tool_search_call(
                object(), "tool_call", {"name": "lacvon_workspace", "arguments": {}}
            )
        self.assertIn("did not resolve", block)

    def test_non_bridge_tools_are_not_modified(self):
        name, args, block = tool_executor._unwrap_tool_search_call(
            object(), "lacvon_workspace", {}
        )
        self.assertEqual((name, args, block), ("lacvon_workspace", {}, None))
