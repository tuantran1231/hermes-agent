"""Regression tests for LACVON/CYCLOP bridge tool authorization.

These tests validate:

- tool_search and tool_describe are treated as catalog reads (allow).
- tool_call delegates per-tool authorization to the policy engine.
- The gate fails closed when the policy engine is unavailable.
- The hermes-agent policy filter drops tools the agent is not authorized
  to invoke, while falling back to unfiltered results when the policy
  file is missing.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------
# The gate lives outside the hermes-agent package, so we load it with
# importlib.  We must set the policy environment variable BEFORE the gate
# module is imported, because POLICY_PATH is a module-level constant.

_GATE_SOURCE = "/tmp/lacvon_gate_fixed.py"
_gate_module = None


def _load_gate_module(policy_path: str = "/tmp/test_lacvon_policy.json"):
    """Import the gate module, forcing POLICY_PATH to the supplied path.
    
    HERMES_PROFILE is cleared so _profile_from() falls through to the
    payload-provided profile instead of picking up the engineering profile
    from the test runner's shell environment.
    """
    import importlib.util

    os.environ["TRANSIQ_CC_POLICY"] = policy_path
    # Prevent the engineering profile from leaking into gate evaluation.
    os.environ.pop("HERMES_PROFILE", None)
    os.environ.pop("HERMES_PROFILE_NAME", None)

    spec = importlib.util.spec_from_file_location("traniq_gate_lacvon_test", _GATE_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Force the policy path to the test policy (the module-level POLICY_PATH
    # may be overridden by the test environment).
    mod.POLICY_PATH = policy_path
    return mod


def _make_payload(tool_name: str, tool_input: dict | None = None,
                  agent: str = "lacvon-chief-of-staff",
                  profile: str = "lacvon-chief-of-staff") -> dict:
    return {
        "hook_event_name": "pre_tool_call",
        "tool_name": tool_name,
        "tool_input": tool_input or {},
        "session_id": "test-session",
        "cwd": "/tmp",
        "profile": profile,
    }


class TestLacvonBridgeAuthorization(unittest.TestCase):
    """Gate-level tests for bridge tool handling."""

    @classmethod
    def setUpClass(cls):
        # Write a minimal test policy so the gate can load it.
        cls._policy_path = "/tmp/test_lacvon_bridge_policy.json"
        cls._policy_doc = {
            "version": 1,
            "default_effect": "deny",
            "agents": {
                "lacvon-chief-of-staff": {
                    "rules": [
                        {
                            "id": "lacvon-cyclop-inspection",
                            "effect": "allow",
                            "match": {"tool": "^lacvon_(workspace|evidence|verify)$"},
                            "description": "Allow LACVON inspection tools",
                        },
                        {
                            "id": "lacvon-no-terminal",
                            "effect": "deny",
                            "match": {"tool": "^terminal$"},
                            "description": "Deny terminal access",
                        },
                    ]
                }
            },
            "global_rules": [],
        }
        with open(cls._policy_path, "w") as f:
            json.dump(cls._policy_doc, f)

    def setUp(self):
        # Reload the gate module fresh for each test (POLICY_PATH set in setUpClass).
        self.gate = _load_gate_module(self._policy_path)

    def test_tool_search_is_allowed(self):
        """tool_search is a catalog read — the gate should allow it."""
        result = self.gate.decide(_make_payload("tool_search"))
        self.assertEqual(result, {})

    def test_tool_describe_is_allowed(self):
        """tool_describe is a catalog read — the gate should allow it."""
        result = self.gate.decide(_make_payload("tool_describe"))
        self.assertEqual(result, {})

    def test_tool_call_allowed_tool_is_allowed(self):
        """tool_call for an authorized tool is allowed."""
        result = self.gate.decide(_make_payload(
            "tool_call",
            {"calls": [{"name": "lacvon_workspace", "arguments": {}}]},
        ))
        self.assertEqual(result, {})

    def test_tool_call_denied_tool_is_blocked(self):
        """tool_call for an unauthorized tool is blocked."""
        result = self.gate.decide(_make_payload(
            "tool_call",
            {"calls": [{"name": "terminal", "arguments": {"cmd": "ls"}}]},
        ))
        self.assertEqual(result["action"], "block")
        self.assertIn("terminal", result["message"])

    def test_tool_call_multiple_tools_one_denied(self):
        """If any tool in a batch is denied, the entire tool_call is blocked."""
        result = self.gate.decide(_make_payload(
            "tool_call",
            {"calls": [
                {"name": "lacvon_workspace", "arguments": {}},
                {"name": "terminal", "arguments": {"cmd": "ls"}},
            ]},
        ))
        self.assertEqual(result["action"], "block")

    def test_connector_tool_call_is_not_policy_checked(self):
        """Connector tools (connectors__*) skip per-tool policy checks."""
        result = self.gate.decide(_make_payload(
            "tool_call",
            {"calls": [{"name": "connectors__gmail_search", "arguments": {}}]},
        ))
        self.assertEqual(result, {})

    def test_direct_tool_call_is_checked(self):
        """Direct tool calls (not bridge tools) are checked by the policy."""
        result = self.gate.decide(_make_payload(
            "terminal",
            {"cmd": "ls"},
        ))
        self.assertEqual(result["action"], "block")

    def test_gate_fails_closed_when_policy_engine_unavailable(self):
        """If the policy engine is None, the gate should fail closed."""
        gate = _load_gate_module(self._policy_path)
        original = gate._policy_mod
        gate._policy_mod = None
        try:
            result = gate.decide(_make_payload("terminal", tool_input={"cmd": "ls"}))
        finally:
            gate._policy_mod = original
        self.assertEqual(result["action"], "block")

    def tearDown(self):
        # Reset cached _POLICY to force reload on next test.
        self.gate._POLICY = None
        self.gate._LOAD_ERROR = ""


class TestLacvonPolicyFilter(unittest.TestCase):
    """hermes-agent policy filter tests for tool_search results."""

    def setUp(self):
        self._policy_path = "/tmp/test_lacvon_filter_policy.json"
        with open(self._policy_path, "w") as f:
            json.dump({
                "version": 1,
                "default_effect": "deny",
                "agents": {
                    "lacvon-chief-of-staff": {
                        "rules": [
                            {
                                "id": "lacvon-cyclop-inspection",
                                "effect": "allow",
                                "match": {"tool": "^lacvon_(workspace|evidence)$"},
                            }
                        ]
                    }
                },
                "global_rules": [],
            }, f)

    def tearDown(self):
        if os.path.exists(self._policy_path):
            os.remove(self._policy_path)
        # Reset the module-level cache.
        import tools.tool_search as ts
        ts._POLICY_FILTER_CACHE = None

    def test_filter_allows_authorized_tool(self):
        from tools.tool_search import _is_tool_allowed_by_policy
        with patch.dict(os.environ, {"TRANSIQ_CC_POLICY": self._policy_path}):
            self.assertTrue(_is_tool_allowed_by_policy("lacvon_workspace", "lacvon-chief-of-staff"))

    def test_filter_denies_unauthorized_tool(self):
        from tools.tool_search import _is_tool_allowed_by_policy
        with patch.dict(os.environ, {"TRANSIQ_CC_POLICY": self._policy_path}):
            self.assertFalse(_is_tool_allowed_by_policy("terminal", "lacvon-chief-of-staff"))

    def test_filter_falls_back_when_policy_missing(self):
        from tools.tool_search import _is_tool_allowed_by_policy
        with patch.dict(os.environ, {"TRANSIQ_CC_POLICY": "/tmp/nonexistent_filter_policy.json"}, clear=True):
            self.assertIsNone(_is_tool_allowed_by_policy("lacvon_workspace", "lacvon-chief-of-staff"))

    def test_filter_unknown_agent_returns_default(self):
        from tools.tool_search import _is_tool_allowed_by_policy
        with patch.dict(os.environ, {"TRANSIQ_CC_POLICY": self._policy_path}):
            # unknown agent → no matching rules → default_effect=deny
            self.assertFalse(_is_tool_allowed_by_policy("lacvon_workspace", "unknown-agent"))


if __name__ == "__main__":
    unittest.main()
