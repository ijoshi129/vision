from __future__ import annotations

import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console
from rich.text import Text

from vision.cli import _cd_target, _failed_turn_delta, _split_usage_arg, _usage_renderable, _usage_selection


class FailedTurnDeltaTests(unittest.TestCase):
    def test_empty_reply_gets_the_wrapper(self):
        err = "You've hit your session limit · resets 6:40pm (America/New_York)"
        self.assertEqual(_failed_turn_delta("", err), f"**Vision could not answer:** {err}")
        self.assertEqual(_failed_turn_delta(None, err, markdown=False), f"Vision could not answer: {err}")

    def test_same_text_is_not_repeated(self):
        err = "You've hit your session limit · resets 6:40pm (America/New_York)"
        self.assertEqual(_failed_turn_delta(err, err), "")
        self.assertEqual(_failed_turn_delta(err + "\n", err), "")
        self.assertEqual(_failed_turn_delta(f"Vision could not answer: {err}", err, markdown=False), "")
        self.assertEqual(_failed_turn_delta(f"**Vision could not answer:** {err}", err), "")

    def test_real_reply_then_error_is_appended(self):
        self.assertEqual(
            _failed_turn_delta("half an answer", "boom"),
            "\n\n**Vision could not answer:** boom",
        )


class UsageRenderableTests(unittest.TestCase):
    report = (
        "Current session: 40% used · resets 6:40pm (America/New_York)\n"
        "Current week: 12% used · resets Sep 24\n"
        "\n"
        "Sonnet 4.6: 80% of usage\n"
    )

    def _print(self, full=False):
        buf = io.StringIO()
        with patch("vision.usage.claude_banked", return_value=None):  # no network in tests
            r = _usage_renderable(self.report, full=full)
        Console(file=buf, force_terminal=False, width=120, color_system=None).print(r)
        return buf.getvalue()

    def test_hides_token_hint_and_contribution_until_full(self):
        text = self._print()
        self.assertIn("Current session", text)
        self.assertIn("40%", text)
        self.assertNotIn("vision usage --full", text)
        self.assertNotIn("/usage full", text)
        self.assertNotIn("Sonnet 4.6", text)

    def test_full_shows_contribution_without_the_hint(self):
        text = self._print(full=True)
        self.assertIn("Sonnet 4.6", text)
        self.assertNotIn("vision usage --full", text)
        self.assertNotIn("/usage full", text)

    def test_all_selects_every_provider_and_keeps_going_if_one_is_unavailable(self):
        self.assertEqual(_split_usage_arg("all full"), ("all", True))
        requested = []

        def select(_cfg, _current, provider):
            requested.append(provider)
            return SimpleNamespace(provider=provider)

        def render(brain, full):
            self.assertTrue(full)
            if brain.provider == "codex":
                raise RuntimeError("not logged in")
            return Text(f"{brain.provider} usage")

        with patch("vision.cli._usage_brain", side_effect=select), patch("vision.cli._usage_for", side_effect=render):
            result = _usage_selection(SimpleNamespace(), SimpleNamespace(provider="claude"), "all", full=True)

        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=120, color_system=None).print(result)
        text = buf.getvalue()
        self.assertEqual(requested, ["claude", "codex", "grok"])
        self.assertIn("claude usage", text)
        self.assertIn("Codex usage unavailable: not logged in", text)
        self.assertIn("grok usage", text)


if __name__ == "__main__":
    unittest.main()


class ActiveSummaryTests(unittest.TestCase):
    """The status row and open card name what answers the next turn: while /talk is on, spoken input
    goes to the [conversation] model and typed input to the brain, so both are shown."""

    def _cfg(self, voice_model: str = "qwen3.6", voice_effort: str = "off"):
        from vision.config import Config

        cfg = Config()
        cfg.brain.model, cfg.brain.effort = "sonnet", "high"
        cfg.conversation.model, cfg.conversation.effort = voice_model, voice_effort
        return cfg

    def _brain(self, cfg, resolved: str = "claude-sonnet-5"):
        return SimpleNamespace(cfg=cfg.brain, resolved_model=lambda: resolved)

    def test_talk_names_only_the_conversation_model(self):
        from vision.cli import _active_summary

        cfg = self._cfg()
        self.assertEqual(_active_summary(cfg, self._brain(cfg), True, " "), "Qwen 3.6 35B-A3B thinking off")

    def test_off_talk_is_just_the_brain(self):
        from vision.cli import _active_summary

        cfg = self._cfg()
        self.assertEqual(_active_summary(cfg, self._brain(cfg), False, " "), "Sonnet 5 high")

    def test_same_model_for_both_is_named_once(self):
        from vision.cli import _active_summary

        cfg = self._cfg("sonnet", "high")
        self.assertEqual(_active_summary(cfg, self._brain(cfg), True, " "), "Sonnet 5 high")


class CornerLabelTests(unittest.TestCase):
    """The input box's bottom-right border names the typed brain the way the Grok CLI does:
    `Sonnet 5 (high)`, `· fast` when the speed tier is on, no parenthesis with effort off."""

    def _cfg(self, effort: str = "high", fast: bool = False):
        from vision.config import Config

        cfg = Config()
        cfg.brain.model, cfg.brain.effort, cfg.brain.fast = "sonnet", effort, fast
        return cfg

    def _brain(self, cfg, resolved: str = "claude-sonnet-5"):
        return SimpleNamespace(cfg=cfg.brain, resolved_model=lambda: resolved)

    def test_model_and_effort_in_grok_form(self):
        from vision.cli import _corner_label

        cfg = self._cfg()
        self.assertEqual(_corner_label(cfg, self._brain(cfg)), "Sonnet 5 (high)")

    def test_fast_tier_is_appended(self):
        from vision.cli import _corner_label

        cfg = self._cfg(fast=True)
        self.assertEqual(_corner_label(cfg, self._brain(cfg)), "Sonnet 5 (high) · fast")

    def test_effort_off_has_no_parenthesis(self):
        from vision.cli import _corner_label

        cfg = self._cfg(effort="")
        self.assertEqual(_corner_label(cfg, self._brain(cfg)), "Sonnet 5")

    def test_corner_frame_builds_with_and_without_a_label(self):
        from prompt_toolkit.layout import to_container
        from prompt_toolkit.widgets import TextArea

        from vision.ui import CornerFrame

        for text in ("Sonnet 5 (high)", ""):
            frame = CornerFrame(TextArea(), lambda text=text: text)
            self.assertIsNotNone(to_container(frame))


class CdTargetTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        os.mkdir(os.path.join(self.root, "sub"))
        os.mkdir(os.path.join(self.root, "sub", "deeper"))
        with open(os.path.join(self.root, "file.txt"), "w") as f:
            f.write("x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_relative_absolute_and_dotdot(self):
        sub = os.path.join(self.root, "sub")
        self.assertEqual(_cd_target("sub", self.root, None), sub)
        self.assertEqual(_cd_target("sub/deeper", self.root, None), os.path.join(sub, "deeper"))
        self.assertEqual(_cd_target(sub, "/", None), sub)
        self.assertEqual(_cd_target("..", os.path.join(sub, "deeper"), None), sub)
        self.assertEqual(_cd_target(" sub ", self.root, None), sub)

    def test_home_and_previous(self):
        self.assertEqual(_cd_target("~", self.root, None), os.path.expanduser("~"))
        self.assertEqual(_cd_target("-", self.root, "/"), os.path.normpath("/"))
        with self.assertRaisesRegex(ValueError, "no previous directory"):
            _cd_target("-", self.root, None)

    def test_missing_and_files_are_refused(self):
        with self.assertRaisesRegex(ValueError, "no such directory"):
            _cd_target("nope", self.root, None)
        with self.assertRaisesRegex(ValueError, "not a directory"):
            _cd_target("file.txt", self.root, None)
