"""The guided setup (vision/welcome.py) with scripted answers."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vision import clis, config, providers, welcome


class ScriptedIO:
    """Answers in order; every question and command is kept for the assertions."""

    def __init__(self, picks=(), answers=(), confirms=(), run_rc=0):
        self.picks, self.answers, self.confirms = list(picks), list(answers), list(confirms)
        self.said: list[str] = []
        self.asked: list[str] = []
        self.ran: list = []
        self.run_rc = run_rc

    def say(self, text=""):
        self.said.append(text)

    def pick(self, title, options, current=""):
        self.asked.append((title, [o[0] for o in options]))
        return self.picks.pop(0) if self.picks else None

    def ask(self, question, default=""):
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else default

    def confirm(self, question, default=True):
        self.asked.append(question)
        return self.confirms.pop(0) if self.confirms else default

    def run(self, cmd, shell=False):
        self.ran.append(cmd)
        return self.run_rc


class WizardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"
        self.path.write_text('[brain]\nmodel = "opus"\neffort = "high"\n', encoding="utf-8")
        self.saved = []
        self._patches = [patch.object(config, "CONFIG_PATH", self.path), patch.object(config, "ensure_dirs"),
                         patch("vision.config.save_brain_defaults", side_effect=lambda m, e: self.saved.append((m, e))),
                         patch("vision.cli.save_brain_defaults", side_effect=lambda m, e: self.saved.append((m, e))),
                         patch.object(clis, "refresh_models_soon"),
                         # no CLI anywhere: not on PATH, and not at the ~/.local/bin fallbacks the finders try
                         patch.dict(os.environ, {"PATH": "/nonexistent", "HOME": self.tmp.name})]
        for p in self._patches:
            p.start()
        providers.forget_ready()
        self.addCleanup(providers.register_endpoints, {})
        self.addCleanup(providers.forget_ready)
        self.addCleanup(self.tmp.cleanup)
        for p in self._patches:
            self.addCleanup(p.stop)

    def text(self, io):
        return "\n".join(io.said)

    def test_a_server_is_added_checked_and_made_the_default(self):
        io = ScriptedIO(picks=[welcome.SERVER, welcome.DONE, "ollama/qwen3"], answers=["ollama", "http://127.0.0.1:11434/v1", "Ollama"])
        with patch.object(clis, "local_model_ids", return_value=["qwen3", "llama4"]), \
                patch.object(providers, "_server_answers", side_effect=lambda url: "11434" in url):
            ready = welcome.Wizard(io, config.load_config()).run()
        self.assertEqual(ready, ["ollama"])
        self.assertIn('[providers.ollama]', self.path.read_text())
        self.assertIn('base_url = "http://127.0.0.1:11434/v1"', self.path.read_text())
        self.assertIn("2 models: qwen3, llama4", self.text(io))
        self.assertEqual(self.saved, [("ollama/qwen3", "off")])  # a local model starts with thinking off
        self.assertEqual(io.asked[-1][0], "Which model should Vision start on?")
        self.assertEqual(io.asked[-1][1], ["ollama/qwen3", "ollama/llama4"])  # only the ready provider's models
        self.assertIn("All set.", self.text(io))
        self.assertEqual(io.asked[0][1][:3], ["claude", "codex", "grok"])  # the menu starts with the CLIs

    def test_a_cli_is_installed_and_logged_in_on_request(self):
        io = ScriptedIO(picks=["claude", welcome.DONE], confirms=[True, True])

        def find():
            if not io.ran:  # not there until the installer has run
                raise RuntimeError("not found")
            return "/usr/local/bin/claude"

        with patch("vision.brain.find_claude", side_effect=find), patch("shutil.which", side_effect=lambda n: "/usr/local/bin/claude" if n == "claude" else None), \
                patch.object(providers, "model_tabs", return_value=[("Claude", [("opus", "Opus", "")], "")]):
            ready = welcome.Wizard(io, config.load_config()).run()
        self.assertEqual(io.ran, ["curl -fsSL https://claude.ai/install.sh | bash", ["/usr/local/bin/claude"]])
        self.assertIn("Claude Code is installed.", self.text(io))
        self.assertEqual(ready, ["claude"])
        self.assertEqual(self.saved, [("opus", "high")])

    def test_declining_the_installer_points_at_the_page_and_reports_still_missing(self):
        io = ScriptedIO(picks=["grok", welcome.DONE], confirms=[True])
        ready = welcome.Wizard(io, config.load_config()).run()
        self.assertEqual(io.ran, [])
        self.assertIn("https://x.ai/cli", self.text(io))
        self.assertIn("Still can't find `grok`", self.text(io))
        self.assertIn("Nothing is ready yet", self.text(io))
        self.assertEqual((ready, self.saved), ([], []))

    def test_a_hosted_api_and_an_agent(self):
        io = ScriptedIO(picks=[welcome.HOSTED, welcome.AGENT, welcome.DONE],
                        answers=["openrouter", "https://openrouter.ai/api/v1", "OR_KEY", "a/b, c/d", "OpenRouter",
                                 "gemini", "gemini --experimental-acp", "Gemini"])
        with patch.object(clis, "local_model_ids", return_value=["a/b", "c/d"]):
            welcome.Wizard(io, config.load_config()).run()
        text = self.path.read_text()
        self.assertIn('api_key_env = "OR_KEY"', text)
        self.assertIn('models = ["a/b", "c/d"]', text)
        self.assertIn('command = ["gemini", "--experimental-acp"]', text)
        self.assertIn("OR_KEY isn't set in this shell", self.text(io))
        self.assertIn("`gemini` isn't on PATH here", self.text(io))
        self.assertEqual(sorted(n for n, p in providers.REGISTRY.items() if not p.builtin), ["gemini", "openrouter"])

    def test_first_run_only_when_nothing_is_ready_and_there_is_a_terminal(self):
        cfg = config.load_config()
        io = ScriptedIO(picks=[welcome.DONE])
        self.assertFalse(welcome.first_run(cfg, io, interactive=False))
        self.assertTrue(welcome.first_run(cfg, io, interactive=True))
        self.assertIn("Welcome to Vision", self.text(io))
        with patch.object(providers, "ready", return_value=True):
            self.assertFalse(welcome.first_run(cfg, ScriptedIO(), interactive=True))
        with patch.dict(os.environ, {"VISION_NO_SETUP": "1"}):
            self.assertFalse(welcome.first_run(cfg, ScriptedIO(), interactive=True))


if __name__ == "__main__":
    unittest.main()
