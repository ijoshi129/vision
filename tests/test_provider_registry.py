from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vision import config, providers
from vision.config import Config
from vision.ui import QuestionForm


def _cfg(*enabled: str, model: str = "opus") -> Config:
    cfg = Config()
    cfg.providers.enabled = list(enabled)
    cfg.brain.model = model
    return cfg


class RegistryTests(unittest.TestCase):
    def setUp(self):
        providers.forget_ready()
        self.addCleanup(providers.forget_ready)

    def ready(self, *names):
        return patch.object(providers, "ready", side_effect=lambda name, cfg=None: name in names)

    def test_model_tabs_follow_the_enabled_list_and_its_order(self):
        with self.ready("claude", "codex", "grok", "local"):
            tabs = providers.model_tabs(_cfg("local", "claude"))
        self.assertEqual([t[0] for t in tabs], ["Local", "Claude"])

    def test_a_provider_not_set_up_is_one_setup_row(self):
        with self.ready("claude"):
            tabs = dict((t[0], t[1]) for t in providers.model_tabs(_cfg("claude", "codex")))
            phone = [t[0] for t in providers.model_tabs(_cfg("claude", "codex"), setup_rows=False)]
        self.assertEqual(len(tabs["Codex"]), 1)
        value, label, desc = tabs["Codex"][0]
        self.assertEqual((value, label), ("setup:codex", "Not set up"))
        self.assertIn("npm i -g @openai/codex", desc)
        self.assertIn("isn't set up here yet", providers.setup_note(value))
        self.assertIsNone(providers.setup_note("opus"))
        self.assertEqual(phone, ["Claude"])  # the phone can't install anything: left out

    def test_unavailable_reason(self):
        cfg = _cfg("claude", "grok")
        with self.ready("claude"):
            self.assertIsNone(providers.unavailable_reason("claude", cfg))
            self.assertIn("switched off", providers.unavailable_reason("codex", cfg))
            self.assertIn("isn't set up", providers.unavailable_reason("grok", cfg))

    def test_settle_moves_a_fresh_install_to_what_it_has(self):
        cfg = _cfg("claude", "local", model="opus")
        with self.ready("local"), patch("vision.models.provider_default", return_value="qwen3.6"):
            note = providers.settle(cfg)
        self.assertEqual(cfg.brain.model, "qwen3.6")
        self.assertIn("Claude isn't set up here yet", note)
        self.assertIn("config.toml unchanged", note)

    def test_settle_leaves_a_working_model_alone(self):
        cfg = _cfg("claude", model="opus")
        with self.ready("claude"):
            self.assertEqual(providers.settle(cfg), "")
        self.assertEqual(cfg.brain.model, "opus")

    def test_settle_with_nothing_set_up_says_how(self):
        cfg = _cfg("claude", "codex", model="opus")
        with self.ready():
            note = providers.settle(cfg)
        self.assertEqual(cfg.brain.model, "opus")
        self.assertIn("No provider is set up yet", note)
        self.assertIn("claude.ai/install.sh", note)

    def test_create_brain_takes_the_class_from_the_registry(self):
        from vision.brain import create_brain

        with patch("vision.clis.refresh_models_soon"), patch("vision.local.LocalBrain.__init__", return_value=None) as init:
            brain = create_brain(config.BrainConfig(model="qwen3.6", effort=""))
        self.assertEqual(type(brain).__name__, "LocalBrain")
        init.assert_called_once()


class ConfigTests(unittest.TestCase):
    def test_enabled_providers_load_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text('[providers]\nenabled = ["local", "nope", "claude", "local"]\n', encoding="utf-8")
            with patch.object(config, "CONFIG_PATH", path), patch.object(config, "ensure_dirs"):
                self.assertEqual(config.load_config().providers.enabled, ["local", "claude"])
                config.save_enabled_providers(["codex"])
                self.assertEqual(config.load_config().providers.enabled, ["codex"])

    def test_default_is_every_provider(self):
        self.assertEqual(Config().providers.enabled, ["claude", "codex", "grok", "local"])


class PreselectedFormTests(unittest.TestCase):
    def test_ticks_start_on(self):
        form = QuestionForm([{"question": "Which?", "header": "P", "multiSelect": True, "preselected": [0, 2],
                              "options": [{"label": "A"}, {"label": "B"}, {"label": "C"}]}])
        form.toggle(0)  # untick A
        self.assertTrue(form.confirm())
        self.assertEqual(form.answers(), {"Which?": "C"})

    def test_enter_ticks_and_submit_finishes(self):
        form = QuestionForm([{"question": "Which?", "header": "P", "multiSelect": True, "preselected": [0], "other": False,
                              "options": [{"label": "A"}, {"label": "B"}]}])
        self.assertEqual(form.rows, 3)  # A, B, Submit: no ✎ row
        self.assertFalse(form.enter())  # untick A
        form.down()
        self.assertFalse(form.enter())  # tick B
        form.down()
        self.assertEqual(form.idx, form.submit)
        self.assertTrue(form.enter())
        self.assertEqual(form.answers(), {"Which?": "B"})

    def test_typed_text_in_a_multi_select_waits_for_submit(self):
        form = QuestionForm([{"question": "Which?", "header": "P", "multiSelect": True, "options": [{"label": "A"}]}])
        self.assertFalse(form.enter("my own"))
        self.assertEqual(form.idx, form.submit)
        self.assertTrue(form.enter())
        self.assertEqual(form.answers(), {"Which?": "my own"})

    def test_single_select_enter_still_confirms(self):
        form = QuestionForm([{"question": "One?", "header": "P", "options": [{"label": "A"}, {"label": "B"}]}])
        form.down()
        self.assertTrue(form.enter())
        self.assertEqual(form.answers(), {"One?": "B"})


if __name__ == "__main__":
    unittest.main()
