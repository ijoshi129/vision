from __future__ import annotations

import unittest

from vision.ui import _as_tabs, _locate, _picker_height, _picker_lines


class SingleSelectorTests(unittest.TestCase):
    def test_flat_options_render_as_single_selector(self):
        options = [
            ("codex", "Codex", "local coding agent"),
            ("claude", "Claude", "external assistant"),
            ("none", "None", "skip for now"),
        ]

        tabs = _as_tabs(options)
        tab, idx = _locate(tabs, "claude")
        rendered = "".join(text for _, text in _picker_lines("Choose provider", tabs, tab, idx, "claude"))

        self.assertEqual(tabs, [("", options, "")])
        self.assertEqual((tab, idx), (0, 1))
        self.assertEqual(_picker_height(tabs), 5)
        self.assertIn("Choose provider", rendered)
        self.assertIn("❯ 2 Claude", rendered)
        self.assertIn("(current)", rendered)
        self.assertNotIn("provider ·", rendered)
        self.assertNotIn("◉", rendered)
        self.assertNotIn("○", rendered)


if __name__ == "__main__":
    unittest.main()
