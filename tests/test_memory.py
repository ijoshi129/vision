import tempfile
import unittest
from pathlib import Path
from unittest import mock


class MemoryTests(unittest.TestCase):
    def setUp(self):
        from vision import memory

        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name) / "memory"
        self.patches = [mock.patch.object(memory, "MEMORY_DIR", d), mock.patch.object(memory, "MEMORY_FILE", d / "MEMORY.md")]
        for p in self.patches:
            p.start()
        self.memory = memory

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_remember_creates_file_with_dated_lines(self):
        m = self.memory
        self.assertEqual(m.facts(), [])
        line = m.remember("  the user's   laptop is a ThinkPad ")
        self.assertRegex(line, r"^- \d{4}-\d{2}-\d{2}: the user's laptop is a ThinkPad$")
        m.remember("no sudo for Vision")
        self.assertTrue(m.read().startswith("# Vision's memory"))
        self.assertEqual(len(m.facts()), 2)
        with self.assertRaises(ValueError):
            m.remember("   ")

    def test_forget_drops_matching_lines_only(self):
        m = self.memory
        m.remember("likes tea")
        m.remember("Likes coffee in the morning")
        m.remember("uses Fedora")
        self.assertEqual(len(m.forget("LIKES")), 2)
        self.assertEqual([ln.split(": ", 1)[1] for ln in m.facts()], ["uses Fedora"])
        self.assertEqual(m.forget("nothing like this"), [])
        with self.assertRaises(ValueError):
            m.forget("")

    def test_prompt_section_shows_facts_and_rules(self):
        m = self.memory
        self.assertIn("empty so far", m.prompt_section())
        m.remember("calls the user boss")
        sec = m.prompt_section()
        self.assertIn("calls the user boss", sec)
        self.assertIn(str(m.MEMORY_FILE), sec)
        self.assertIn("/remember", sec)
        self.assertNotIn("# Vision's memory", sec)  # header stays out of the prompt

    def test_prompt_section_truncates_huge_files(self):
        m = self.memory
        big = "\n".join(f"- 2026-01-01: fact number {i} " + "x" * 100 for i in range(200))
        sec = m.prompt_section(big)
        self.assertIn("consolidate", sec)
        self.assertLess(len(sec), m.MAX_PROMPT_CHARS + 1500)

    def test_persona_carries_memory_for_both_brains(self):
        from vision.persona import system_prompt

        for provider in ("claude", "codex"):
            p = system_prompt(False, tools=["Bash"], provider=provider, memory="- 2026-09-16: the cat is called Pixel")
            self.assertIn("the cat is called Pixel", p)
            self.assertIn("MEMORY:", p)
            self.assertLess(p.index("Today is"), p.index("MEMORY:"))

    def test_codex_workspace_write_gets_memory_dir_as_writable_root(self):
        from vision.codex import CodexBrain, toml_str
        from vision.config import BrainConfig

        cfg = BrainConfig()
        cfg.codex.sandbox = "workspace-write"
        with mock.patch("vision.codex.find_codex", return_value="/fake/codex"):
            brain = CodexBrain(cfg, voice_mode=False)
        cmd = brain._command()
        self.assertIn(f"sandbox_workspace_write.writable_roots=[{toml_str(str(self.memory.MEMORY_DIR))}]", cmd)
        cfg.codex.sandbox = "read-only"
        self.assertFalse(any("writable_roots" in a for a in brain._command()))


if __name__ == "__main__":
    unittest.main()
