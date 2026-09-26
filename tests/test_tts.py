import unittest

from vision.tts import select_reply_voice


PHOENIX_VOICES = [
    "phoenix-conversational",
    "phoenix-expressive",
    "phoenix-reassuring",
    "phoenix-london",
]


class VoiceStyleSelectionTests(unittest.TestCase):
    def test_keeps_non_phoenix_voice(self):
        self.assertEqual(select_reply_voice("narrator", "Nice one!", PHOENIX_VOICES), "narrator")

    def test_uses_expressive_reference_for_excited_reply(self):
        self.assertEqual(
            select_reply_voice("phoenix-conversational", "Perfect! We got it!", PHOENIX_VOICES),
            "phoenix-expressive",
        )

    def test_reassurance_beats_exclamation(self):
        self.assertEqual(
            select_reply_voice("phoenix-conversational", "Don't worry! We'll sort it!", PHOENIX_VOICES),
            "phoenix-reassuring",
        )

    def test_code_exclamation_is_not_excitement(self):
        self.assertEqual(
            select_reply_voice("phoenix-conversational", "Change it to a != b and rerun.", PHOENIX_VOICES),
            "phoenix-conversational",
        )

    def test_apology_is_reassuring(self):
        self.assertEqual(
            select_reply_voice("phoenix-conversational", "Sorry, my mistake. Fixed now.", PHOENIX_VOICES),
            "phoenix-reassuring",
        )

    def test_falls_back_when_a_variant_is_not_installed(self):
        self.assertEqual(
            select_reply_voice("phoenix-conversational", "Nice one!", ["phoenix-conversational"]),
            "phoenix-conversational",
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(__import__("sys").platform == "win32", "the Windows path of _quiet")
class QuietWindowsTests(unittest.TestCase):
    def test_streams_and_std_handles_come_back(self):
        import ctypes
        import sys
        from ctypes import wintypes

        from vision.tts import _quiet

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        std_out = wintypes.DWORD(-11).value
        stdout, stderr, handle = sys.stdout, sys.stderr, kernel32.GetStdHandle(std_out)
        with _quiet():
            print("hidden")
            self.assertIsNot(sys.stdout, stdout)
            self.assertNotEqual(kernel32.GetStdHandle(std_out), handle)  # children inherit NUL meanwhile
        self.assertIs(sys.stdout, stdout)
        self.assertIs(sys.stderr, stderr)
        self.assertEqual(kernel32.GetStdHandle(std_out), handle)
