from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock, patch

from vision.tts import StreamingSpeaker
from vision.ui import ChatScreen, _ReplyEntry, _Reveal


def plain(rows):
    return "\n".join("".join(fragment[1] for fragment in row) for row in rows)


class TextRevealTests(unittest.TestCase):
    def test_incoming_letter_has_intermediate_frames(self):
        reveal = _Reveal("Hello world\n", 11)
        frames = [reveal.cut(2 + fraction) for fraction in (0.1, 0.3, 0.6, 0.9)]
        self.assertEqual([plain(frame) for frame in frames], ["Hel"] * 4)
        self.assertEqual(len({frame[-1][-1][0] for frame in frames}), 4)
        self.assertTrue(all(frame[-1][-2] == ("", "He") for frame in frames))
        self.assertEqual(plain(reveal.cut(3)), "Hel")
        self.assertEqual(reveal.cut(reveal.total), reveal.rows)
        self.assertEqual(reveal.rows, [[("", "Hello world")]])

    def test_fade_preserves_wrapping_and_completed_styles(self):
        reveal = _Reveal("\x1b[1mHi\x1b[0m there\nnext\n", 13)
        self.assertEqual(plain(reveal.cut(0)), "")
        self.assertEqual(plain(reveal.cut(2.5)), "Hi t")
        self.assertEqual(reveal.cut(2.5)[0][0], ("bold", "Hi"))
        self.assertEqual(plain(reveal.cut(7.5)), "Hi there\nn")
        for position in (0.1, 1.5, 3.2, 7.5, 10.9):
            self.assertEqual(
                plain(reveal.cut(position)), plain(reveal.cut(math.ceil(position))),
            )

    def test_progress_change_during_render_cannot_leave_stale_cached_text(self):
        entry = _ReplyEntry(markdown=False)
        entry.buf, entry.shown, entry.ended = "abcdef", 1, entry.started
        reveal = entry.full(80)
        original_cut = reveal.cut

        def advance_while_drawing(position):
            entry.shown = 3
            entry.invalidate()
            return original_cut(position)

        with patch.object(reveal, "cut", side_effect=advance_while_drawing):
            entry.lines(80)
        self.assertTrue(plain(entry.lines(80)).endswith("abc"))

    def test_voice_clock_keeps_sub_character_precision(self):
        speaker = StreamingSpeaker.__new__(StreamingSpeaker)
        speaker._audible = lambda: 0.125
        speaker._chunks = [[20, 2.0, 0.0, True]]
        speaker._est = speaker._blend = None
        speaker._floor = 0
        self.assertEqual(speaker.spoken_position, 1.25)
        self.assertEqual(speaker.spoken, 1)

    def test_pacer_animates_between_letters_without_running_ahead_of_voice(self):
        entry = _ReplyEntry(markdown=False)
        entry.buf = "abcdefghij" * 3
        now, frames = [0.0], []
        entry.gate = lambda: min(30.0, now[0] * 10)

        def sleep(delay):
            now[0] += delay
            if now[0] >= 3:
                entry.finished = True
            if now[0] > 4:
                self.fail("reveal never finished")

        def redraw():
            frames.append((now[0], entry.shown))

        screen = ChatScreen.__new__(ChatScreen)
        screen.reveal_cps, screen.max_reveal_cps, screen.catch_up = 42, 72, 2.5
        screen._width = lambda: 80
        screen._settle = Mock()
        screen.app = types.SimpleNamespace(invalidate=redraw)
        clock = types.SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
        with patch("vision.ui.time", clock):
            screen._pace(entry)

        self.assertGreater(len(frames), 150, "slow speech still needs intermediate visual frames")
        self.assertTrue(any(position != int(position) for _, position in frames))
        self.assertTrue(all(position <= t * 10 + 1e-8 for t, position in frames))
        self.assertLess(max(b - a for (_, a), (_, b) in zip(frames, frames[1:])), 0.2)
        self.assertEqual(entry.shown, 30)
        screen._settle.assert_called_once_with(entry)


if __name__ == "__main__":
    unittest.main()
