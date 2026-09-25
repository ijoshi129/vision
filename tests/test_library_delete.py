import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision import library


class DeleteTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(library, "_session_logs", return_value=[])
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_deletes_upload_and_video_digest(self):
        clip = self.dir / "20260925-120000.mov"
        clip.write_bytes(b"x")
        clip.with_suffix(".json").write_text("{}")
        frames = self.dir / "20260925-120000-frames"
        frames.mkdir()
        (frames / "f1.jpg").write_bytes(b"x")
        self.assertTrue(library.delete(str(clip), self.dir, lambda p: True))
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_refuses_paths_the_library_does_not_list(self):
        outside = Path(tempfile.mkdtemp()) / "keep.png"
        outside.write_bytes(b"x")
        self.assertFalse(library.delete(str(outside), self.dir, lambda p: True))
        self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
