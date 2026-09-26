import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from nanodiffusion.cli import main


class CLITests(unittest.TestCase):
    def test_offline_toy_saves_loadable_frames_and_sampling_metadata(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                target = Path(directory) / "sample.pt"
                self.assertEqual(main(["--height", "8", "--width", "12", "--frames", "2",
                                       "--steps", "2", "--seed", "23", "--output", str(target), "--quiet"]), 0)
                output = torch.load(target, weights_only=True)
                metadata = json.loads(target.with_suffix(".json").read_text())
                self.assertEqual(output["frames"].shape, (1, 2, 8, 12, 3))
                self.assertEqual(output["latents"].shape, (1, 4, 2, 2, 3))
                self.assertEqual(metadata, output["metadata"])
                self.assertTrue(metadata["untrained"])
                self.assertEqual(metadata["sampling"]["seed"], 23)
                self.assertEqual(metadata["num_model_calls"], 4)
        finally:
            torch.set_num_threads(threads)

    def test_invalid_wan_arguments_fail_without_loading_optional_dependencies(self):
        for argv in (["--model", "wan"],
                     ["--model", "wan", "--checkpoint", "/unread", "--attention", "sage"],
                     ["--steps", "0"], ["--output", "sample.png"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
