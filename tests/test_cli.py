import contextlib
import io
import unittest
from nanodiffusion.cli import main


class CLITests(unittest.TestCase):
    def test_requires_real_checkpoint(self):
        for argv in ([], ['--checkpoint', '/no-such-checkpoint']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 2)
