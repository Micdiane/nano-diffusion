"""Check architecture math against Diffusers; real-weight CUDA checks live in scripts/."""
import unittest
import torch
from diffusers import SanaTransformer2DModel
from nanodiffusion.models.sana_transformer import SanaTransformer


class TransformerTests(unittest.TestCase):
    def test_checkpoint_layout_and_forward_against_reference(self):
        torch.manual_seed(7)
        reference = SanaTransformer2DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=8,
            num_layers=2, num_cross_attention_heads=2, cross_attention_head_dim=8,
            cross_attention_dim=16, caption_channels=12, sample_size=4,
        ).eval()
        native = SanaTransformer(dict(reference.config)).eval()
        native.load_state_dict(reference.state_dict(), strict=True)
        with torch.inference_mode():
            for batch in (1, 2):
                x, caption = torch.randn(batch, 4, 4, 3), torch.randn(batch, 5, 12)
                mask = torch.tensor([[1, 1, 1, 0, 0]]).expand(batch, -1)
                time = torch.full((batch,), 743.0)
                expected = reference(x, caption, time, encoder_attention_mask=mask).sample
                actual = native(x, time, caption, mask)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                # Masked caption tokens must not influence the result.
                caption[:, 3:] = 1000
                torch.testing.assert_close(native(x, time, caption, mask), actual, rtol=1e-5, atol=1e-6)

    def test_unsupported_architecture_fails_early(self):
        for config in ({'patch_size': 2}, {'guidance_embeds': True}, {'qk_norm': 'rms_norm'}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, 'Unsupported'):
                SanaTransformer(config)
