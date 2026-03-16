import unittest

import torch

from trainer.models.hyvideo.models.transformers.ar_action_dc_hunyuanvideo_1_5_transformer import (
    DynamicChunkingModule,
)
from trainer.models.hyvideo.models.transformers.modules.dynamic_chunking.config import (
    DynamicChunkingConfig,
)
from trainer.models.hyvideo.models.transformers.modules.dynamic_chunking.dc import (
    compute_spatial_3d_nearest_boundary_idx,
)


class TestDynamicChunkingV1(unittest.TestCase):
    def _build_module(self) -> DynamicChunkingModule:
        config = DynamicChunkingConfig(
            enabled=True,
            downsample_factor=4.0,
            routing_module_type=["spatial_3d", None],
            dechunk_plug_back_mode=["nearest_3d", None],
            dechunk_smooth_mode=["spatial_kernel", None],
            use_ste=True,
        )
        return DynamicChunkingModule(
            hidden_size=16,
            config=config,
            temporal_boundary_threshold=0.5,
            temporal_min_chunk_frames=2,
            temporal_max_chunk_frames=6,
            temporal_target_chunk_frames=4,
        )

    def test_temporal_segments_cover_full_range(self):
        module = self._build_module()
        x = torch.randn(2, 20 * 6, 16)
        segments = module._predict_temporal_segments(
            hidden_states=x,
            num_frames=20,
            num_rows=2,
            num_cols=3,
        )
        self.assertGreaterEqual(len(segments), 1)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], 20)
        for start, end in segments:
            self.assertGreater(end, start)

    def test_chunk_dechunk_roundtrip_shape(self):
        module = self._build_module()
        x = torch.randn(1, 12 * 4, 16)
        chunked, segment_states, _ = module.chunk_with_temporal_segments(
            hidden_states=x,
            num_frames=12,
            num_rows=2,
            num_cols=2,
            mask=None,
        )
        restored = module.dechunk_with_temporal_segments(
            chunked_states=chunked,
            segment_states=segment_states,
            num_rows=2,
            num_cols=2,
            mask=None,
        )
        self.assertEqual(restored.shape, x.shape)

    def test_nearest_3d_index_bounds(self):
        num_frames, num_rows, num_cols = 4, 2, 3
        length = num_frames * num_rows * num_cols
        boundary_mask = torch.zeros(2, length, dtype=torch.bool)
        boundary_mask[0, [0, 5, 9, 23]] = True
        boundary_mask[1, [0, 6, 12, 23]] = True
        idx = compute_spatial_3d_nearest_boundary_idx(
            boundary_mask=boundary_mask,
            num_frames=num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        self.assertEqual(idx.shape, boundary_mask.shape)
        for batch_idx in range(boundary_mask.shape[0]):
            num_boundaries = int(boundary_mask[batch_idx].sum().item())
            self.assertTrue((idx[batch_idx] >= 0).all())
            self.assertTrue((idx[batch_idx] < num_boundaries).all())


if __name__ == "__main__":
    unittest.main()
