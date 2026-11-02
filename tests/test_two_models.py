import unittest
import numpy as np
import pandas as pd
import pipeline as p


class TwoModelContractTests(unittest.TestCase):
    def test_surface_checkpoint_static_shape_and_epoch(self):
        model, ck, cfg, sc = p.checkpoint_model(model_id='surface_runoff_static8')
        self.assertEqual(ck['epoch'], 20)
        self.assertEqual(sum(v.numel() for v in model.parameters()), 74932)
        self.assertEqual(len(sc['static_names']), 9)
        self.assertEqual(model.lstm.input_size, 12)

    def test_surface_inference_reproduces_saved_rows(self):
        x = pd.read_csv(p.ROOT / 'examples/weekly_inputs_surface_static8.csv')
        meta = p.read_json(p.ROOT / 'examples/forcing_metadata_surface_static8.json')
        result, _ = p.infer_frame(x, meta, model_id='surface_runoff_static8')
        base, run, _ = p.model_paths('surface_runoff_static8')
        saved = pd.read_csv(run / 'test_paired_weekly_predictions.csv')
        g = result.merge(saved, on=['basin_id', 'start', 'end', 'days'], validate='one_to_one', suffixes=('', '_saved'))
        self.assertEqual(len(g), 8)
        area = pd.read_csv(base / 'data/static_attributes.csv').set_index('elf_id').loc[g.basin_id, 'area_km2'].to_numpy()
        np.testing.assert_allclose(g.predicted_mean_mm_day, g.predicted_volume_m3_saved / (area * 7000), rtol=0, atol=5e-5)
        self.assertNotIn('corrected_volume_m3', result)

    def test_cross_version_metadata_and_checkpoints_are_rejected(self):
        meta = p.read_json(p.ROOT / 'examples/forcing_metadata_surface_static8.json')
        with self.assertRaisesRegex(ValueError, 'runoff_definition'):
            p.validate_metadata(meta, 'total_runoff')
        with self.assertRaisesRegex(ValueError, 'runoff_definition'):
            p.validate_metadata(p.read_json(p.ROOT / 'examples/forcing_metadata.json'), 'surface_runoff_static8')
        with self.assertRaisesRegex(ValueError, 'Checkpoint does not match'):
            p.checkpoint_model(p.RUN / 'best_model.pt', model_id='surface_runoff_static8')

    def test_spatial_mapping_retains_source_groups_and_training_masks(self):
        a, b = p.load_arrays('total_runoff'), p.load_arrays('surface_runoff_static8')
        np.testing.assert_array_equal(a['eligible'], b['eligible'])
        np.testing.assert_array_equal(a['dynamic'][:, :, :2], b['dynamic'][:, :, :2])
        np.testing.assert_array_equal(a['static'], b['static'][:, :1])
        m = pd.read_csv(p.ROOT / 'spatial/site_region_mapping.csv')
        self.assertEqual(m.source_region_id.nunique(), 43)
        self.assertEqual(len(m), 52)
        self.assertEqual(m.set_index('basin_id').loc['ELF50', 'source_region_id'], 'SOURCE_BEAR_WILSON_PICKWICK')

    def test_both_artifact_roots_are_protected(self):
        for model in p.REGISTRY:
            with self.assertRaises(ValueError):
                p.new_output(p.model_paths(model)[0] / 'new_output')


if __name__ == '__main__':
    unittest.main()
