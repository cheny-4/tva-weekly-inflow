from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = pd.read_csv(p.ROOT / 'examples/weekly_inputs.csv')
        cls.metadata = p.read_json(p.ROOT / 'examples/forcing_metadata.json')

    def test_original_masks_and_split_counts(self):
        a = p.load_arrays()
        self.assertEqual(a['dynamic'].shape, (1816, 52, 3))
        for name, code in p.SPLITS.items():
            t, b = np.where(a['eligible'] & (a['split'][:, None] == code))
            self.assertEqual(len(t), {'train': 50084, 'validation': 9868, 'test': 15025}[name])
            self.assertTrue(a['target_valid_all7'][t,b].all())
            self.assertTrue(np.isfinite(a['dynamic'][t[:,None]+np.arange(-12,1), b[:,None]]).all())

    def test_saved_checkpoint_is_selected_epoch_and_shape(self):
        model, ck, cfg, sc = p.checkpoint_model()
        self.assertEqual(ck['epoch'], 19)
        self.assertEqual(sum(x.numel() for x in model.parameters()), 72884)
        self.assertEqual(cfg['basin_ids'], p.read_json(p.BASE / 'data/site_head_mapping.json')['basin_ids'])

    def test_inference_and_volume_conversion_match_existing_site_weeks(self):
        result, _ = p.infer_frame(self.inputs, self.metadata)
        self.assertEqual(len(result), 8)
        self.assertTrue((result.status == 'predicted').all())
        reference = pd.read_csv(p.RUN / 'test_paired_weekly_predictions.csv')
        paired = result.merge(reference,on=['basin_id','start','end','days'],validate='one_to_one',suffixes=('','_saved'))
        self.assertEqual(len(paired), 8)
        area = pd.read_csv(p.BASE / 'data/static_attributes.csv').set_index('elf_id').loc[paired.basin_id,'area_km2'].to_numpy()
        saved_depth = paired.predicted_volume_m3_saved / (area * 7000)
        np.testing.assert_allclose(paired.predicted_mean_mm_day,saved_depth,rtol=0,atol=5e-5)
        np.testing.assert_allclose(paired.predicted_volume_m3,paired.predicted_mean_mm_day*area*7000)

    def test_elf_observations_are_not_inference_inputs(self):
        x = self.inputs.copy();x['observed_volume_m3'] = 1e99
        before, _ = p.infer_frame(self.inputs, self.metadata)
        after, _ = p.infer_frame(x, self.metadata)
        np.testing.assert_array_equal(before.predicted_mean_mm_day,after.predicted_mean_mm_day)

    def test_runoff_definition_mismatch_is_rejected(self):
        bad = {**self.metadata,'runoff_definition':'surface_runoff_only'}
        with self.assertRaisesRegex(ValueError,'runoff_definition'):
            p.infer_frame(self.inputs,bad)

    def test_duplicate_weeks_and_unknown_heads_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique Monday'):
            p.infer_frame(pd.concat([self.inputs,self.inputs.iloc[:1]]),self.metadata)
        unknown=self.inputs.copy();unknown.loc[unknown.basin_id=='ELF13','basin_id']='ELF99'
        with self.assertRaisesRegex(ValueError,'No trained output heads'):
            p.infer_frame(unknown,self.metadata)

    def test_gap_is_flagged_and_not_filled(self):
        x=self.inputs[self.inputs.basin_id=='ELF13'].reset_index(drop=True).drop(index=5)
        result,_=p.infer_frame(x,self.metadata)
        self.assertEqual(len(result),3)
        self.assertTrue((result.status=='invalid_history_or_input').all())
        self.assertTrue(result.predicted_volume_m3.isna().all())

    def test_baseline_and_existing_outputs_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):p.new_output(p.BASE / 'new_results')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):p.new_output(folder)

    def test_metrics_use_paired_valid_rows_and_sign_convention(self):
        m=p.hydrologic_metrics([1,2,3,np.nan],[2,4,6,0])
        self.assertEqual(m['n'],3)
        self.assertAlmostEqual(m['pbias_percent'],100.)
        self.assertAlmostEqual(p.hydrologic_metrics([1,2,3],[1,2,3])['nse'],1.)


if __name__=='__main__':unittest.main()
