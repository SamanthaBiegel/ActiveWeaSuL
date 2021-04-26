import unittest

class TestImports(unittest.TestCase):
    def test_active_weasu_pipeline(self):
        """Test if we can import some modules from the package."""
        from activeweasul.active_weasul import ActiveWeaSuLPipeline
        from activeweasul.datasets import CustomTensorDataset
        from activeweasul.discriminative_model import DiscriminativeModel
        from activeweasul.experiments import (
            process_exp_dict, process_exp_dict, add_baseline, plot_metrics
        )
        from activeweasul.lf_utils import analyze_lfs, apply_lfs
        from activeweasul.label_model import LabelModel
        from activeweasul.logisticregression import LogisticRegression
        from activeweasul.performance import PerformanceMixin
        from activeweasul.plot import plot_probs, plot_train_loss, PlotMixin
        from activeweasul.query import ActiveLearningQuery
        from activeweasul.synthetic_data import SyntheticDataGenerator
        from activeweasul.visualrelation import crop_img_arr
        from activeweasul.vr_utils import balance_dataset
