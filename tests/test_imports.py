import unittest

class TestImports(unittest.TestCase):
    def test_active_weasu_pipeline(self):
        """Test if we can import some modules from the package."""
        from activeweasul.plot import (
            plot_probs, plot_train_loss, PlotMixin
        )
        from activeweasul.query import (
            ActiveLearningQuery
        )
