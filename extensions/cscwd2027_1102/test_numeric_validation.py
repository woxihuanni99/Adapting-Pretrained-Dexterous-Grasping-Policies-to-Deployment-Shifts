import unittest
from numeric_validation import assert_record_close

class NumericValidationTests(unittest.TestCase):
    def test_quantile_roundoff(self):
        self.assertGreater(assert_record_close({'CI':[23.1+1.5e-10,33.]},{'CI':[23.1,33.]}),0)
    def test_material_number_error(self):
        with self.assertRaises(AssertionError):assert_record_close(23.1001,23.1)
    def test_counts_are_exact(self):
        with self.assertRaises(AssertionError):assert_record_close({'n':4},{'n':3})
    def test_structure_is_exact(self):
        with self.assertRaises(AssertionError):assert_record_close([1.], [1.,2.])
    def test_labels_are_exact(self):
        with self.assertRaises(AssertionError):assert_record_close('LF','PCL')
    def test_nan_fails(self):
        with self.assertRaises(AssertionError):assert_record_close(float('nan'),1.)

if __name__=='__main__':unittest.main()
