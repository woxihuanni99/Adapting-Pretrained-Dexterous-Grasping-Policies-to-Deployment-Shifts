"""Compare exact record structure and allow only negligible float roundoff."""
import math

ABS_TOL = 1e-8
REL_TOL = 1e-10

def assert_record_close(actual, expected, path='record'):
    """Return the largest float error; counts, labels and structure stay exact."""
    if type(actual) is not type(expected):
        raise AssertionError(f'{path}: type mismatch {type(actual).__name__} != {type(expected).__name__}')
    if isinstance(expected, float):
        if not (math.isfinite(actual) and math.isfinite(expected)):
            raise AssertionError(f'{path}: non-finite number')
        if not math.isclose(actual, expected, abs_tol=ABS_TOL, rel_tol=REL_TOL):
            raise AssertionError(f'{path}: {actual!r} != {expected!r}; absolute difference {abs(actual-expected):.3g}')
        return abs(actual-expected)
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'{path}: dictionary keys differ')
        return max((assert_record_close(actual[k], expected[k], f'{path}.{k}') for k in expected), default=0.)
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{path}: lengths differ')
        return max((assert_record_close(a,b,f'{path}[{i}]') for i,(a,b) in enumerate(zip(actual,expected))),default=0.)
    if actual != expected:
        raise AssertionError(f'{path}: {actual!r} != {expected!r}')
    return 0.
