import pytest
import sys

def test_print_meerpower(meerpower):
    print ("meerpower dir: %s" % meerpower)
    sys.path.insert(1, meerpower)
    import Init
    import plot
