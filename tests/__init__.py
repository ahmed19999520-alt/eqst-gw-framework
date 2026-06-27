import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
os.makedirs(TEST_DATA_DIR, exist_ok=True)

TOLERANCE_RELATIVE = 1.0e-6
TOLERANCE_ABSOLUTE = 1.0e-10

USE_MOCK_DATA = True
SKIP_SLOW_TESTS = os.environ.get('EQST_SKIP_SLOW', '1') == '1'
SKIP_DOWNLOAD_TESTS = os.environ.get('EQST_SKIP_DOWNLOAD', '1') == '1'