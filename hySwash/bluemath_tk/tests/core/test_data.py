import pytest
import pandas as pd
from bluemath_tk.core.data import normalize

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'A': [0, 1, 2, 3, 4],
        'B': [1, 30, 60, 180, 359],
        'C': [0, 100, 200, 300, 400]
    })
    ix_directional = ['B']
    scale_factor = {}
    return data, ix_directional, scale_factor

def test_normalize_no_scale_factor(sample_data):
    data, ix_directional, scale_factor = sample_data
    data_norm, scale_factor = normalize(data, ix_directional, scale_factor)
    print(data_norm)
    
    expected_data_norm = pd.DataFrame({
        'A': [0.0, 0.25, 0.5, 0.75, 1.0],
        'B': [0.0, 0.25, 0.5, 0.75, 1.0],
        'C': [0.0, 0.25, 0.5, 0.75, 1.0]
    })
    
    pd.testing.assert_frame_equal(data_norm, expected_data_norm)

