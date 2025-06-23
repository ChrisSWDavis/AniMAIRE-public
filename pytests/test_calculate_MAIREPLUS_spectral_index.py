"""
Tests for the calculate_MAIREPLUS_spectral_index module.
"""

import unittest
import datetime as dt
from unittest.mock import patch, MagicMock

from AniMAIRE.maire_plus_tools.calculate_MAIREPLUS_spectral_index import calculate_MAIREPLUS_spectral_index


class TestCalculateMaireplusSpectralIndex(unittest.TestCase):
    """Test cases for calculate_MAIREPLUS_spectral_index function."""
    
    @patch('AniMAIRE.maire_plus_tools.calculate_MAIREPLUS_spectral_index.NeutronMonitor')
    @patch('AniMAIRE.maire_plus_tools.calculate_MAIREPLUS_spectral_index.calculate_spectral_index_for_target_ratio')
    def test_calculate_MAIREPLUS_spectral_index(self, mock_calculate_index, mock_neutron_monitor):
        """Test that calculate_MAIREPLUS_spectral_index correctly calculates the index."""
        # Setup mocks
        mock_nm1 = MagicMock()
        mock_nm1.altitude_in_km = 0.0
        mock_nm1.calculate_vertical_cutoff_rigidity.return_value = 0.67
        
        mock_nm2 = MagicMock()
        mock_nm2.altitude_in_km = 0.0
        mock_nm2.calculate_vertical_cutoff_rigidity.return_value = 2.8
        
        mock_neutron_monitor.side_effect = [mock_nm1, mock_nm2]
        mock_calculate_index.return_value = 2.4
        
        # Call function
        result = calculate_MAIREPLUS_spectral_index(
            neutron_monitor_1_location=(65.0, 25.0, 0.0),
            neutron_monitor_1_percentage_increase=1.05,
            neutron_monitor_2_location=(50.0, 5.0, 0.0),
            neutron_monitor_2_percentage_increase=1.0,
            datetime=dt.datetime(2000, 1, 1),
            kp_index=1.0
        )
        
        # Assert
        self.assertEqual(result, 2.4)
        mock_neutron_monitor.assert_any_call(latitude=65.0, longitude=25.0, altitude_in_km=0.0)
        mock_neutron_monitor.assert_any_call(latitude=50.0, longitude=5.0, altitude_in_km=0.0)
        mock_nm1.calculate_vertical_cutoff_rigidity.assert_called_once_with(datetime=dt.datetime(2000, 1, 1), kp_index=1.0)
        mock_nm2.calculate_vertical_cutoff_rigidity.assert_called_once_with(datetime=dt.datetime(2000, 1, 1), kp_index=1.0)
        mock_calculate_index.assert_called_once_with(
            target_ratio=1.05,
            cut_off_rigidity_1=0.67,
            altitude_in_km_1=0.0,
            cut_off_rigidity_2=2.8,
            altitude_in_km_2=0.0
        )


if __name__ == "__main__":
    unittest.main() 