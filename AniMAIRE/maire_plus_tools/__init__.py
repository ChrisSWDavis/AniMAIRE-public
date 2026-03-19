"""
Tools module for MairePlus software.
"""
 
# Import your tools here as you create them
from .neutron_monitor import NeutronMonitor
from .calculate_spectral_index import calculate_spectral_index_for_target_ratio

# Important: keep this as a *module* attribute (not a function) so that
# `unittest.mock.patch('AniMAIRE.maire_plus_tools.calculate_MAIREPLUS_spectral_index.*')`
# resolves consistently across Python versions.
from . import calculate_MAIREPLUS_spectral_index as calculate_MAIREPLUS_spectral_index

# Optional convenient alias (function), without shadowing the module name above.
calculate_MAIREPLUS_spectral_index_fn = (
    calculate_MAIREPLUS_spectral_index.calculate_MAIREPLUS_spectral_index
)
# from .your_tool import YourTool 
