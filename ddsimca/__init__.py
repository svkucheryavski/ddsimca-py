"""
DD-SIMCA: one-class classification method
"""

__version__ = "1.0.3"
__author__ = "Sergey Kucheryavskiy"
__email__ = "svkucheryavski@gmail.com"

from .ddsimca import DDSIMCA, DDSIMCARes, ddsimca, get_distparams, get_limits, process_members, process_strangers

__all__ = ["DDSIMCA", "DDSIMCARes", "ddsimca", "get_distparams", "get_limits", "process_members", "process_strangers"]
