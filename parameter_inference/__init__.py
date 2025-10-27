"""
Parameter inference modules for cosmology.
"""
from .inference import load_model, generate_samples
from .sampling import ParameterInference

__all__ = ['load_model', 'generate_samples', 'ParameterInference']

