"""
Pybind11 example plugin
-----------------------

.. currentmodule:: scikit_build_example

.. autosummary::
    :toctree: _generate

    add
    subtract
"""
import numpy as np


def erode(array:np.array,mask:np.array, it:int , connectivity:int) -> np.array:
    """
    Erode a 3D array by 3X3 kernal    
    """


def dilate(array:np.array,mask:np.array, it:int , connectivity:int) -> np.array:
    """
    Dilate a 3D array by 3X3 kernal    
    """


def grid_cut(array:np.array,relabel:np.array, N:int, loop:int=-1) -> np.array:
    """
    Grid cut a voxelgrid with alpha-expansion  
    """


def grid_expansion(orient_array:np.array, unorient_array:np.array, sign_array:np.array, N:int) -> np.array:
    """
    Expansion the unorient region based on orient region
    """


def label_graph(array:np.array,mask2:array, int re_N, float t=1.0):
    """
    Extract label connectivity graph
    """


def label_graph_merge(array:np.array,sign:array, int re_N, float t):
    """
    Merge label cross block
    """


def inplace_label(array:np.array, label_map:dict):
    """
    Extract label connectivity graph
    """


def m3c_py(label:np.array, value:np.array):
    """
    M3C py api
    """


def is_bound(label:np.array):
    """
    Detecting bound block
    """
