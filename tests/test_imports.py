def test_imports():
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import sklearn
    except ImportError as e:
        assert False, f"Missing dependency: {e}"
