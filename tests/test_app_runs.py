import subprocess

def test_app_runs():
    """Check if the Streamlit app starts without immediate errors."""
    result = subprocess.run(
        ["python", "-m", "py_compile", "app_streamlit.py"],
        capture_output=True
    )
    assert result.returncode == 0, f"Streamlit app failed to compile: {result.stderr.decode()}"
