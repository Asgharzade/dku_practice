# Enhanced Jupyter Notebook Execution

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create an executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': ''}})
    except Exception as e:
        print(f'Error executing the notebook: {e}')
        raise
    
    # Optionally, save the executed notebook
    output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    return nb
