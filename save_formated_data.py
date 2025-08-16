import scipy.io
import pandas as pd
import numpy as np
import os
from pathlib import Path

def mat_to_parquet(mat_file_path, parquet_file_path=None):
    """
    Load a .mat file (connecctvity or peprocessed imaging data) and save it as a .parquet file.
    Parquet format is preferable at multiple off-the-shelf models 
    Parameters:
    mat_file_path (str): Path to the input .mat file
    parquet_file_path (str, optional): Path for the output .parquet file. 
    If None, uses the same name as input with .parquet extension
    """
    
    # Load the .mat file
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        print(f"Successfully loaded {mat_file_path}")
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return
    
    # Remove MATLAB metadata keys
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    # Convert to DataFrame
    df_dict = {}
    
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray):
            # Handle different array dimensions
            if value.ndim == 1:
                df_dict[key] = value
            elif value.ndim == 2:
                # If it's a 2D array, check if it's a single column/row
                if value.shape[0] == 1:
                    df_dict[key] = value.flatten()
                elif value.shape[1] == 1:
                    df_dict[key] = value.flatten()
                else:
                    # For true 2D arrays, convert each column to a separate series
                    for i in range(value.shape[1]):
                        df_dict[f"{key}_col_{i}"] = value[:, i]
            else:
                # For higher dimensional arrays, flatten them
                df_dict[key] = value.flatten()
        else:
            # For scalar values
            df_dict[key] = [value]
    
    # Create DataFrame
    df = pd.DataFrame(df_dict)

    # Generate output filename if not provided
    if parquet_file_path is None:
        mat_path = Path(mat_file_path)
        parquet_file_path = mat_path.with_suffix('.parquet')
    
    # Save as Parquet
    df.to_parquet(parquet_file_path, index=False)