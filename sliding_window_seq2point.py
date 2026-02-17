import numpy as np

def sliding_window_seq2point(data_array, window_size, input_col_idx, target_col_idx):
    """
    data_array: np.array of shape (samples, features)
    window_size: int, length of input sequence
    input_col_idx: int, index of the aggregate signal (e.g. net_load)
    target_col_idx: int, index of the target appliance (e.g. shiftable_loads)
    
    Returns:
    X: (Batch, 1, Window) - Input windows
    y: (Batch, 1) - Target value at the midpoint of each window
    """
    X_list = []
    y_list = []
    
    num_samples = len(data_array)
    padding = window_size // 2
    
    # Extract columns
    input_data = data_array[:, input_col_idx]
    target_data = data_array[:, target_col_idx]
    
    # Loop over the data
    # We stop at num_samples - window_size to ensure we have a full window
    for i in range(num_samples - window_size + 1):
        window = input_data[i : i + window_size]
        
        # Target is the midpoint of the window
        mid_point = i + padding
        target = target_data[mid_point]
        
        X_list.append(window)
        y_list.append(target)
        
    X = np.array(X_list)
    # Reshape for Conv1D: (Batch, Channels, Length)
    # Here Channels=1
    X = X[:, np.newaxis, :] 
    
    y = np.array(y_list)
    y = y[:, np.newaxis] # (Batch, 1)
    
    return X, y
