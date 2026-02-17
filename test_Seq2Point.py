import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import logging
import argparse
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scale import scale
from sliding_window_seq2point import sliding_window_seq2point
from Seq2Point import Seq2Point

def main():
    parser = argparse.ArgumentParser(description="Test Seq2Point CNN for Load Disaggregation")
    parser.add_argument("--window_size", type=int, default=100, help="Window size (Seq2Point input length)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_path", type=str, default="best_seq2point_model.pth", help="Path to saved model")
    
    parser.add_argument("--input_channel", type=str, default="net_load", 
                        help="Name of the aggregate input channel (e.g., 'net_load', 'demand')")
    parser.add_argument("--target_channel", type=str, default="shiftable_loads", 
                        help="Name of the target appliance channel (e.g., 'shiftable_loads', 'base_loads')")

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # Load Data
    logging.info("Loading and scaling test data...")
    # scale returns: train_scaled, val_scaled, test_scaled, node_idx, ctx_idx
    node_names = ['shiftable_loads', 'base_loads', 'demand', 'generation', 'net_load']
    
    _, _, test_scaled, node_idx, ctx_idx = scale()
    
    try:
        input_pos = node_names.index(args.input_channel)
        target_pos = node_names.index(args.target_channel)
        
        input_idx = node_idx[input_pos]
        target_idx = node_idx[target_pos]
        
        logging.info(f"Using Input Channel: '{args.input_channel}' (Index: {input_idx})")
        logging.info(f"Using Target Channel: '{args.target_channel}' (Index: {target_idx})")
        
    except ValueError as e:
        logging.error(f"Invalid channel name specified. Available channels: {node_names}. Error: {e}")
        return
    
    # Create Windows
    X_test, y_test = sliding_window_seq2point(test_scaled, args.window_size, input_idx, target_idx)
    logging.info(f"Test Data Shape: X={X_test.shape}, y={y_test.shape}")
    
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    model = Seq2Point(window_size=args.window_size).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Loaded model from {args.model_path}")
    else:
        logging.error(f"Model file {args.model_path} not found!")
        return

    model.eval()
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    total_mse = 0.0
    total_mae = 0.0
    predictions = []
    ground_truth = []
    
    logging.info("Evaluating...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            output = model(X_batch)
            
            mse = criterion(output, y_batch)
            mae = mae_criterion(output, y_batch)
            
            total_mse += mse.item() * X_batch.size(0)
            total_mae += mae.item() * X_batch.size(0)
            
            predictions.append(output.cpu().numpy())
            ground_truth.append(y_batch.cpu().numpy())
            
    avg_mse = total_mse / len(test_loader.dataset)
    avg_mae = total_mae / len(test_loader.dataset)
    
    logging.info(f"Test MSE: {avg_mse:.6f}")
    logging.info(f"Test MAE: {avg_mae:.6f}")
    
    # Concatenate results
    predictions = np.concatenate(predictions, axis=0).flatten()
    ground_truth = np.concatenate(ground_truth, axis=0).flatten()
    
    # Plot first 500 samples
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truth[:500], label='Ground Truth', color='black', linewidth=1)
    plt.plot(predictions[:500], label='Prediction', color='red', alpha=0.7, linewidth=1)
    plt.title(f'Seq2Point Load Disaggregation ({args.target_channel})')
    plt.xlabel('Time Step')
    plt.ylabel('Scaled Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig('seq2point_test_result.png', dpi=300)
    logging.info("Saved plot to seq2point_test_result.png")

if __name__ == "__main__":
    main()
