import sys
import logging
import numpy as np

from scale import scale
from sliding_window_seq2point import sliding_window_seq2point
from Seq2Point import Seq2Point

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def train(args):
    # logging
    setup_logging(args.log_file)
    logging.info(f"Starting training with args: {args}")

    # Scale the data. 
    try:
        logging.info("Scaling info ...")
        # scale returns: train_scaled, val_scaled, test_scaled, node_idx, ctx_idx
        # node_idx corresponds to these columns in order:
        node_names = ['shiftable_loads', 'base_loads', 'demand', 'generation', 'net_load']
        
        train_scaled, val_scaled, test_scaled, node_idx, ctx_idx = scale()
        logging.info(f"Data is scaled. shape of train data: {train_scaled.shape}, shape of validation data: {val_scaled.shape}, shape of test data: {test_scaled.shape}")
        
        # Determine indices based on arguments
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
            
    except Exception as e:
        logging.info(f"Error during Scaling: {e}", exc_info=True)
        return
    
    # Slice the data. 
    try:
        logging.info("Preparing data by a sliding window ....")
        X_train, y_train = sliding_window_seq2point(train_scaled, args.window_size, input_idx, target_idx)
        X_val, y_val = sliding_window_seq2point(val_scaled, args.window_size, input_idx, target_idx)
        
        logging.info(f"The data is prepared for training. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}.")
        logging.info(f"The data is prepared for validation. X_val shape: {X_val.shape}, y_val shape: {y_val.shape}.")
        
    except Exception as e:
        logging.info(f"Error during preparing the data using a sliding window: {e}", exc_info=True)
        return

    # Create DataLoaders. 
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # Initialize the model.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    logging.info(f"Using device: {device}")
    
    model = Seq2Point(window_size=args.window_size).to(device)
    logging.info("Initialized the model.")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    epochs = args.epochs
    if args.dry_run:
        epochs = 1
        logging.info("Dry run mode enabled. Training for 1 epoch.")
        
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)

        logging.info(f"Epoch: {epoch+1}: Training Loss: {train_loss: .6f}, Validation Loss: {val_loss: .6f}")
        
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            if not args.dry_run:
                torch.save(model.state_dict(), 'best_seq2point_model.pth')
                logging.info("New model saved to best_seq2point_model.pth.")

    logging.info(f"Training is completed. Best Validation Loss = {best_val_loss: .6f}")
    return np.array(train_loss_list), np.array(val_loss_list)
