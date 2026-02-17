import argparse
import logging
from train_Seq2Point import train
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Seq2Point CNN for Load Disaggregation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    # Window size must be > 29 for the current architecture. Default to 100.
    parser.add_argument("--window_size", type=int, default=100, help="Window size (Seq2Point input length)")
    parser.add_argument("--dry_run", action="store_true", help="Run for 1 epoch to test pipeline")
    parser.add_argument("--log_file", type=str, default="training_seq2point.log", help="Path to log file")
    
    # New arguments for channel selection
    parser.add_argument("--input_channel", type=str, default="net_load", 
                        help="Name of the aggregate input channel (e.g., 'net_load', 'demand')")
    parser.add_argument("--target_channel", type=str, default="shiftable_loads", 
                        help="Name of the target appliance channel (e.g., 'shiftable_loads', 'base_loads')")
    
    args = parser.parse_args()
    
    try:
        train_loss_list, val_loss_list = train(args)
        
        if train_loss_list is not None:
            # Plotting the loss for IEEE Publication
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 10
            plt.rcParams['figure.titlesize'] = 14

            plt.figure(figsize=(6, 4))
            
            plt.plot(train_loss_list, label='Training Loss', color='b', linestyle='-', linewidth=2)
            plt.plot(val_loss_list, label='Validation Loss', color='r', linestyle='--', linewidth=2)
            
            plt.title('Seq2Point Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.legend(loc='best', frameon=True)
            plt.grid(True, linestyle=':', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig('seq2point_loss.png', dpi=300, bbox_inches='tight')
            logging.info("Loss plot saved to seq2point_loss.png")
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
