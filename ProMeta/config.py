import argparse

class Config:
    """Holds model and training configurations."""
    def __init__(self, args):
        self.query_size = 128
        self.batch_size = args.batch_size
        
        # Learning Rates
        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        
        # Training loop
        self.inner_step = 5
        self.epochs = 100
        self.patience = 10
        self.dropout_rate = args.dropout
        
        # Architecture parameters
        self.hidden_dim = 128
        self.embed_dim = 64
        self.num_heads = 2
        self.num_layers = 2
        
        # Loss parameters
        self.focal_alpha = 0.75
        self.focal_gamma = 2.0
        self.l1_lambda = args.l1_lambda

def parse_args():
    parser = argparse.ArgumentParser(description="ProMeta: Pathway-Gated Meta-Learning for Proteomics")
    
    # Path Arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing input pkl files")
    parser.add_argument("--proteomics_csv", type=str, required=True, help="Path to preprocessed proteomics CSV")
    parser.add_argument("--cpdb_path", type=str, required=True, help="Path to CPDB pathways .tab file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save models and logs")
    
    # Training Arguments
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU Device ID")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--support_size", type=int, default=32, help="Support set size (k-shot)")
    parser.add_argument("--batch_size", type=int, default=4, help="Meta-batch size (tasks per batch)")
    
    # Hyperparameters
    parser.add_argument("--outer_lr", type=float, default=1e-4, help="Outer loop learning rate")
    parser.add_argument("--inner_lr", type=float, default=0.005, help="Inner loop learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--l1_lambda", type=float, default=1e-3, help="L1 regularization coefficient for gate")
    
    args = parser.parse_args()
    config = Config(args)
    return args, config