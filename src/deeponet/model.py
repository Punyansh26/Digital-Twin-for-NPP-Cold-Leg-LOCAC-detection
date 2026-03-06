"""
DeepONet Architecture Implementation
Neural operator for learning CFD solutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchNet(nn.Module):
    """Branch network - processes simulation parameters"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Args:
            input_dim: Input dimension (e.g., 3 for [velocity, break_size, temperature])
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (basis functions)
        """
        super(BranchNet, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Branch input [batch_size, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        return self.network(x)


class TrunkNet(nn.Module):
    """Trunk network - processes spatial coordinates"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Args:
            input_dim: Input dimension (e.g., 3 for [x, y, z])
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (basis functions)
        """
        super(TrunkNet, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Trunk input [n_nodes, input_dim]
        Returns:
            [n_nodes, output_dim]
        """
        return self.network(x)


class DeepONet(nn.Module):
    """
    Deep Operator Network
    Learns mapping from parameters to solution fields
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        super(DeepONet, self).__init__()
        
        branch_config = config['deeponet']['branch_net']
        trunk_config = config['deeponet']['trunk_net']
        self.n_outputs = config['deeponet']['n_outputs']
        
        # Create branch and trunk networks for each output field
        self.branch_nets = nn.ModuleList([
            BranchNet(
                branch_config['input_dim'],
                branch_config['hidden_dims'],
                branch_config['output_dim']
            )
            for _ in range(self.n_outputs)
        ])
        
        self.trunk_nets = nn.ModuleList([
            TrunkNet(
                trunk_config['input_dim'],
                trunk_config['hidden_dims'],
                trunk_config['output_dim']
            )
            for _ in range(self.n_outputs)
        ])
        
        # Bias terms
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(self.n_outputs)
        ])
        
    def forward(self, branch_input, trunk_input):
        """
        Forward pass
        
        Args:
            branch_input: [batch_size, branch_dim]
            trunk_input: [n_nodes, trunk_dim]
            
        Returns:
            outputs: [batch_size, n_outputs, n_nodes]
        """
        batch_size = branch_input.shape[0]
        n_nodes = trunk_input.shape[0]
        
        outputs = []
        
        for i in range(self.n_outputs):
            # Branch network output: [batch_size, basis_dim]
            branch_out = self.branch_nets[i](branch_input)
            
            # Trunk network output: [n_nodes, basis_dim]
            trunk_out = self.trunk_nets[i](trunk_input)
            
            # Dot product: [batch_size, n_nodes]
            # branch_out: [batch_size, basis_dim]
            # trunk_out: [n_nodes, basis_dim]
            output = torch.matmul(branch_out, trunk_out.transpose(0, 1))  # [batch_size, n_nodes]
            
            # Add bias
            output = output + self.biases[i]
            
            outputs.append(output)
        
        # Stack outputs: [batch_size, n_outputs, n_nodes]
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepONetLoss(nn.Module):
    """Custom loss function for DeepONet"""
    
    def __init__(self, weights=None):
        """
        Args:
            weights: Optional weights for each output field
        """
        super(DeepONetLoss, self).__init__()
        self.weights = weights if weights is not None else [1.0] * 4
        
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss
        
        Args:
            predictions: [batch_size, n_outputs, n_nodes]
            targets: [batch_size, n_outputs, n_nodes]
            
        Returns:
            loss: scalar
        """
        n_outputs = predictions.shape[1]
        
        total_loss = 0.0
        for i in range(n_outputs):
            field_loss = F.mse_loss(predictions[:, i, :], targets[:, i, :])
            total_loss += self.weights[i] * field_loss
        
        return total_loss


def test_deeponet():
    """Test DeepONet architecture"""
    import yaml
    from pathlib import Path
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = DeepONet(config)
    
    print(f"DeepONet created")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    n_nodes = 1000
    
    branch_input = torch.randn(batch_size, 3)
    trunk_input = torch.randn(n_nodes, 3)
    
    output = model(branch_input, trunk_input)
    
    print(f"\nTest forward pass:")
    print(f"Branch input: {branch_input.shape}")
    print(f"Trunk input: {trunk_input.shape}")
    print(f"Output: {output.shape}")
    
    # Expected: [4, 4, 1000]
    assert output.shape == (batch_size, 4, n_nodes), "Output shape mismatch"
    
    print("\n✓ DeepONet test passed")


if __name__ == "__main__":
    test_deeponet()
