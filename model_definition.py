import torch
import torch.nn as nn

class StackedLSTM_MultiPooling(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Hidden dimension for each LSTM layer.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate applied in LSTM and classifier.
        """
        super(StackedLSTM_MultiPooling, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
       
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            curr_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(input_size=curr_input_size, hidden_size=hidden_size,
                        num_layers=1, batch_first=True, dropout=dropout if num_layers > 1 else 0) # Corrected dropout usage
            )
        # Total feature dimension for classification is: num_layers * hidden_size * 2 (for mean and max pooling).
        pooled_feature_size = num_layers * hidden_size * 2
       
        self.classifier = nn.Sequential(
            nn.Linear(pooled_feature_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
        # Manual dropout between LSTM layers (if desired).
        self.manual_dropout = nn.Dropout(dropout)
    
    def pool_outputs(self, lstm_output):
        """Helper function to perform mean and max pooling."""
        mean_pool = lstm_output.mean(dim=1)  # Shape: (batch, hidden_size)
        max_pool, _ = lstm_output.max(dim=1)  # Shape: (batch, hidden_size)
        return torch.cat([mean_pool, max_pool], dim=1)  # Shape: (batch, hidden_size * 2)
    
    def forward(self, x, return_intermediates=False): # Keep return_intermediates for flexibility, though we might not use it in API
        """
        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, input_size).
            return_intermediates (bool): If True, returns intermediate pooled outputs.
        Returns:
            logits (Tensor): Class logits of shape (batch, num_classes).
            intermediates (dict, optional): Dictionary with pooled outputs from each layer.
        """
        # Ensure input is on the same device as the model parameters
        x = x.to(next(self.parameters()).device)

        current_input = x # Use a different variable name to avoid confusion with outer scope x
        all_pooled_outputs = [] # Store pooled outputs from each layer to be concatenated

        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_out, _ = lstm_layer(current_input)
            
            # Apply manual dropout only if it's not the last LSTM layer
            if i < self.num_layers - 1:
                lstm_out = self.manual_dropout(lstm_out)
            
            pooled = self.pool_outputs(lstm_out)
            all_pooled_outputs.append(pooled)
            
            # The output of the current LSTM layer becomes the input to the next
            current_input = lstm_out
        
        # Concatenate the pooled outputs from all LSTM layers
        final_representation = torch.cat(all_pooled_outputs, dim=1)
        logits = self.classifier(final_representation)
        
        if return_intermediates: # This part can be kept for potential future use or removed if API never uses it
            # This part of the original logic seems to store outputs in a dict by layer name,
            # but then concatenates `list(pooled_outputs.values())`.
            # For clarity, let's just return the concatenated representation if intermediates are needed.
            # Or, if you need the dictionary for some reason:
            intermediate_dict = {f"LSTM_Layer_{idx+1}_pooled": p_out for idx, p_out in enumerate(all_pooled_outputs)}
            return logits, intermediate_dict
            
        return logits