import os
import shutil


def patch_model_file(echoself_repo_path):
    """
    Patches the nanecho_model.py file in the echoself repository to add
    the Reservoir-Augmented Transformer (RAT) architecture.
    """
    model_file_path = os.path.join(echoself_repo_path, 'NanEcho', 'nanecho_model.py')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found at: {model_file_path}")

    # Backup the original file
    shutil.copyfile(model_file_path, model_file_path + '.bak')

    with open(model_file_path, 'r') as f:
        lines = f.readlines()

    # Find the Block class definition
    block_start_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('class Block(nn.Module):'):
            block_start_index = i
            break

    if block_start_index == -1:
        raise RuntimeError("Could not find the 'Block' class in the model file.")

    # RATBlock definition
    rat_block_definition = """
class RATBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # Replace MLP with ReservoirReadout
        self.reservoir_readout = CognitiveReadout(config.n_embd, config.n_embd, config.reservoir_config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.reservoir_readout(self.ln_2(x))
        return x

"""

    # CognitiveReadout definition
    cognitive_readout_definition = """
class CognitiveReadout(nn.Module):

    def __init__(self, input_dim, output_dim, reservoir_config):
        super().__init__()
        self.reservoir = EchoReservoir(**reservoir_config)
        self.readout = nn.Linear(reservoir_config['n_units'], output_dim)

    def forward(self, x):
        # NOTE: This is a simplified example for demonstration.
        # A real implementation needs to handle the sequence dimension and the
        # reservoir's internal state properly.
        # This will likely involve iterating over the sequence and updating the state.
        batch_size, seq_len, _ = x.shape
        hidden_states = []
        for t in range(seq_len):
            hidden_states.append(self.reservoir(x[:, t, :]))
        
        reservoir_state = torch.stack(hidden_states, dim=1)
        output = self.readout(reservoir_state)
        return output

"""

    # Find where to insert the new classes
    # We'll insert them right before the original Block class
    lines.insert(block_start_index, rat_block_definition)
    lines.insert(block_start_index, cognitive_readout_definition)

    # Now, we need to modify the main Transformer class to use RATBlock
    # This is more complex as it requires changing the class name in the ModuleList
    # For simplicity, we will assume the user will manually change the model config
    # to use 'RATBlock' instead of 'Block'. The script will just make the class available.

    with open(model_file_path, 'w') as f:
        f.writelines(lines)

    print(f"Successfully patched {model_file_path} with RATBlock and CognitiveReadout.")

if __name__ == '__main__':
    # Example usage:
    # This assumes the echoself repo is cloned at a known location.
    repo_path = os.path.expanduser('~/echoself') # Or pass as an argument
    if not os.path.isdir(repo_path):
        print(f"'echoself' repository not found at {repo_path}. Please clone it first.")
    else:
        patch_model_file(repo_path)

