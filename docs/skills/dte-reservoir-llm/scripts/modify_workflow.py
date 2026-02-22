import os
import shutil

def patch_workflow_file(echoself_repo_path):
    """
    Patches the netrain-cached.yml workflow file to add an option
    for training the Reservoir-Augmented Transformer (RAT).
    """
    workflow_file_path = os.path.join(echoself_repo_path, ".github", "workflows", "netrain-cached.yml")
    if not os.path.exists(workflow_file_path):
        raise FileNotFoundError(f"Workflow file not found at: {workflow_file_path}")

    # Backup the original file
    shutil.copyfile(workflow_file_path, workflow_file_path + ".bak")

    with open(workflow_file_path, "r") as f:
        lines = f.readlines()

    # Add a new input to the workflow_dispatch section
    dispatch_index = -1
    for i, line in enumerate(lines):
        if "workflow_dispatch:" in line:
            dispatch_index = i
            break

    if dispatch_index == -1:
        raise RuntimeError("Could not find workflow_dispatch section.")

    inputs_index = -1
    for i in range(dispatch_index, len(lines)):
        if "inputs:" in lines[i]:
            inputs_index = i
            break

    if inputs_index == -1:
        # Add inputs section if it doesn't exist
        lines.insert(dispatch_index + 1, "    inputs:\n")
        inputs_index = dispatch_index + 1

    rat_input = """          model_type:
            description: "Type of model to train (NanEcho or RAT)"
            required: false
            default: "NanEcho"
            type: choice
            options:
              - NanEcho
              - RAT
"""
    lines.insert(inputs_index + 1, rat_input)

    # Find the training step and add a conditional argument
    train_step_index = -1
    for i, line in enumerate(lines):
        if "python train_cached.py" in line:
            train_step_index = i
            break

    if train_step_index == -1:
        raise RuntimeError("Could not find the training step.")

    # Add the conditional model_type argument
    original_line = lines[train_step_index]
    new_line = original_line.rstrip() + " --model_type ${{ github.event.inputs.model_type }}\n"
    lines[train_step_index] = new_line

    with open(workflow_file_path, "w") as f:
        f.writelines(lines)

    print(f"Successfully patched {workflow_file_path} to support RAT training.")

if __name__ == "__main__":
    repo_path = os.path.expanduser("~/echoself")
    if not os.path.isdir(repo_path):
        print(f"'echoself' repository not found at {repo_path}. Please clone it first.")
    else:
        patch_workflow_file(repo_path)
