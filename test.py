import torch

# Path to your old weights
path = "models/weights/cache/gan_v1.pth"
state_dict = torch.load(path, map_location="cpu")

# List of components to fix
components = ["generator", "discriminator"]

for comp in components:
    old_key = f"{comp}.label_emb.weight"
    new_weight_key = f"{comp}.label_proj.weight"
    new_bias_key = f"{comp}.label_proj.bias"
    
    if old_key in state_dict:
        print(f"Converting {old_key}...")
        # 1. Transpose the weight (10, 50) -> (50, 10) to match Linear requirements
        state_dict[new_weight_key] = state_dict[old_key].t()
        
        # 2. Create a zero bias vector of size 50
        state_dict[new_bias_key] = torch.zeros(50)
        
        # 3. Remove the old key
        del state_dict[old_key]

# Save the "fixed" weights back
torch.save(state_dict, path)
print("Surgery complete. You can now load the model.")