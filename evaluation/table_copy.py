import json
import pandas as pd
import os
import numpy as np

TARGET_CLASS = 0
REGISTRY = f"models/weights/optimized_model_registry_{TARGET_CLASS}.json"
FINAL_RESULTS = f"evaluation_data/final_results_target_{TARGET_CLASS}.csv"

def format_value(val):
    if isinstance(val, float):
        if val == 0: return "0"
        if val < 0.001 or val >= 1000:
            exponent = f"{val:.0e}".split('e')
            base = exponent[0]
            exp = int(exponent[1])
            return f"${base} times 10^({exp})$"
        return str(val)
    return str(val)

def base_param_table(data):
    header = "#figure(\n  table(\n    columns: (auto, 1fr, 1fr, 2fr),\n    inset: 5pt, stroke: none, align: (left, center, center, center),\n    table.hline(y: 0, stroke: 1.5pt),\n    [*Model*], [*Latent Dim*], [*Hidden Dim*], [*Learning Rate*],\n    table.hline(y: 1, stroke: 0.8pt),"
    print(header)
    
    for model, params in data.items():
        z = params.get("z_dim") or params.get("latent_dim") or "-"
        h = params.get("h_dim") or params.get("h_dim1") or params.get("hidden_dim") or "-"
        lr = format_value(params.get("lr") or params.get("lr_G"))
        print(f"    [{model}], [{z}], [{h}], [{lr}],")
        
    print("    table.hline(y: 6, stroke: 1.5pt)\n  ),\n  caption: [Base model architecture and training hyperparameters.],\n    <tab_baseparam>)\n")

def sa_param_table(data):
    header = "#figure(\n  table(\n    columns: (auto, 2fr, 2fr, 1fr, 2fr),\n    inset: 5pt, stroke: none, align: (left, center, center, center, center),\n    table.hline(y: 0, stroke: 1.5pt),\n    [*Model*], [*$gamma$* (Replay)], [*$lambda$* (EWC)], [*Loss*], [*Learning Rate*],\n    table.hline(y: 1, stroke: 0.8pt),"
    print(header)
    
    for model, params in data.items():
        sa = params["forgetting_config"]
        print(f"    [{model}], [{format_value(sa['gamma'])}], [{format_value(sa['lmbda'])}], [{sa['loss_type'].upper()}], [{format_value(sa['lr'])}],")
        
    print("    table.hline(y: 6, stroke: 1.5pt)\n  ),\n  caption: [Optimal Selective Amnesia (SA) hyperparameters for target class 0.],\n    <tab_SAparam>\n)")

def all_acc_table():
    models = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]
    results = {m: {} for m in models}

    for c in range(10):
        csv_path = f"evaluation_data/final_results_target_{c}.csv"
        
        if not os.path.exists(csv_path):
            for m in models:
                results[m][c] = ("-", "-")
            continue
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            m = row['Model']
            acc = row['Final_Target_Acc']
            drop = row['Final_Retained_Drop']
            results[m][c] = (acc, drop)

    print("#figure(")
    print("  table(")
    print("    columns: (auto, " + ", ".join(["1fr"] * 10) + ", 1.2fr),")
    print("    inset: 4pt, stroke: none, align: center,")
    print("    table.hline(y: 0, stroke: 1.5pt),")
    
    headers = [f"[*T{i}*]" for i in range(10)]
    print("    align(left)[*Model*], " + ", ".join(headers) + ", [*Mean*],")
    print("    table.hline(y: 1, stroke: 0.8pt),")
    
    for i, model in enumerate(models):
        row_str = f"    align(left)[{model}], "
        cols = []
        valid_accs = []
        valid_drops = []
        
        for c in range(10):
            val = results.get(model, {}).get(c, ("-", "-"))
            if val == ("-", "-"):
                cols.append("[-]")
            else:
                acc, drop = val
                cols.append(f"[{acc:.2f}\n({drop:+.2f})]")
                valid_accs.append(acc)
                valid_drops.append(drop)
        
        if valid_accs and valid_drops:
            mean_acc = np.mean(valid_accs)
            mean_drop = np.mean(valid_drops)
            cols.append(f"[*{(mean_acc):.2f}*\n*({(mean_drop):+.2f})*]")
        else:
            cols.append("[-]")
            
        row_str += ", ".join(cols) + ","
        print(row_str)

    print(f"    table.hline(y: {len(models) + 1}, stroke: 1.5pt)")
    print("  ),")
    print("  caption: [Target accuracy and average retained accuracy drop across all 10 targeted classes. 0.0 target accuracy and 0.0 drop indicate optimal SA. Format: Target Acc (Drop).],")
    print(")\n")


if __name__ == "__main__":
    with open(REGISTRY, 'r') as f:
        registry = json.load(f)

    base_param_table(registry)
    sa_param_table(registry)

    all_acc_table()