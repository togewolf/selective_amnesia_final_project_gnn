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

    # 1. Collect Data
    for c in range(10):
        csv_path = f"evaluation_data/final_results_target_{c}.csv"
        if not os.path.exists(csv_path):
            for m in models:
                results[m][c] = ("-", "-")
            continue
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            m = row['Model']
            if m in results:
                acc = row['Final_Target_Acc']
                drop = row['Final_Retained_Drop']
                results[m][c] = (acc, drop)

    # 2. Calculate Means and Find the "Best" Row
    model_stats = []
    for model in models:
        valid_accs = [results[model][c][0] for c in range(10) if results[model][c] != ("-", "-")]
        valid_drops = [results[model][c][1] for c in range(10) if results[model][c] != ("-", "-")]
        
        mean_acc = np.mean(valid_accs) if valid_accs else float('inf')
        mean_drop = np.mean(valid_drops) if valid_drops else float('inf')
        model_stats.append((model, mean_acc, mean_drop))

    # Identify the best (minimum) mean target accuracy
    min_mean = min(s[1] for s in model_stats)

    # 3. Print Typst Figure
    print("#figure(")
    print("  table(")
    print("    columns: (auto, " + ", ".join(["1fr"] * 10) + ", 1.2fr),")
    print("    inset: 4pt, stroke: none, align: center,")
    print("    table.hline(y: 0, stroke: 1.5pt),")
    
    headers = [f"[*T{i}*]" for i in range(10)]
    print("    align(left)[*Model*], " + ", ".join(headers) + ", [*Mean*],")
    print("    table.hline(y: 1, stroke: 0.8pt),")
    
    for model, mean_acc, mean_drop in model_stats:
        row_cells = []
        # Format the Model Name cell
        model_cell = f"[{model}]"
        row_str = f"    align(left){model_cell}, "
        
        # Format T0-T9 cells
        for c in range(10):
            val = results[model].get(c, ("-", "-"))
            if val == ("-", "-"):
                cell_content = "[-]"
            else:
                acc, drop = val
                content = f"{acc:.2f}\\ ({drop:+.2f})"
                cell_content = f"[{content}]"
            row_cells.append(cell_content)
        
        # Format Mean cell
        if mean_acc != float('inf'):
            mean_content = f"{mean_acc:.2f}\\ ({mean_drop:+.2f})"
            row_cells.append(f"[*{(mean_acc):.2f}*\n*({(mean_drop):+.2f})*]")
        else:
            row_cells.append("[-]")
            
        row_str += ", ".join(row_cells) + ","
        print(row_str)

    print(f"    table.hline(y: {len(models) + 1}, stroke: 1.5pt)")
    print("  ),")
    print("  caption: [Target accuracy and average retained accuracy drop across all 10 targeted classes. 0.0 target accuracy and 0.0 drop indicate optimal SA. Format: Target Acc (Drop).],")
    print(")\n")

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

    # 1. Collect Data
    for c in range(10):
        csv_path = f"evaluation_data/final_results_target_{c}.csv"
        if not os.path.exists(csv_path):
            for m in models:
                results[m][c] = ("-", "-")
            continue
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            m = row['Model']
            if m in results:
                acc = row['Final_Target_Acc']
                drop = row['Final_Retained_Drop']
                results[m][c] = (acc, drop)

    # 2. Calculate Means and Find the "Best" Row
    model_stats = []
    for model in models:
        valid_accs = [results[model][c][0] for c in range(10) if results[model][c] != ("-", "-")]
        valid_drops = [results[model][c][1] for c in range(10) if results[model][c] != ("-", "-")]
        
        mean_acc = np.mean(valid_accs) if valid_accs else float('inf')
        mean_drop = np.mean(valid_drops) if valid_drops else float('inf')
        model_stats.append((model, mean_acc, mean_drop))

    # Identify the best (minimum) mean target accuracy
    min_mean = min(s[1] for s in model_stats)

    # 3. Print Typst Figure
    print("#figure(")
    print("  table(")
    print("    columns: (auto, " + ", ".join(["1fr"] * 10) + ", 1.2fr),")
    print("    inset: 4pt, stroke: none, align: center,")
    print("    table.hline(y: 0, stroke: 1.5pt),")
    
    headers = [f"[*T{i}*]" for i in range(10)]
    print("    align(left)[*Model*], " + ", ".join(headers) + ", [*Mean*],")
    print("    table.hline(y: 1, stroke: 0.8pt),")
    
    for model, mean_acc, mean_drop in model_stats:
        row_cells = []
        # Format the Model Name cell
        model_cell = f"[{model}]"
        row_str = f"    align(left){model_cell}, "
        
        # Format T0-T9 cells
        for c in range(10):
            val = results[model].get(c, ("-", "-"))
            if val == ("-", "-"):
                cell_content = "[-]"
            else:
                acc, drop = val
                content = f"{acc:.2f}\\ ({drop:+.2f})"
                cell_content = f"[{content}]"
            row_cells.append(cell_content)
        
        # Format Mean cell
        if mean_acc != float('inf'):
            row_cells.append(f"[*{(mean_acc):.2f}*\n*({(mean_drop):+.2f})*]")
        else:
            row_cells.append("[-]")
            
        row_str += ", ".join(row_cells) + ","
        print(row_str)

    print(f"    table.hline(y: {len(models) + 1}, stroke: 1.5pt)")
    print("  ),")
    print("  caption: [Target accuracy and average retained accuracy drop across all 10 targeted classes. 0.0 target accuracy and 0.0 drop indicate optimal SA. Format: Target Acc (Drop).],")
    print("  <tab_all_acc>")
    print(")\n")

def base_acc_table():
    models = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]
    results = {m: {d: [] for d in range(10)} for m in models}

    # 1. Collect Data across all 10 target csvs
    for c in range(10):
        csv_path = f"evaluation_data/final_results_target_{c}.csv"
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            m = row['Model']
            if m in results:
                for d in range(10):
                    col_before = f'digit_{d}_before'
                    if col_before in row:
                        results[m][d].append(row[col_before])

    # 2. Print Typst Figure
    print("#figure(")
    print("  table(")
    print("    columns: (auto, " + ", ".join(["1fr"] * 10) + ", 1.2fr),")
    print("    inset: 4pt, stroke: none, align: center,")
    print("    table.hline(y: 0, stroke: 1.5pt),")
    
    headers = [f"[*D{i}*]" for i in range(10)]
    print("    align(left)[*Model*], " + ", ".join(headers) + ", [*Mean*],")
    print("    table.hline(y: 1, stroke: 0.8pt),")
    
    for model in models:
        row_cells = []
        row_str = f"    align(left)[{model}], "
        
        model_means = []
        for d in range(10):
            vals = results[model][d]
            if vals:
                mean_val = np.mean(vals)
                model_means.append(mean_val)
                row_cells.append(f"[{mean_val:.3f}]") # using 3 decimal places for clarity
            else:
                row_cells.append("[-]")
        
        if model_means:
            overall_mean = np.mean(model_means)
            row_cells.append(f"[*{(overall_mean):.3f}*]")
        else:
            row_cells.append("[-]")
            
        row_str += ", ".join(row_cells) + ","
        print(row_str)

    print(f"    table.hline(y: {len(models) + 1}, stroke: 1.5pt)")
    print("  ),")
    print("  caption: [Base model accuracies before selective amnesia for each digit. Averaged over all 10 evaluation runs.],")
    print("  <tab_base_acc>")
    print(")\n")

def generation_acc_table():
    # We only need the target 0 file to get the base model generation accuracies
    csv_path = f"evaluation_data/final_results_target_0.csv"
    
    print("#figure(")
    print("  table(")
    print("    columns: (auto, 1fr, 1fr),")
    print("    inset: 5pt, stroke: none, align: (left, center, center),")
    print("    table.hline(y: 0, stroke: 1.5pt),")
    print("    [*Model*], [*Target 0 Accuracy*], [*Mean Accuracy*],")
    print("    table.hline(y: 1, stroke: 0.8pt),")

    if not os.path.exists(csv_path):
        print("    [Error: CSV not found], [-], [-],")
    else:
        df = pd.read_csv(csv_path)
        # Using your specific model order
        models = ["VAE", "NVP", "GAN", "RectifiedFlow", "Autoregressive"] 
        
        for model in models:
            row = df[df['Model'] == model]
            if not row.empty:
                # Get Target 0 before accuracy
                t0_acc = row.iloc[0]['digit_0_before']
                
                # Calculate mean of all 10 digits before
                all_digits = [row.iloc[0][f'digit_{d}_before'] for d in range(10)]
                mean_acc = np.mean(all_digits)
                
                # Format as percentages (e.g., ~97.5%)
                print(f"    [{model}], [~{t0_acc*100:.1f}%], [~{mean_acc*100:.1f}%],")
            else:
                print(f"    [{model}], [-], [-],")

    print(f"    table.hline(y: 6, stroke: 1.5pt)")
    print("  ),")
    print("  caption: [Base model generation accuracy as judged by the Oracle (Target 0 and Mean).],")
    print(") <tab_base_model_generation_accuracy>\n")

if __name__ == "__main__":
    # if os.path.exists(REGISTRY):
    #     with open(REGISTRY, 'r') as f:
    #         registry = json.load(f)

    #     base_param_table(registry)
    #     sa_param_table(registry)
    # else:
    #     print(f"Registry file {REGISTRY} not found. Skipping parameter tables.\n")

    # all_acc_table()
    #base_acc_table()
    generation_acc_table()