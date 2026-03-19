import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration
RESULTS_CSV = 'evaluation_data/results_target_class_0.csv'
TARGET_CLASS = 0  # This matches your CSV snippet

def parameter_trend_plot(csv_path=RESULTS_CSV, target_class=TARGET_CLASS):
    os.makedirs('evaluation_data', exist_ok=True)
    df = pd.read_csv(csv_path)

    # Use the specific digit column as the target for the y-axis
    target_col = f'digit_{target_class}_after'

    if 'gamma' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='gamma', y=target_col, hue='Model', marker='o', errorbar=None)
        plt.title(f'Trend: Forgetting Effectiveness vs. Gamma (Target: Digit {target_class})')
        plt.ylabel(f'Accuracy on Target Digit {target_class}')
        plt.xlabel('Gamma (Replay Strength)')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_data/trend_gamma.png')
        plt.close()

    if 'lmbda' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='lmbda', y=target_col, hue='Model', marker='s', errorbar=None)
        plt.title(f'Trend: Forgetting Effectiveness vs. Lambda (Target: Digit {target_class})')
        plt.ylabel(f'Accuracy on Target Digit {target_class}')
        plt.xlabel('Lambda (Weight Protection Strength)')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_data/trend_lambda.png')
        plt.close()

    if 'loss_type' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.pointplot(data=df, x='loss_type', y=target_col, hue='Model', 
                      dodge=True, linestyles="", markers='D', errorbar=None)
        plt.title(f'Trend: Forgetting Effectiveness vs. Loss Function (Target: Digit {target_class})')
        plt.ylabel(f'Accuracy on Target Digit {target_class}')
        plt.xlabel('Unlearning Loss Function')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_data/trend_loss_type.png')
        plt.close()

def all_class_drop_plot(csv_path=RESULTS_CSV):
    df = pd.read_csv(csv_path)

    target_after_col = f'digit_{TARGET_CLASS}_after'
    before_cols = [f'digit_{i}_before' for i in range(10)]
    retention_cols = [f'digit_{i}_after' for i in range(10) if i != TARGET_CLASS]

    # Filtering logic
    baseline_threshold = 0.85
    forgot_threshold = 0.05

    best_reps = []

    for model in df['Model'].unique():
        m_df = df[df['Model'] == model].copy()
        
        quality_mask = (m_df[before_cols] >= baseline_threshold).all(axis=1)
        m_df_qualified = m_df[quality_mask].copy()
        
        if m_df_qualified.empty:
            print(f"Warning: No runs for {model} met the baseline quality of {baseline_threshold}.")
            m_df_qualified = m_df
            print("  Using non-qualified data as a fallback.")

        # Find runs where the target digit was successfully forgotten
        successes = m_df_qualified[m_df_qualified[target_after_col] <= forgot_threshold].copy()
        
        if successes.empty:
            # Fallback: take the run with the lowest accuracy on the target class
            best_run = m_df_qualified.loc[m_df_qualified[target_after_col].idxmin()]
        else:
            # Of the successful forgets, take the one with highest average retention on other digits
            successes['mean_retention'] = successes[retention_cols].mean(axis=1)
            best_run = successes.loc[successes['mean_retention'].idxmax()]
        
        best_reps.append(best_run)

    best_df = pd.DataFrame(best_reps)

    # Prepare data for heatmap
    delta_data = []
    model_labels = []

    for _, row in best_df.iterrows():
        deltas = [row[f'digit_{i}_after'] - row[f'digit_{i}_before'] for i in range(10)]
        delta_data.append(deltas)
        model_labels.append(f"{row['Model']}")

    delta_matrix = pd.DataFrame(delta_data, 
                                columns=[f'Digit {i}' for i in range(10)], 
                                index=model_labels)

    plt.figure(figsize=(12, 5))
    sns.heatmap(delta_matrix, 
                annot=True, 
                cmap='RdYlGn', 
                center=0, 
                fmt=".2f", 
                linewidths=.5,
                cbar_kws={'label': '$\Delta$ Accuracy'})

    plt.title(f'Selective Unlearning Heatmap (Target: Digit {TARGET_CLASS})', fontsize=14)
    plt.ylabel('Model Architecture', fontsize=12)
    plt.xlabel('MNIST Classes', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('evaluation_data/performance_comparison_heatmap.png', dpi=300)

    # Typst-ready Table output
    print("\n--- Typst Table Rows ---")
    for _, row in best_df.iterrows():
        gamma = row.get('gamma', '-')
        lmbda = row.get('lmbda', '-')
        loss = row.get('loss_type', '-')
        
        # Calculate retention specifically excluding the target class
        retention = np.mean([row[f'digit_{i}_after'] for i in range(10) if i != TARGET_CLASS])
        
        print(f'([{row["Model"]}], [{gamma}], [{lmbda}], [{loss}], [{row[target_after_col]:.3f}], [{retention:.3f}]),')

if __name__ == "__main__":
    parameter_trend_plot()
    all_class_drop_plot()