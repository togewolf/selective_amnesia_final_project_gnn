import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

TARGET_CLASS = 0
CSV_FOLDER = 'evaluation_data_backup/fake_for_test'
RESULTS_CSV = f'evaluation_data_backup/fake_for_test/results_target_{TARGET_CLASS}.csv'

def parameter_trend_plot(RESULTS_CSV=RESULTS_CSV, target_class=TARGET_CLASS):
    os.makedirs('evaluation_data', exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    target_col = f'digit_{target_class}_after'

    unique_models = sorted(df['Model'].unique())
    palette = dict(zip(unique_models, sns.color_palette("plasma", len(unique_models))))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    line_kwargs = {
        'hue': 'Model',
        'style': 'Model',
        'palette': palette,
        'lw': 3,
        'alpha': 0.7,
        'markers': 'D',
        'errorbar': None,
        'legend': False 
    }

    if 'gamma' in df.columns:
        sns.lineplot(data=df, x='gamma', y=target_col, ax=axes[0], **line_kwargs)
        axes[0].set_xscale('log')
        axes[0].set_title(f'Trend: Gamma (Target: {target_class})')
        axes[0].set_xlabel('Gamma (Replay Strength)')

    if 'lmbda' in df.columns:
        df_no_gan = df[df['Model'] != 'GAN']
        sns.lineplot(data=df_no_gan, x='lmbda', y=target_col, ax=axes[1], **line_kwargs)
        axes[1].set_xscale('log')
        axes[1].set_title(f'Trend: Lambda (Target: {target_class})')
        axes[1].set_xlabel('Lambda (Weight Protection)')

    if 'loss_type' in df.columns:
        sns.pointplot(data=df, x='loss_type', y=target_col, hue='Model', 
                      palette=palette, dodge=0.4, linestyles="", markers='D', 
                      errorbar=None, ax=axes[2])
        axes[2].set_title(f'Trend: Loss Function')
        axes[2].set_xlabel('SA Loss Type')
        axes[2].legend(title='Model', loc='best', frameon=False, fontsize=12)

    if 'lr' in df.columns:
        sns.lineplot(data=df, x='lr', y=target_col, ax=axes[3], **line_kwargs)
        axes[3].set_xscale('log')
        axes[3].set_title(f'Trend: Learning Rate')
        axes[3].set_xlabel('Learning Rate')

    for ax in axes:
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3, linestyle='--')


    plt.tight_layout()
    plt.savefig(f'evaluation_data/plots/trends_{TARGET_CLASS}_grid.png', bbox_inches='tight')



def heatmap_plot(RESULTS_CSV=RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)

    target_after_col = f'digit_{TARGET_CLASS}_after'
    before_cols = [f'digit_{i}_before' for i in range(10)]
    retention_cols = [f'digit_{i}_after' for i in range(10) if i != TARGET_CLASS]

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

        successes = m_df_qualified[m_df_qualified[target_after_col] <= forgot_threshold].copy()
        
        if successes.empty:
            best_run = m_df_qualified.loc[m_df_qualified[target_after_col].idxmin()]
        else:
            successes['mean_retention'] = successes[retention_cols].mean(axis=1)
            best_run = successes.loc[successes['mean_retention'].idxmax()]
        
        best_reps.append(best_run)

    best_df = pd.DataFrame(best_reps)

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
                cbar_kws={'label': r'$\Delta$ Accuracy'})

    plt.title(f'Accuracy after SA for all Digits (Target: Digit {TARGET_CLASS})', fontsize=14)
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('MNIST Classes', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'evaluation_data/plots/heatmap_{TARGET_CLASS}.png', dpi=300)

    for _, row in best_df.iterrows():
        gamma = row.get('gamma', '-')
        lmbda = row.get('lmbda', '-')
        loss = row.get('loss_type', '-')
        lr = row.get('lr', '-')
        
        retention = np.mean([row[f'digit_{i}_after'] for i in range(10) if i != TARGET_CLASS])
        
        print(f'[{row["Model"]}], [{gamma}], [{lmbda}], [{loss}],[{lr}], [{row[target_after_col]:.3f}], [{retention:.3f}],')

def get_best_runs_across_all_targets():
    all_best_runs = []
    
    for target_class in range(10):
        csv_path = CSV_FOLDER + f'/results_target_{target_class}.csv'
        if not os.path.exists(csv_path):
            print(f"Skipping Target {target_class}: File not found.")
            continue
            
        df = pd.read_csv(csv_path)
        target_after_col = f'digit_{target_class}_after'
        before_cols = [f'digit_{i}_before' for i in range(10)]
        retention_cols = [f'digit_{i}_after' for i in range(10) if i != target_class]

        baseline_threshold = 0.85
        forgot_threshold = 0.05

        for model in df['Model'].unique():
            m_df = df[df['Model'] == model].copy()
            quality_mask = (m_df[before_cols] >= baseline_threshold).all(axis=1)
            m_df_qualified = m_df[quality_mask].copy()
            
            if m_df_qualified.empty:
                m_df_qualified = m_df # Fallback

            successes = m_df_qualified[m_df_qualified[target_after_col] <= forgot_threshold].copy()
            
            if successes.empty:
                best_run = m_df_qualified.loc[m_df_qualified[target_after_col].idxmin()].copy()
            else:
                successes['mean_retention'] = successes[retention_cols].mean(axis=1)
                best_run = successes.loc[successes['mean_retention'].idxmax()].copy()
            
            best_run['Target_Class'] = target_class
            best_run['Target_Accuracy_After'] = best_run[target_after_col]
            all_best_runs.append(best_run)

    return pd.DataFrame(all_best_runs)
def stability_boxplot():
    """
    Creates a box plot showing the variance of target forgetting accuracy 
    across all 10 digits for each architecture.
    """
    os.makedirs('evaluation_data/plots', exist_ok=True)
    master_df = get_best_runs_across_all_targets()
    
    if master_df.empty:
        print("No data found to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=master_df, x='Model', y='Target_Accuracy_After', 
                palette='plasma', width=0.5, fliersize=5)
    
    sns.swarmplot(data=master_df, x='Model', y='Target_Accuracy_After', 
                  color=".25", size=6, alpha=0.7)

    plt.title('Unlearning Stability Across All MNIST Classes (Lower is Better)', fontsize=14, pad=15)
    plt.ylabel('Target Class Accuracy After SA', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.axhline(0.05, ls='--', color='green', alpha=0.5, label='Perfect Forgetting Threshold (0.05)')
    plt.legend()
    
    plt.tight_layout()
    save_path = 'evaluation_data/plots/stability_boxplot_master.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved Master Boxplot to {save_path}")


def entanglement_matrix(model_name='GAN'):

    os.makedirs('evaluation_data/plots', exist_ok=True)
    master_df = get_best_runs_across_all_targets()
    
    if master_df.empty:
        return

    model_df = master_df[master_df['Model'] == model_name]
    
    if model_df.empty:
        print(f"No data for model {model_name}")
        return

    model_df = model_df.sort_values('Target_Class')

    delta_matrix = []
    actual_targets = []
    
    for _, row in model_df.iterrows():
        deltas = [row[f'digit_{i}_after'] - row[f'digit_{i}_before'] for i in range(10)]
        delta_matrix.append(deltas)
        actual_targets.append(f"Target {int(row['Target_Class'])}")

    delta_df = pd.DataFrame(delta_matrix, 
                            index=actual_targets,
                            columns=[f'Digit {i}' for i in range(10)])

    plt.figure(figsize=(10, 8))
    

    sns.heatmap(delta_df, 
                annot=True, 
                cmap='RdYlGn', 
                center=0,
                vmin=-1.0, vmax=0.2,
                fmt=".2f", 
                linewidths=.5,
                cbar_kws={'label': r'$\Delta$ Accuracy (Negative = Forgotten)'})

    plt.title(f'Entanglement Strength Matrix: {model_name}', fontsize=16, pad=15)
    plt.ylabel('Target MNIST Classe', fontsize=12)
    plt.xlabel('All MNIST Classes', fontsize=12)
    
    ax = plt.gca()
    for i in range(10):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))

    plt.tight_layout()
    save_path = f'evaluation_data/plots/entanglement_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    parameter_trend_plot()
    heatmap_plot()
    stability_boxplot()
    
    for model in ["GAN", "Autoregressive", "VAE", "RectifiedFlow", "NVP"]:
        entanglement_matrix(model_name=model)