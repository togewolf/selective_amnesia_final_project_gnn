import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})

TARGET_CLASS = 0
CSV_FOLDER = 'evaluation_data'
RESULTS_CSV = f'evaluation_data/results_target_{TARGET_CLASS}.csv'
FINAL_BEST_CSV = f'evaluation_data/final_results_target_{TARGET_CLASS}.csv'

def parameter_trend_plot(results_csv=RESULTS_CSV, target_class=TARGET_CLASS):
    os.makedirs('evaluation_data/plots', exist_ok=True)
    df = pd.read_csv(results_csv)
    target_col = f'digit_{target_class}_after'

    unique_models = sorted(df['Model'].unique())
    palette = dict(zip(unique_models, sns.color_palette("plasma", len(unique_models))))
    
    marker_map = dict(zip(unique_models, ['D', 's', 'o', '^', 'v'][:len(unique_models)]))

    line_kwargs = {
        'hue': 'Model',
        'style': 'Model',
        'palette': palette,
        'markers': marker_map,
        'lw': 3,
        'alpha': 0.7,
        'errorbar': None,
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    if 'gamma' in df.columns:
        sns.lineplot(data=df, x='gamma', y=target_col, ax=axes[0], **line_kwargs)
        axes[0].set_xscale('log')
        axes[0].set_title(f'Trend: Gamma (Target: {target_class})\n')
        axes[0].set_xlabel('Gamma (Replay Strength)')

    if 'lmbda' in df.columns:
        df_no_gan = df[df['Model'] != 'GAN']
        sns.lineplot(data=df_no_gan, x='lmbda', y=target_col, ax=axes[1], legend=True, **line_kwargs)
        axes[1].set_xscale('log')
        axes[1].set_title(f'Trend: Lambda (Target: {target_class})\n')
        axes[1].legend(title='Model', loc='best', frameon=False)

    
    if 'loss_type' in df.columns:
        dodge_val: any = 0.4 if df['Model'].nunique() > 1 else False

        sns.pointplot(
            data=df, 
            x='loss_type', 
            y=target_col, 
            hue='Model', 
            palette=palette, 
            dodge=dodge_val, # Pylance is now happy
            linestyles="", 
            markers='D', 
            errorbar=None, 
            ax=axes[2]
        )
        axes[2].set_title(f'Trend: Loss Function\n')
        axes[2].set_xlabel('SA Loss Type')
        axes[2].legend(title='Model', loc='best', frameon=False)


    if 'lr' in df.columns:
        sns.lineplot(data=df, x='lr', y=target_col, ax=axes[3], **line_kwargs)
        axes[3].set_xscale('log')
        axes[3].set_title(f'Trend: Learning Rate\n')
        axes[3].set_xlabel('Learning Rate')

    for ax in axes:
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3, linestyle='--')


    plt.tight_layout()
    plt.savefig(f'evaluation_data/plots/trends_{TARGET_CLASS}_grid.png', bbox_inches='tight')



def heatmap_plot(csv_path=FINAL_BEST_CSV, target_class=TARGET_CLASS):

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run generate_final_models first.")
        return

    df = pd.read_csv(csv_path)

    delta_data = []
    model_labels = []

    for _, row in df.iterrows():
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

    plt.title(f'Accuracy after SA for all Digits (Target: Digit {target_class})\n')
    plt.ylabel('Model')
    plt.xlabel('MNIST Classes')
    
    plt.tight_layout()
    plt.savefig(f'evaluation_data/plots/heatmap_{target_class}.png', dpi=300)
    
def get_best_runs_across_all_targets():
    all_best_runs = []
    
    for target_class in range(10):
        target_csv = f'evaluation_data/final_results_target_{target_class}.csv'
        
        if os.path.exists(target_csv):
            df = pd.read_csv(target_csv)
            df['Target_Class'] = target_class
            df['Target_Accuracy_After'] = df[f'digit_{target_class}_after']
            all_best_runs.append(df)
        else:
            print(f"Warning: Missing data for target {target_class}")

    if not all_best_runs:
        return pd.DataFrame()
        
    return pd.concat(all_best_runs, ignore_index=True)

def stability_boxplot():
    os.makedirs('evaluation_data/plots', exist_ok=True)
    master_df = get_best_runs_across_all_targets()
    
    if master_df.empty:
        print("No data found to plot. Check if final_results_target_X.csv files exist.")
        return

    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=master_df, 
        x='Model', 
        y='Target_Accuracy_After', 
        hue='Model', 
        palette='plasma', 
        width=0.5, 
        fliersize=0,
        legend=False
    )
    
    sns.swarmplot(
        data=master_df, 
        x='Model', 
        y='Target_Accuracy_After', 
        color=".25", 
        size=6, 
        alpha=0.7,
    )
    datapoints_handle = mlines.Line2D([], [], color='.25', marker='o', linestyle='None',markersize=6, alpha=0.7, label='Class Accuracy')
    threshold_line = plt.axhline(0.05, ls='--', color='green', alpha=0.5, label='Forgetting Threshold')    

    plt.legend(handles=[datapoints_handle, threshold_line], 
            loc='upper right', 
            frameon=True,
            fontsize=12)

    plt.title('Unlearning Stability Across All MNIST Classes\n(Lower Accuracy = Better Forgetting)', pad=15)
    plt.ylabel('Target Class Accuracy After SA')
    plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_data/plots/stability_boxplot_master.png', dpi=300)

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

    plt.title(f'Entanglement Strength Matrix: {model_name}\n', pad=15)
    plt.ylabel('Target MNIST Classe')
    plt.xlabel('All MNIST Classes')
    
    ax = plt.gca()
    for i in range(10):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))

    plt.tight_layout()
    save_path = f'evaluation_data/plots/entanglement_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300)


def plot_all(target_classes=range(10)):
    models = ["GAN", "Autoregressive", "VAE", "RectifiedFlow", "NVP"]
    
    for c in target_classes:
        res = f'evaluation_data/results_target_{c}.csv'
        best = f'evaluation_data/final_results_target_{c}.csv'
        
        missing = [f for f in [res, best] if not os.path.exists(f)]
        if missing:
            logging.warning(f"Target {c}: Missing files {missing}")
            continue

        logging.info(f"Generating plots for target {c}")
        parameter_trend_plot(res, c)
        heatmap_plot(best, c)
        stability_boxplot()
    
    for model in models:
        entanglement_matrix(model_name=model)

if __name__ == "__main__":
    #plot_all(range(10))
    stability_boxplot()