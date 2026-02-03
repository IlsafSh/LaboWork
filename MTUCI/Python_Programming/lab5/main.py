# Demonstration and CLI for running experiments
# Usage: python main.py --dataset_type keystroke --data_path path/to/dataset.csv --simulate False

import argparse
import os
from config_manager import *

def main(args):
    # Создаем папку reports рядом со скриптом
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    dataset_handler = DatasetHandler(args.dataset_type, args.data_path)

    input_size = dataset_handler.X.shape[1]
    output_size = len(np.unique(dataset_handler.y))
    
    # Example experiments for comparison
    models = [
        LogisticRegressionConfig(input_size=input_size, output_size=output_size),  # Adjust sizes based on dataset
        SVMConfig(input_size=input_size, output_size=output_size),
        RandomForestConfig(input_size=input_size, output_size=output_size),
        GradientBoostingConfig(input_size=input_size, output_size=output_size),
        XGBoostConfig(input_size=input_size, output_size=output_size),
        CatBoostConfig(input_size=input_size, output_size=output_size),
        LSTMConfig(input_size=input_size, output_size=output_size)
    ]
    
    train_config = TrainingConfig(epochs=10, batch_size=32)
    
    experiments = []
    for i, model_config in enumerate(models):
        exp = Experiment(f"EXP{i:03d}", model_config, train_config, dataset_handler, simulate=args.simulate)
        exp.run()
        config_path = os.path.join(reports_dir, f"{exp.exp_id}_config.json")
        exp.save_config(config_path)
        experiments.append(exp)
    
    # Compare
    Experiment.compare_experiments(experiments, "comparison.csv")
    print("Experiments complete. See comparison.csv and reports.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Config Manager for Biometrics")
    parser.add_argument("--dataset_type", type=str, default="keystroke", choices=["keystroke", "mouse"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--simulate", type=bool, default=False, help="Simulate instead of real training")
    args = parser.parse_args()
    main(args)