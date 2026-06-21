from pathlib import Path
from Read_data import read_mopvrp_instance
from Greedy_insertion import greedy_insertion
from ALNS import alns

def main():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "MoPVRP-instance" / "Benchmark" / "MPRP_c++_100_1_4" / "RC201.dat"

    instance = read_mopvrp_instance(file_path)
    greedy_solution = greedy_insertion(instance, print_steps=False)
    
    # 2. Define your Grid Search space
    param_grid = [
    {'alpha': 0.3, 'epsilon_decay_rate': 0.95, 'eta_reward': 0.6, 'C': 1.0},
    {'alpha': 0.3, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.8, 'C': 1.0},
    {'alpha': 0.5, 'epsilon_decay_rate': 0.95, 'eta_reward': 0.6, 'C': 1.0},
    {'alpha': 0.5, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.8, 'C': 1.0},
    {'alpha': 0.3, 'epsilon_decay_rate': 0.95, 'eta_reward': 0.6, 'C': 10.0},
    {'alpha': 0.3, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.8, 'C': 10.0},
    {'alpha': 0.5, 'epsilon_decay_rate': 0.95, 'eta_reward': 0.6, 'C': 10.0},
    {'alpha': 0.5, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.8, 'C': 10.0},
    {'alpha': 0.7, 'epsilon_decay_rate': 0.95, 'eta_reward': 0.8, 'C': 5.0},
    {'alpha': 0.7, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.8, 'C': 5.0},
    {'alpha': 0.5, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.9, 'C': 5.0},
    {'alpha': 0.5, 'epsilon_decay_rate': 0.99, 'eta_reward': 0.9, 'C': 50.0},
    ]

    best_cost = float('inf')
    best_config = None

    print(f"Starting Parameter Tuning on {len(param_grid)} configurations...")

    # 3. Execution loop
    for config in param_grid:
        print(f"\nTesting configuration: {config}")
        
        # We run for fewer iterations during tuning to save time
        # The paper used 400 for tuning
        result = alns(
            instance, 
            iterations=400, 
            random_seed=42,
            initial_solution=greedy_solution,
            **config # Unpacks the dictionary into alpha, epsilon_decay_rate, eta_reward, C
        )
        
        if result and result.total_cost < best_cost:
            best_cost = result.total_cost
            best_config = config
            print(f"New best cost found: {round(best_cost, 2)}")

    print("\n==============================")
    print("TUNING COMPLETE")
    print(f"Optimal parameters: {best_config}")
    print(f"Best cost achieved: {round(best_cost, 2)}")

if __name__ == "__main__":
    main()