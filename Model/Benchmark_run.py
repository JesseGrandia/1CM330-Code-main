"""
This script outputs the data for every run (10 runs per instance). 
This is used to perform the Wilcoxon signed-rank test.
"""

import csv
import os
import time
from pathlib import Path

from Read_data import read_mopvrp_instance
from ALNS import alns
from Greedy_insertion import greedy_insertion


# ---- SETTINGS ----------------------------------------------------------
#   set MODE to "qlearning" or "base".
#   "qlearning" -> Q-learning combination agent + Simulated Annealing (hybrid)
#   "base"      -> AOS roulette wheel + threshold acceptance (pure metaheuristic)
MODE = "qlearning"

MODEL_NAME = MODE                         
OUTPUT_FILE = f"{MODE}_results.csv"    

N_RUNS = 10                      
BASE_SEED = 42                  
ITERATIONS = 1500                 
LAMBDA_1 = 0.10
LAMBDA_2 = 0.40

# Q-learning params (only used when MODE == "qlearning")
alpha = 0.5
gamma = 0.7
epsilon_decay_rate = 0.997
eta_reward = 0.6
C = 1.0

# Base params (only used when MODE == "base")
THRESHOLD_START = 0.10
SEGMENT_SIZE = 100
REACTION_FACTOR = 0.1

TARGET_FOLDERS = [
    "MPRP_c++_25_2_2",
    "MPRP_c++_50_3_3",
    "MPRP_c++_100_1_1",
]
# ---------------------------------------------------------------------------

def get_size(folder_name):
    """
    Extracts the instance size from the folder name.
    Example: "MPRP_c++_10_2_3" -> "10"
    """
    return folder_name.split("_")[2]


def main():
    base_dir = Path(__file__).resolve().parent
    benchmark_dir = base_dir.parent / "data" 
    
    # Fallback to local MoPVRP-instance folder if the global data folder isn't found
    if not benchmark_dir.exists():
        benchmark_dir = base_dir / "MoPVRP-instance" / "Benchmark"

    csv_file_path = base_dir / OUTPUT_FILE

    print(f"Starting benchmark -> {OUTPUT_FILE}")
    print(f"{N_RUNS} runs per instance, {ITERATIONS} iteraties per run\n")

    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow([
            "Model",
            "Folder",
            "Instance",
            "Seed",
            "Run_Number",
            "Objective_Cost",
            "Time_s",
            "Feasible"
        ])

        for folder_name in TARGET_FOLDERS:
            folder_path = benchmark_dir / folder_name

            if not folder_path.exists():
                print(f"Folder missing: {folder_path}")
                continue

            dat_files = sorted(folder_path.glob("*.dat"))

            for file_path in dat_files:
                size = get_size(folder_name)
                instance_name = file_path.stem
                instance_label = f"{size}{instance_name}"  

                try:
                    instance = read_mopvrp_instance(file_path)
                    greedy_solution = greedy_insertion(instance, print_steps=False)

                    for run_index in range(N_RUNS):
                        seed = BASE_SEED + run_index
                        start = time.time()

                        solution = alns(
                            instance,
                            iterations=ITERATIONS,
                            mode=MODE,
                            lambda_1=LAMBDA_1,
                            lambda_2=LAMBDA_2,
                            initial_solution=greedy_solution,
                            random_seed=seed,
                            alpha=alpha,
                            gamma=gamma,
                            epsilon_decay_rate=epsilon_decay_rate,
                            eta_reward=eta_reward,
                            C=C,
                            sa_start_pct=0.05,
                            sa_end_pct=0.0001,
                            sa_accept_prob_start=0.5,
                            threshold_start=THRESHOLD_START,
                            segment_size=SEGMENT_SIZE,
                            reaction_factor=REACTION_FACTOR,
                        )

                        runtime = time.time() - start

                        if solution is not None and solution.feasible:
                            cost = round(solution.total_cost, 2)
                            feasible = True
                        else:
                            cost = "N/A"
                            feasible = False

                        writer.writerow([
                            MODEL_NAME,
                            folder_name,
                            instance_label,
                            seed,
                            run_index + 1,
                            cost,
                            round(runtime, 2),
                            feasible
                        ])

                        print(f"{MODEL_NAME} | {instance_label} | run {run_index + 1}/{N_RUNS} | cost: {cost} | time: {round(runtime, 2)}s", flush=True)

                        csv_file.flush()
                        os.fsync(csv_file.fileno())

                except Exception as e:
                    print(f"{instance_name} | ERROR: {e}", flush=True)

    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()