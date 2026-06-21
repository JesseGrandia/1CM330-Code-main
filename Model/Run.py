from pathlib import Path

from Read_data import read_mopvrp_instance
from Greedy_insertion import greedy_insertion
from ALNS import alns
from Output import print_solution_summary


def main():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "MoPVRP-instance" / "Benchmark" / "MPRP_c++_10_2_3" / "RC103.dat"

    lambda_1 = 0.10
    lambda_2 = 0.40
    random_seed = 42
    iterations = 3000

    instance = read_mopvrp_instance(file_path)

    greedy_solution = greedy_insertion(instance, print_steps=False)

    print("\nOriginal greedy solution:")
    print_solution_summary(greedy_solution, instance)
    method = "qlearning" 

    alns_solution = alns(
    instance,
    iterations=iterations,
    lambda_1=lambda_1,
    lambda_2=lambda_2,
    initial_solution=greedy_solution,
    random_seed=random_seed,
    sa_start_pct=0.05,
    sa_end_pct=0.0001,
    sa_accept_prob_start=0.5,
)

    # print("\nBest ALNS solution:")
    # print_solution_summary(alns_solution, instance)


if __name__ == "__main__":
    main()