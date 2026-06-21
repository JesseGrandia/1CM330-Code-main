import math
import random
from copy import deepcopy
from itertools import product

from QAgent import QAgent
from AOS import AOS
from Greedy_insertion import greedy_insertion
from Destroy import (
    random_removal,
    worst_removal,
    worst_distance_removal,
    worst_delay_removal,
    geographical_removal,
    demand_removal,
)
from Repair import (
    regret_1_repair,
    regret_2_repair,
    regret_3_repair,
    regret_4_repair,
)
from Evaluate import evaluate_solution


# ---------------------------------------------------------------------------
# Helper for the Q-learning state
# ---------------------------------------------------------------------------
def get_state(iteration, max_iterations, stagnation_counter, stagnation_limit):
    """Returns a discrete state string for the Q-learning agent."""
    temp_state = "high_temp" if (iteration / max_iterations) < 0.5 else "low_temp"
    stag_state = "improving" if stagnation_counter < stagnation_limit else "stagnant"
    return f"{temp_state}_{stag_state}"


# ---------------------------------------------------------------------------
# Simulated Annealing acceptance (used by the "qlearning" hybrid mode)
# ---------------------------------------------------------------------------
def sa_accept(candidate_cost, current_cost, temperature):
    """
    Accept if the candidate is not worse.
    If the candidate is worse, accept with probability exp(-delta / T).
    """
    if candidate_cost <= current_cost:
        return True

    if temperature <= 1e-12:
        return False

    delta = candidate_cost - current_cost
    exponent = -delta / temperature

    # Prevent numerical underflow for very negative exponents
    if exponent < -700:
        return False

    probability = math.exp(exponent)
    return random.random() < probability


def linear_temperature(iteration, max_iterations, start_temperature, end_temperature):
    """
    Linearly decreases the temperature from start_temperature to end_temperature.
    """
    if max_iterations <= 1:
        return end_temperature

    progress = (iteration - 1) / (max_iterations - 1)
    temperature = start_temperature - progress * (start_temperature - end_temperature)

    return max(end_temperature, temperature)


def compute_sa_temperatures(
    initial_cost,
    sa_start_pct=0.05,
    sa_end_pct=0.0001,
    sa_accept_prob_start=0.5
):
    """
    Computes instance-scaled SA start and end temperatures.

    The start temperature is chosen such that a solution that is
    sa_start_pct worse than the initial solution is accepted with
    probability sa_accept_prob_start at the start of the search.
    """
    if initial_cost <= 0:
        return 1.0, 1e-9

    if not 0 < sa_accept_prob_start < 1:
        raise ValueError("sa_accept_prob_start must be between 0 and 1.")

    denominator = -math.log(sa_accept_prob_start)

    start_temp = (sa_start_pct * initial_cost) / denominator
    end_temp = (sa_end_pct * initial_cost) / denominator

    return start_temp, max(end_temp, 1e-9)


# ---------------------------------------------------------------------------
# Threshold acceptance (used by the "base" mode, Dueck & Scheuer 1990)
# ---------------------------------------------------------------------------
def linear_threshold(iteration, max_iterations, threshold_start):
    """
    Linearly decreases the acceptance threshold from threshold_start to 0
    over the course of the search (paper Algorithm 1, line 18).
    threshold_start is expressed as a fraction (e.g. 0.10 == 10%).
    """
    if max_iterations <= 1:
        return 0.0
    return threshold_start * (max_iterations - iteration) / (max_iterations - 1)


# ---------------------------------------------------------------------------
# Operator definitions shared by both modes
# ---------------------------------------------------------------------------
def _build_operators():
    """Returns the active destroy, repair and noise operator dictionaries."""
    destroy_operators = {
        "Random removal": random_removal,
        "Worst removal": worst_removal,
        "Geographical removal": geographical_removal,
        "Demand removal": demand_removal,
    }
    repair_operators = {
        "Regret-2": regret_2_repair,
        "Regret-3": regret_3_repair,
        "Regret-4": regret_4_repair,
    }
    noise_options = {"With_Noise": True, "No_Noise": False}
    return destroy_operators, repair_operators, noise_options


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def alns(
    instance,
    iterations,
    mode="qlearning",
    # --- shared ---
    lambda_1=0.10,
    lambda_2=0.40,
    initial_solution=None,
    random_seed=1,
    # --- qlearning (hybrid) params: Q-learning + Simulated Annealing ---
    alpha=0.5,
    gamma=0.7,
    epsilon_decay_rate=0.99,
    eta_reward=0.6,
    C=1.0,
    sa_start_pct=0.05,
    sa_end_pct=0.0001,
    sa_accept_prob_start=0.5,
    # --- base params: AOS roulette wheel + threshold acceptance ---
    threshold_start=0.10,
    segment_size=100,
    reaction_factor=0.1,
):
    """
    ALNS for the MoP-VRP with two selectable modes:

      mode="qlearning" : Hybrid. A single Q-learning agent selects a full
                         COMBINATION of (destroy, repair, noise) as one action,
                         and acceptance is governed by Simulated Annealing.

      mode="base"      : Pure metaheuristic from Wang et al. (2023). Three
                         independent AOS roulette-wheel selectors with adaptive
                         weights (destroy / repair / noise), and threshold
                         acceptance.
    """
    if mode not in ("qlearning", "base"):
        raise ValueError("mode must be 'qlearning' or 'base'.")

    random.seed(random_seed)

    # 1. Initial solution (shared)
    if initial_solution is None:
        current_solution = greedy_insertion(
            instance, print_steps=False, print_warning=False
        )
    else:
        current_solution = deepcopy(initial_solution)

    best_solution = None
    if current_solution.feasible:
        best_solution = deepcopy(current_solution)

    # Noise magnitude (shared)
    eta_dist = 0.025
    max_distance = max(max(row) for row in instance.distance_matrix)
    max_N = eta_dist * max_distance

    destroy_operators, repair_operators, noise_options = _build_operators()

    if mode == "qlearning":
        return _run_qlearning(
            instance, iterations, current_solution, best_solution, max_N,
            destroy_operators, repair_operators, noise_options,
            lambda_1, lambda_2, alpha, gamma, epsilon_decay_rate,
            eta_reward, C, sa_start_pct, sa_end_pct, sa_accept_prob_start,
        )
    else:
        return _run_base(
            instance, iterations, current_solution, best_solution, max_N,
            destroy_operators, repair_operators, noise_options,
            lambda_1, lambda_2, threshold_start, segment_size, reaction_factor,
        )


# ---------------------------------------------------------------------------
# qlearning mode: Q-learning combination agent + Simulated Annealing
# ---------------------------------------------------------------------------
def _run_qlearning(
    instance, iterations, current_solution, best_solution, max_N,
    destroy_operators, repair_operators, noise_options,
    lambda_1, lambda_2, alpha, gamma, epsilon_decay_rate,
    eta_reward, C, sa_start_pct, sa_end_pct, sa_accept_prob_start,
):
    # Build the combined action space: destroy x repair x noise.
    # Each action maps a readable name to the (destroy_fn, repair_fn, noise_flag)
    # triple that should be applied when that action is chosen, so the reward is
    # assigned to the combination that was actually applied.
    combination_operators = {}
    for (d_name, d_fn), (r_name, r_fn), (n_name, n_flag) in product(
        destroy_operators.items(),
        repair_operators.items(),
        noise_options.items(),
    ):
        combo_name = f"{d_name} | {r_name} | {n_name}"
        combination_operators[combo_name] = (d_fn, r_fn, n_flag)

    sel_combo = QAgent(
        operators=combination_operators,
        alpha=alpha,
        gamma=gamma,
        epsilon_decay_rate=epsilon_decay_rate,
        eta_reward=eta_reward,
        C=C,
    )

    stagnation_limit = max(100, iterations // 50)
    improving_lambda = (lambda_1, lambda_2)
    stagnant_lambda = (min(0.25, lambda_2), lambda_2)

    start_temp, end_temp = compute_sa_temperatures(
        initial_cost=current_solution.total_cost,
        sa_start_pct=sa_start_pct,
        sa_end_pct=sa_end_pct,
        sa_accept_prob_start=sa_accept_prob_start,
    )

    stagnation_counter = 0

    for iteration in range(1, iterations + 1):

        current_state = get_state(
            iteration, iterations, stagnation_counter, stagnation_limit
        )

        combo_name, combo = sel_combo.select_operator(current_state)
        destroy_operator, repair_operator, apply_noise = combo

        if "stagnant" in current_state:
            current_lambda_1, current_lambda_2 = stagnant_lambda
        else:
            current_lambda_1, current_lambda_2 = improving_lambda

        destroyed_solution, removed_customers = destroy_operator(
            current_solution, instance,
            lambda_1=current_lambda_1, lambda_2=current_lambda_2, seed=None,
        )
        destroyed_solution = evaluate_solution(
            destroyed_solution, instance, check_all_customers=False
        )

        candidate_solution = repair_operator(
            destroyed_solution, instance, removed_customers,
            apply_noise=apply_noise, max_N=max_N,
        )

        temperature = linear_temperature(
            iteration=iteration,
            max_iterations=iterations,
            start_temperature=start_temp,
            end_temperature=end_temp,
        )

        candidate_cost = candidate_solution.total_cost
        current_cost = current_solution.total_cost
        best_cost = best_solution.total_cost if best_solution else current_cost

        if not candidate_solution.feasible:
            reward = -5
            stagnation_counter += 1
        else:
            delta_global = max((best_cost - candidate_cost) / best_cost, 0.0)
            delta_local = max((current_cost - candidate_cost) / current_cost, 0.0)

            delta_improvement = (eta_reward * delta_global) + ((1 - eta_reward) * delta_local)

            if delta_global > 0 or delta_local > 0:
                reward = (delta_improvement * iteration) / C
            else:
                if sa_accept(candidate_cost, current_cost, temperature):
                    reward = 0 if stagnation_counter >= stagnation_limit else 0.5
                else:
                    reward = -2

        if not candidate_solution.feasible:
            pass
        elif best_solution is None or candidate_cost < best_cost:
            best_solution = deepcopy(candidate_solution)
            current_solution = deepcopy(candidate_solution)
            stagnation_counter = 0
        else:
            delta = candidate_cost - current_cost
            epsilon_tol = 1e-9

            if delta < -epsilon_tol:
                current_solution = deepcopy(candidate_solution)
                stagnation_counter = 0
            elif abs(delta) <= epsilon_tol:
                current_solution = deepcopy(candidate_solution)
                stagnation_counter += 1
            elif sa_accept(candidate_cost, current_cost, temperature):
                current_solution = deepcopy(candidate_solution)
                stagnation_counter += 1
            else:
                stagnation_counter += 1

        next_state = get_state(
            iteration + 1, iterations, stagnation_counter, stagnation_limit
        )

        sel_combo.update_q_value(reward, next_state)
        sel_combo.decay_epsilon()

    return best_solution


# ---------------------------------------------------------------------------
# base mode: AOS roulette wheel + threshold acceptance (pure metaheuristic)
# ---------------------------------------------------------------------------
def _run_base(
    instance, iterations, current_solution, best_solution, max_N,
    destroy_operators, repair_operators, noise_options,
    lambda_1, lambda_2, threshold_start, segment_size, reaction_factor,
):
    # Three independent adaptive selectors, exactly as in the paper: destroy,
    # repair and noise each have their own roulette-wheel weights that are
    # updated every segment_size iterations.
    aos_destroy = AOS(destroy_operators, segment_size=segment_size,
                      reaction_factor=reaction_factor)
    aos_repair = AOS(repair_operators, segment_size=segment_size,
                     reaction_factor=reaction_factor)
    aos_noise = AOS(noise_options, segment_size=segment_size,
                    reaction_factor=reaction_factor)

    for iteration in range(1, iterations + 1):

        destroy_name, destroy_operator = aos_destroy.select_operator()
        repair_name, repair_operator = aos_repair.select_operator()
        noise_name, apply_noise = aos_noise.select_operator()

        destroyed_solution, removed_customers = destroy_operator(
            current_solution, instance,
            lambda_1=lambda_1, lambda_2=lambda_2, seed=None,
        )
        destroyed_solution = evaluate_solution(
            destroyed_solution, instance, check_all_customers=False
        )

        candidate_solution = repair_operator(
            destroyed_solution, instance, removed_customers,
            apply_noise=apply_noise, max_N=max_N,
        )

        candidate_cost = candidate_solution.total_cost
        current_cost = current_solution.total_cost
        best_cost = best_solution.total_cost if best_solution is not None else float("inf")

        # Threshold acceptance: accept a candidate if its gap above the best
        # solution found so far is below the (linearly shrinking) threshold T.
        T = linear_threshold(iteration, iterations, threshold_start)

        score_type = None
        accept = False

        if candidate_solution.feasible:
            if best_solution is None or candidate_cost < best_cost:
                accept = True
                score_type = "global_best"
            else:
                gap = (candidate_cost - best_cost) / best_cost if best_cost > 0 else 0.0
                accept = gap < T
                if accept:
                    score_type = "better" if candidate_cost < current_cost else "accepted"

            if accept:
                current_solution = deepcopy(candidate_solution)
                if best_solution is None or candidate_cost < best_cost:
                    best_solution = deepcopy(candidate_solution)

        elif best_solution is None:
            # No feasible solution found yet: accept infeasible candidates as
            # current_solution if they reduce total cost, so the search can
            # keep moving and reach feasibility from different positions.
            # best_solution is NOT updated here — it stays None until a
            # feasible candidate is found.
            if candidate_cost < current_cost:
                current_solution = deepcopy(candidate_solution)

        # Update adaptive weights for all three selectors with the same outcome
        # score (standard ALNS credit assignment).
        for agent, op_name in (
            (aos_destroy, destroy_name),
            (aos_repair, repair_name),
            (aos_noise, noise_name),
        ):
            if score_type is not None:
                agent.update_score(op_name, score_type)
            agent.end_of_iteration_update()

    return best_solution