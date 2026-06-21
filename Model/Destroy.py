import random
from copy import deepcopy

from Evaluate import evaluate_solution, evaluate_route
from Classes import Route


# -------------------------------------------------
# General helper functions
# -------------------------------------------------

def get_all_customers_in_solution(solution):
    """
    Returns all customers currently present in the solution.
    """

    customers = []

    for route in solution.routes:
        for customer_id in route.customer_sequence:
            customers.append(customer_id)

    return customers


def remove_customer_from_solution(solution, customer_id):
    """
    Removes a customer from whichever route contains it.
    Also removes its machine assignment.
    """

    for route in solution.routes:
        if customer_id in route.customer_sequence:
            route.customer_sequence.remove(customer_id)

            if customer_id in route.machine_assignment:
                del route.machine_assignment[customer_id]

            return solution

    raise ValueError(f"Customer {customer_id} not found in solution.")


def determine_number_to_remove(solution, lambda_1, lambda_2):
    """
    Determines how many customers to remove based on the lambda ratios.

    lambda_1 = minimum removal ratio
    lambda_2 = maximum removal ratio

    Example:
    n = 10, lambda_1 = 0.10, lambda_2 = 0.40
    -> remove between 1 and 4 customers
    """

    all_customers = get_all_customers_in_solution(solution)
    n = len(all_customers)

    min_remove = max(1, int(lambda_1 * n))
    max_remove = max(1, int(lambda_2 * n))

    # Safety check: cannot remove more customers than currently present
    max_remove = min(max_remove, n)

    if min_remove > max_remove:
        min_remove = max_remove

    return random.randint(min_remove, max_remove)

# -------------------------------------------------
# Helper for worst-type removals
# -------------------------------------------------

def calculate_removal_saving(solution, instance, customer_id, saving_type="total"):
    """
    Calculates how much cost is saved by removing one customer.

    saving_type can be:
    - "total"
    - "travel"
    - "delay"
    """

    original_solution = evaluate_solution(
        deepcopy(solution),
        instance,
        check_all_customers=False
    )

    temporary_solution = deepcopy(solution)

    temporary_solution = remove_customer_from_solution(
        temporary_solution,
        customer_id
    )

    temporary_solution = evaluate_solution(
        temporary_solution,
        instance,
        check_all_customers=False
    )

    if saving_type == "total":
        saving = original_solution.total_cost - temporary_solution.total_cost

    elif saving_type == "travel":
        saving = original_solution.total_travel - temporary_solution.total_travel

    elif saving_type == "delay":
        saving = original_solution.total_delay - temporary_solution.total_delay

    else:
        raise ValueError("saving_type must be 'total', 'travel', or 'delay'.")

    return saving


def worst_based_removal(
    solution,
    instance,
    saving_type,
    lambda_1,
    lambda_2,
    seed=None,
    u=3  # <-- Parameter to control randomness (typically 3 to 6)
):
    """
    General worst removal function using the randomized sigma^u |L| selection.
    """
    if seed is not None:
        random.seed(seed)

    destroyed_solution = deepcopy(solution)
    number_to_remove = determine_number_to_remove(
        destroyed_solution,
        lambda_1,
        lambda_2
    )

    removed_customers = []

    for _ in range(number_to_remove):
        # 1. Calculate savings for all currently routed customers.
        #    To stay BIT-IDENTICAL to the original (which deep-copied the whole
        #    solution and called evaluate_solution), we reproduce the exact same
        #    float summation: total cost = sum of route travels + sum of route
        #    delays, summed in route order. We only avoid the deepcopy and the
        #    redundant re-evaluation of unchanged routes.
        routes = destroyed_solution.routes

        base_travel = []
        base_delay = []
        for route in routes:
            rt, rd, _ = evaluate_route(route, instance)
            base_travel.append(rt)
            base_delay.append(rd)

        # Sums accumulated in route order (matches evaluate_solution).
        base_total_travel = 0.0
        base_total_delay = 0.0
        for j in range(len(routes)):
            base_total_travel += base_travel[j]
            base_total_delay += base_delay[j]
        base_total_cost = base_total_travel + base_total_delay

        L = []
        for ri, route in enumerate(routes):
            if not route.customer_sequence:
                continue

            for customer_id in route.customer_sequence:
                new_sequence = [c for c in route.customer_sequence if c != customer_id]
                new_assignment = {
                    c: m for c, m in route.machine_assignment.items()
                    if c != customer_id
                }
                temp_route = Route(
                    vehicle_id=route.vehicle_id,
                    depot_id=route.depot_id,
                    customer_sequence=new_sequence,
                    machine_assignment=new_assignment,
                )
                tr, td, _ = evaluate_route(temp_route, instance)

                # Re-sum in the exact same route order, substituting route ri.
                temp_total_travel = 0.0
                temp_total_delay = 0.0
                for j in range(len(routes)):
                    if j == ri:
                        temp_total_travel += tr
                        temp_total_delay += td
                    else:
                        temp_total_travel += base_travel[j]
                        temp_total_delay += base_delay[j]

                if saving_type == "total":
                    saving = base_total_cost - (temp_total_travel + temp_total_delay)
                elif saving_type == "travel":
                    saving = base_total_travel - temp_total_travel
                elif saving_type == "delay":
                    saving = base_total_delay - temp_total_delay
                else:
                    raise ValueError("saving_type must be 'total', 'travel', or 'delay'.")

                L.append({'customer_id': customer_id, 'saving': saving})

        # 2. Sort descending (highest savings at the top/index 0)
        L.sort(key=lambda x: x['saving'], reverse=True)

        # 3. Apply the sigma^u * |L| formula to pick an index
        sigma = random.random()
        index = int((sigma ** u) * len(L))
        chosen_customer = L[index]['customer_id']

        # 4. Remove the chosen customer
        destroyed_solution = remove_customer_from_solution(
            destroyed_solution,
            chosen_customer
        )

        removed_customers.append(chosen_customer)

    return destroyed_solution, removed_customers


# -------------------------------------------------
# 1. Random removal
# -------------------------------------------------

def random_removal(solution, instance, lambda_1, lambda_2, seed=None):
    """
    Randomly removes q customers from the solution.
    """

    if seed is not None:
        random.seed(seed)

    destroyed_solution = deepcopy(solution)

    all_customers = get_all_customers_in_solution(destroyed_solution)
    number_to_remove = determine_number_to_remove(
        destroyed_solution,
        lambda_1,
        lambda_2
    )

    removed_customers = random.sample(all_customers, number_to_remove)

    for customer_id in removed_customers:
        destroyed_solution = remove_customer_from_solution(
            destroyed_solution,
            customer_id
        )

    return destroyed_solution, removed_customers


# -------------------------------------------------
# 2. Worst removal
# -------------------------------------------------

def worst_removal(solution, instance, lambda_1, lambda_2, seed=None, u=3):
    """
    Removes customers with the highest total cost saving.
    """

    return worst_based_removal(
        solution,
        instance,
        saving_type="total",
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        seed=seed,
        u=u
    )


# -------------------------------------------------
# 3. Worst-distance removal
# -------------------------------------------------

def worst_distance_removal(solution, instance, lambda_1, lambda_2, seed=None, u=3):
    """
    Removes customers with the highest travel distance saving.
    """

    return worst_based_removal(
        solution,
        instance,
        saving_type="travel",
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        seed=seed,
        u=u
    )


# -------------------------------------------------
# 4. Worst-delay removal
# -------------------------------------------------

def worst_delay_removal(solution, instance, lambda_1, lambda_2, seed=None, u=3):
    """
    Removes customers with the highest delay saving.
    """

    return worst_based_removal(
        solution,
        instance,
        saving_type="delay",
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        seed=seed,
        u=u
    )


# -------------------------------------------------
# 5. Geographical removal
# -------------------------------------------------

def geographical_removal(solution, instance, lambda_1, lambda_2, seed=None, u=3):
    """
    Removes one random seed customer and then removes customers
    geographically close to that seed using the randomized selection.
    """
    if seed is not None:
        random.seed(seed)

    destroyed_solution = deepcopy(solution)
    number_to_remove = determine_number_to_remove(
        destroyed_solution,
        lambda_1,
        lambda_2
    )

    all_customers = get_all_customers_in_solution(destroyed_solution)

    seed_customer = random.choice(all_customers)
    removed_customers = [seed_customer]

    destroyed_solution = remove_customer_from_solution(
        destroyed_solution,
        seed_customer
    )

    while len(removed_customers) < number_to_remove:
        remaining_customers = get_all_customers_in_solution(destroyed_solution)

        # 1. Calculate distances from the seed for all remaining customers
        L = []
        for customer_id in remaining_customers:
            distance = instance.distance_matrix[seed_customer][customer_id]
            L.append({'customer_id': customer_id, 'distance': distance})

        # 2. Sort ascending (shortest distance at the top/index 0)
        L.sort(key=lambda x: x['distance'])

        # 3. Apply the sigma^u * |L| formula to pick an index
        sigma = random.random()
        index = int((sigma ** u) * len(L))
        chosen_customer = L[index]['customer_id']

        # 4. Remove the chosen customer
        destroyed_solution = remove_customer_from_solution(
            destroyed_solution,
            chosen_customer
        )

        removed_customers.append(chosen_customer)

    return destroyed_solution, removed_customers


# -------------------------------------------------
# 6. Demand removal
# -------------------------------------------------

def demand_removal(solution, instance, lambda_1, lambda_2, seed=None, u=3):
    """
    Removes one random seed customer and then removes customers
    with similar demand using the randomized selection.
    """
    if seed is not None:
        random.seed(seed)

    destroyed_solution = deepcopy(solution)
    number_to_remove = determine_number_to_remove(
        destroyed_solution,
        lambda_1,
        lambda_2
    )

    all_customers = get_all_customers_in_solution(destroyed_solution)

    seed_customer = random.choice(all_customers)
    seed_demand = instance.customers[seed_customer].demand

    removed_customers = [seed_customer]

    destroyed_solution = remove_customer_from_solution(
        destroyed_solution,
        seed_customer
    )

    while len(removed_customers) < number_to_remove:
        remaining_customers = get_all_customers_in_solution(destroyed_solution)

        # 1. Calculate demand differences for all remaining customers
        L = []
        for customer_id in remaining_customers:
            demand_difference = abs(instance.customers[customer_id].demand - seed_demand)
            L.append({'customer_id': customer_id, 'diff': demand_difference})

        # 2. Sort ascending (smallest difference at the top/index 0)
        L.sort(key=lambda x: x['diff'])

        # 3. Apply the sigma^u * |L| formula to pick an index
        sigma = random.random()
        index = int((sigma ** u) * len(L))
        chosen_customer = L[index]['customer_id']

        # 4. Remove the chosen customer
        destroyed_solution = remove_customer_from_solution(
            destroyed_solution,
            chosen_customer
        )

        removed_customers.append(chosen_customer)

    return destroyed_solution, removed_customers