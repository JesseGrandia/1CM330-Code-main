from Evaluate import evaluate_solution
from Greedy_insertion import find_best_insertion, insert_customer_in_route
from Classes import Solution
from copy import deepcopy

def regret_k_repair(partial_solution, instance, removed_customers, k, apply_noise=False, max_N=0.0):
    """
    Generalized Regret-k repair method.
    Calculates the regret value as the sum of differences between the k-th best 
    routes and the absolute best route. Inserts the customer with the highest regret.
    """
    solution = partial_solution
    unassigned_customers = removed_customers.copy()

    while len(unassigned_customers) > 0:
        best_customer = None
        best_insertion_params = None
        max_regret = -float("inf")

        for customer_id in unassigned_customers:
            
            # Find the best insertion cost for THIS customer in EVERY route
            route_costs = []
            
            for route_index, route in enumerate(solution.routes):
                # find_best_insertion only reads the solution, so we can wrap
                # the single route in a lightweight Solution instead of deep-
                # copying the entire current solution for every route.
                temp_solution = Solution(routes=[route])
                
                r_idx, pos, m_id, cost = find_best_insertion(
                    temp_solution, 
                    instance, 
                    customer_id, 
                    apply_noise, 
                    max_N
                )
                
                if cost != float("inf"):
                    route_costs.append({
                        "route_index": route_index, 
                        "position": pos, 
                        "machine_id": m_id, 
                        "cost": cost
                    })

            # Sort routes by lowest cost increment
            route_costs.sort(key=lambda x: x["cost"])

            if len(route_costs) == 0:
                raise Exception(f"Repair failed: no feasible insertion found for customer {customer_id}.")

            best_cost = route_costs[0]["cost"]
            regret_value = 0
            
            # Calculate regret-k value (Sum of differences up to k-th best route)
            for h in range(1, min(k, len(route_costs))):
                regret_value += (route_costs[h]["cost"] - best_cost)

            # Tie-breaking: pick the one with max regret. If tied, pick the one with lowest best_cost.
            if regret_value > max_regret or (regret_value == max_regret and (best_insertion_params is None or best_cost < best_insertion_params["cost"])):
                max_regret = regret_value
                best_customer = customer_id
                best_insertion_params = route_costs[0]

        # If no customer could be chosen for insertion, raise an explicit error
        if best_customer is None or best_insertion_params is None:
            raise Exception("Repair failed: No valid customer insertion parameters could be determined.")

        # Apply the permanent insertion for the customer we'd regret not inserting the most
        insert_customer_in_route(
            solution.routes[best_insertion_params["route_index"]],
            best_customer,
            best_insertion_params["position"],
            best_insertion_params["machine_id"]
        )

        # Update the timeline and feasibility
        solution = evaluate_solution(solution, instance, check_all_customers=False)
        unassigned_customers.remove(best_customer)

    return evaluate_solution(solution, instance, check_all_customers=True)

# Define the 4 wrappers to act as independent operators for the AOS
def regret_1_repair(solution, instance, removed_customers, apply_noise=False, max_N=0.0):
    return regret_k_repair(solution, instance, removed_customers, k=1, apply_noise=apply_noise, max_N=max_N)

def regret_2_repair(solution, instance, removed_customers, apply_noise=False, max_N=0.0):
    return regret_k_repair(solution, instance, removed_customers, k=2, apply_noise=apply_noise, max_N=max_N)

def regret_3_repair(solution, instance, removed_customers, apply_noise=False, max_N=0.0):
    return regret_k_repair(solution, instance, removed_customers, k=3, apply_noise=apply_noise, max_N=max_N)

def regret_4_repair(solution, instance, removed_customers, apply_noise=False, max_N=0.0):
    return regret_k_repair(solution, instance, removed_customers, k=4, apply_noise=apply_noise, max_N=max_N)