import sys
import random
from Classes import Route, Solution
from Evaluate import evaluate_solution
from tqdm import tqdm


def create_empty_solution(instance):
    """
    Creates one empty route per vehicle, initialized with the depot.
    """
    routes = []
    for vehicle_id in range(instance.n_vehicles):
        depot_id = instance.depot_ids[vehicle_id] if vehicle_id < len(instance.depot_ids) else instance.depot_ids[0]
        route = Route(
            vehicle_id=vehicle_id,
            depot_id=depot_id,
            customer_sequence=[],
            machine_assignment={}
        )
        routes.append(route)
    return Solution(routes=routes)


def insert_customer_in_route(route, customer_id, position, machine_id):
    """
    Inserts the customer permanently into the chosen route, position, and machine.
    """
    route.customer_sequence.insert(position, customer_id)
    route.machine_assignment[customer_id] = machine_id


def find_best_insertion(solution, instance, customer_id, apply_noise=False, max_N=0.0):
    """
    Implementation of Algorithm 2: The insertion strategy for the MoP-VRP.
    Achieves exact logic and O(mn^2) efficiency by calculating increments locally.
    """
    best_route_index = None
    best_position = None
    best_machine_id = None
    best_cost_increment = float("inf")
    
    new_cust = instance.customers[customer_id]

    # Algorithm 2, Line 2: for each Route r do
    for route_index, route in enumerate(solution.routes):
        
        # 1. Capacity Filter (The only hard filter during greedy construction)
        current_load = sum(instance.customers[c].demand for c in route.customer_sequence)
        if current_load + new_cust.demand > instance.vehicle_capacity:
            continue

        # 2. Pre-calculate the base state of the route at every position p
        # This prevents recalculating the start of the route from time 0 for every test.
        state_at_p = [] 
        machine_times = [0.0] * instance.machines_per_vehicle
        current_time = 0.0
        current_node = route.depot_id
        total_base_delay = 0.0
        
        for p in range(len(route.customer_sequence) + 1):
            # Save a snapshot of the timeline exactly before position p
            state_at_p.append((current_time, current_node, list(machine_times), total_base_delay))
            
            if p < len(route.customer_sequence):
                c_id = route.customer_sequence[p]
                m_id = route.machine_assignment[c_id]
                c_obj = instance.customers[c_id]
                
                arr = current_time + instance.distance_matrix[current_node][c_id]
                prod_finish = machine_times[m_id] + c_obj.production_time
                machine_times[m_id] = prod_finish
                
                start = max(arr, prod_finish, c_obj.tw_start)
                total_base_delay += max(0.0, start - c_obj.tw_end)
                
                current_time = start + c_obj.service_time
                current_node = c_id

        # Algorithm 2, Line 3: for each position p in Route r do
        for position in range(len(route.customer_sequence) + 1):
            
            # Algorithm 2, Line 4: Calculate travel increment on position p
            prev_node = route.depot_id if position == 0 else route.customer_sequence[position - 1]
            next_node = route.depot_id if position == len(route.customer_sequence) else route.customer_sequence[position]
            
            delta_travel = (instance.distance_matrix[prev_node][customer_id] + 
                            instance.distance_matrix[customer_id][next_node] - 
                            instance.distance_matrix[prev_node][next_node])
            
            # Fetch the pre-calculated timeline snapshot exactly before this position
            base_time, base_node, base_m_times, base_delay_so_far = state_at_p[position]
            
            # Algorithm 2, Line 5: for each Machine l on Route r do
            for machine_id in range(instance.machines_per_vehicle):
                
                # Load the machine states. 
                # (Line 6 implicitly handled: base_m_times[machine_id] is the exact finish time 
                # of the last customer on this machine before position p).
                m_times = list(base_m_times) 
                
                # Evaluate the new customer's insertion
                arr = base_time + instance.distance_matrix[base_node][customer_id]
                prod_finish = m_times[machine_id] + new_cust.production_time
                m_times[machine_id] = prod_finish
                
                start = max(arr, prod_finish, new_cust.tw_start)
                new_route_delay = base_delay_so_far + max(0.0, start - new_cust.tw_end)
                
                curr_time = start + new_cust.service_time
                curr_node = customer_id
                
                # Algorithm 2, Line 7: Compute delay increment for customers AFTER position p
                for i in range(position, len(route.customer_sequence)):
                    next_c = route.customer_sequence[i]
                    next_m = route.machine_assignment[next_c]
                    next_obj = instance.customers[next_c]
                    
                    arr = curr_time + instance.distance_matrix[curr_node][next_c]
                    prod_finish = m_times[next_m] + next_obj.production_time
                    m_times[next_m] = prod_finish
                    
                    start = max(arr, prod_finish, next_obj.tw_start)
                    new_route_delay += max(0.0, start - next_obj.tw_end)
                    
                    curr_time = start + next_obj.service_time
                    curr_node = next_c
                
                # Calculate base cost increment 
                delta_delay = new_route_delay - total_base_delay
                cost_increment = delta_travel + delta_delay
                
                # 2. APPLY NOISE HERE
                if apply_noise and max_N > 0:
                    noise = random.uniform(-max_N, max_N)
                    cost_increment = max(0.0, cost_increment + noise)
                
                # 3. Compare the (potentially noisy) cost increment
                if cost_increment < best_cost_increment:
                    best_cost_increment = cost_increment
                    best_route_index = route_index
                    best_position = position
                    best_machine_id = machine_id

    return best_route_index, best_position, best_machine_id, best_cost_increment


def greedy_insertion(instance, print_steps=True, print_warning=True):
    """
    Main loop for greedy initialization.
    """
    solution = create_empty_solution(instance)
    unassigned_customers = list(range(instance.n_tasks))
    
    progress_bar = tqdm(total=len(unassigned_customers), disable=not print_steps, file=sys.stdout, desc="Greedy insertion")
    iteration = 1

    # Algorithm 2, Line 1: for each non-inserted Customer i do
    while len(unassigned_customers) > 0:

        best_customer = None
        best_route = None
        best_pos = None
        best_mach = None
        global_best_increment = float("inf")

        # Find the absolute best insertion across all unassigned customers
        for customer_id in unassigned_customers:
            r_idx, pos, m_id, cost_inc = find_best_insertion(solution, instance, customer_id)

            if cost_inc < global_best_increment:
                best_customer = customer_id
                best_route = r_idx
                best_pos = pos
                best_mach = m_id
                global_best_increment = cost_inc

        if best_customer is None:
            raise Exception("Greedy construction failed: No capacity left for remaining customers.")

        # Protect against None values or a stalled greedy phase
        if best_customer is None or best_route is None:
            raise Exception("Greedy construction failed: No valid vehicle routing assignment could be determined.")

        # Apply the best insertion permanently
        insert_customer_in_route(
            solution.routes[best_route], 
            best_customer, 
            best_pos, 
            best_mach
        )
        
        unassigned_customers.remove(best_customer)
        progress_bar.update(1)
        iteration += 1

    progress_bar.close()
    
    # Final evaluation ensures all total costs and feasibility flags (like max_duration) are properly updated
    solution = evaluate_solution(solution, instance, check_all_customers=True)

    if not solution.feasible and print_warning:
        print("\nNote: Initial greedy solution exceeds max duration limits. ALNS will repair this.")

    return solution