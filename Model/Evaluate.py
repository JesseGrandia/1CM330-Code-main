from Classes import Route, Solution


def route_load(route, instance):
    """
    Computes total demand on a route.
    """
    total_demand = 0.0

    for customer_id in route.customer_sequence:
        total_demand += instance.customers[customer_id].demand

    return total_demand


def evaluate_route(route, instance, print_details=False):
    """
    Evaluates one MoP-VRP route.

    Important:
    - TW_start is strict: service cannot start before it.
    - TW_end is soft: violation creates delay cost.
    - Max duration is checked as the absolute return time to the depot:
      return_time <= instance.max_duration.
    """

    # 1. Capacity check
    load = route_load(route, instance)

    if load > instance.vehicle_capacity:
        return float("inf"), float("inf"), False

    # 2. Initialize route state
    current_node = route.depot_id
    current_time = 0.0

    total_travel = 0.0
    total_delay = 0.0
    total_service_time = 0.0

    machine_available_time = [
        0.0 for _ in range(instance.machines_per_vehicle)
    ]

    if print_details:
        print("\nEvaluating route for vehicle", route.vehicle_id)
        print("Depot:", route.depot_id)
        print("Sequence:", route.customer_sequence)

    # 3. Visit customers in route order
    for customer_id in route.customer_sequence:
        customer = instance.customers[customer_id]
        machine_id = route.machine_assignment[customer_id]

        travel_time = instance.distance_matrix[current_node][customer_id]
        arrival_time = current_time + travel_time
        total_travel += travel_time

        production_start = machine_available_time[machine_id]
        production_finish = production_start + customer.production_time
        machine_available_time[machine_id] = production_finish

        service_start = max(
            arrival_time,
            production_finish,
            customer.tw_start
        )

        delay = max(0.0, service_start - customer.tw_end)
        total_delay += delay

        service_finish = service_start + customer.service_time
        total_service_time += customer.service_time

        if print_details:
            print("\nCustomer:", customer_id)
            print("  Machine:", machine_id)
            print("  Arrival time:", round(arrival_time, 2))
            print("  Production start:", round(production_start, 2))
            print("  Production finish:", round(production_finish, 2))
            print("  TW:", customer.tw_start, "-", customer.tw_end)
            print("  Service start:", round(service_start, 2))
            print("  Delay:", round(delay, 2))
            print("  Service finish:", round(service_finish, 2))

        current_time = service_finish
        current_node = customer_id

    # 4. Return to depot
    travel_back = instance.distance_matrix[current_node][route.depot_id]
    total_travel += travel_back

    # Revert to original — this was always correct
    return_time = current_time + travel_back
    feasible = return_time <= instance.max_duration

    if print_details:
        print("\nReturn to depot")
        print("  Travel back:", round(travel_back, 2))
        print("  Return time:", round(return_time, 2))
        print("  Max duration:", instance.max_duration)
        print("  Feasible:", feasible)
        print("  Route travel:", round(total_travel, 2))
        print("  Route service time:", round(total_service_time, 2))
        print("  Route delay:", round(total_delay, 2))
        print("  Route cost:", round(total_travel + total_delay, 2))

    return total_travel, total_delay, feasible


def evaluate_solution(solution, instance, check_all_customers=True):
    """
    Evaluates a full or partial solution.

    If check_all_customers=True:
        The function checks whether every customer is visited exactly once.
        Use this for final solutions.

    If check_all_customers=False:
        The function only checks route feasibility and duplicate visits.
        Use this during greedy construction.
    """

    total_travel = 0.0
    total_delay = 0.0
    feasible = True

    visited_customers = set()

    for route in solution.routes:
        route_travel, route_delay, route_feasible = evaluate_route(route, instance)

        total_travel += route_travel
        total_delay += route_delay

        if not route_feasible:
            feasible = False

        for customer_id in route.customer_sequence:
            if customer_id in visited_customers:
                feasible = False
            visited_customers.add(customer_id)

    # Only check complete coverage when requested
    if check_all_customers:
        expected_customers = set(range(instance.n_tasks))
        actual_customers = set(visited_customers)

        if expected_customers != actual_customers:
            feasible = False

    solution.total_travel = total_travel
    solution.total_delay = total_delay
    solution.total_cost = total_travel + total_delay
    solution.feasible = feasible

    return solution