from Evaluate import evaluate_route, route_load


def print_solution_summary(solution, instance):
    print("\n==============================")
    print("SOLUTION SUMMARY")
    print("==============================")
    print("Instance:", instance.name)
    print("Feasible:", solution.feasible)
    print("Total travel:", round(solution.total_travel, 2))
    print("Total delay:", round(solution.total_delay, 2))
    print("Total cost:", round(solution.total_cost, 2))

    for route in solution.routes:
        route_travel, route_delay, route_feasible = evaluate_route(route, instance)

        print("\nVehicle", route.vehicle_id)
        print("Depot:", route.depot_id)
        print("Route:", route.customer_sequence)
        print("Load:", round(route_load(route, instance), 2))
        print("Travel:", round(route_travel, 2))
        print("Delay:", round(route_delay, 2))
        print("Cost:", round(route_travel + route_delay, 2))
        print("Feasible:", route_feasible)

        machine_usage = {}

        for customer_id in route.customer_sequence:
            machine_id = route.machine_assignment[customer_id]

            if machine_id not in machine_usage:
                machine_usage[machine_id] = []

            machine_usage[machine_id].append(customer_id)

        print("Machine usage:")
        for machine_id in range(instance.machines_per_vehicle):
            assigned = machine_usage.get(machine_id, [])
            print("  Machine", machine_id, ":", assigned)