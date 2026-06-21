from dataclasses import dataclass
from pathlib import Path
import math
from Classes import Customer, Instance

# ---------------------------------- #
# Step 1: Create the distance matrix #
# Using Euclidean distances          #
# ---------------------------------- #

def euclidean_distance(c1: Customer, c2: Customer) -> float:
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def build_distance_matrix(nodes: dict[int, Customer]) -> list[list[float]]:
    max_id = max(nodes.keys())
    matrix = [[0.0 for _ in range(max_id + 1)] for _ in range(max_id + 1)]

    for i, node_i in nodes.items():
        for j, node_j in nodes.items():
            matrix[i][j] = euclidean_distance(node_i, node_j)

    return matrix

# ---------------------------------- #
# Step 2: Read the data from the data files
# ---------------------------------- #

def read_mopvrp_instance(file_path) -> Instance:
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # First five values are instance-level parameters
    n_tasks = int(lines[0])
    machines_per_vehicle = int(lines[1])
    n_vehicles = int(lines[2])
    vehicle_capacity = float(lines[3])
    max_duration = float(lines[4])

    # Then come n_tasks customer rows and n_vehicles depot rows
    node_lines_start = 5
    node_lines_end = node_lines_start + n_tasks + n_vehicles
    node_lines = lines[node_lines_start:node_lines_end]

    # Last line is production time per demand unit
    production_coefficient = float(lines[node_lines_end])

    customers = {}
    depot_ids = []

    for row_index, line in enumerate(node_lines):
        parts = line.split()

        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = float(parts[3])
        tw_start = float(parts[4])
        tw_end = float(parts[5])
        service_time = float(parts[6])

        is_depot = row_index >= n_tasks

        production_time = production_coefficient * demand

        customers[node_id] = Customer(
            id=node_id,
            x=x,
            y=y,
            demand=demand,
            tw_start=tw_start,
            tw_end=tw_end,
            service_time=service_time,
            production_time=production_time,
            is_depot=is_depot
        )

        if is_depot:
            depot_ids.append(node_id)

    distance_matrix = build_distance_matrix(customers)

    return Instance(
        name=file_path.stem,
        n_tasks=n_tasks,
        machines_per_vehicle=machines_per_vehicle,
        n_vehicles=n_vehicles,
        vehicle_capacity=vehicle_capacity,
        max_duration=max_duration,
        production_coefficient=production_coefficient,
        customers=customers,
        depot_ids=depot_ids,
        distance_matrix=distance_matrix
    )