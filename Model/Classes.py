from dataclasses import dataclass


@dataclass
class Customer:
    """
    Class representing a customer in the problem instance (one node).
    """
    id: int
    x: float
    y: float
    demand: float
    tw_start: float
    tw_end: float
    service_time: float
    production_time: float
    is_depot: bool = False


@dataclass
class Instance:
    """
    Class representing a problem instance from the dataset 
    (all customers, all depot nodes, vehicle information, machine information, and distances)"
    """
    name: str
    n_tasks: int
    machines_per_vehicle: int
    n_vehicles: int
    vehicle_capacity: float
    max_duration: float
    production_coefficient: float
    customers: dict[int, Customer]
    depot_ids: list[int]
    distance_matrix: list[list[float]]


@dataclass
class Route:
    """
    Class representing a route for a single vehicle.
    """
    vehicle_id: int
    depot_id: int
    customer_sequence: list
    machine_assignment: dict


@dataclass
class Solution:
    """
    Class representing a complete solution (all routes, total travel time, total delay, total cost, and feasibility).
    """
    routes: list
    total_travel: float = 0.0
    total_delay: float = 0.0
    total_cost: float = 0.0
    feasible: bool = True