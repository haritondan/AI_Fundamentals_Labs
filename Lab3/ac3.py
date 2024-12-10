from collections import deque

from solver import solve_sudoku_with_constraints

# Set up arcs (pairs of cells)
def get_arcs():
    arcs = []
    for row in range(9):
        for col in range(9):
            for i in range(9):
                if i != col:
                    arcs.append(((row, col), (row, i)))
                if i != row:
                    arcs.append(((row, col), (i, col)))
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if (r, c) != (row, col):
                        arcs.append(((row, col), (r, c)))
    return arcs

def ac3_algorithm(domains):
    arcs = deque(get_arcs())
    while arcs:
        (xi, xj) = arcs.popleft()
        if revise(domains, xi, xj):
            if len(domains[xi]) == 0:
                return False  
            for xk in neighbors(xi):
                if xk != xj:
                    arcs.append((xk, xi))
    return True

def revise(domains, xi, xj):
    revised = False
    for x in domains[xi].copy():
        if not any(x != y for y in domains[xj]):
            domains[xi].remove(x)
            revised = True
    return revised

# Neighbor function to find cells in the same row, column, or subgrid
def neighbors(cell):
    row, col = cell
    neighbors = set()
    for i in range(9):
        if i != col:
            neighbors.add((row, i))  
        if i != row:
            neighbors.add((i, col))  
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if (r, c) != cell:
                neighbors.add((r, c))  # Same subgrid
    return neighbors


from constraint_propagation import propagate_constraints, initialize_domains, update_grid_with_domains

def solve_sudoku_with_constraints_ac3(grid):
    domains = initialize_domains(grid)
    if not ac3_algorithm(domains):
        return False  

    while update_grid_with_domains(grid, domains):
        propagate_constraints(grid, domains)

    return solve_sudoku_with_constraints(grid)
