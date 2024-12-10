from constraint_propagation import propagate_constraints, initialize_domains, update_grid_with_domains
from utils import find_empty_location, check_location_is_safe

def solve_sudoku(grid):
    l = [0, 0]
    
    if not find_empty_location(grid, l):
        return True
    
    row, col = l[0], l[1]
    
    for num in range(1, 10):
        if check_location_is_safe(grid, row, col, num):
            grid[row][col] = num
            if solve_sudoku(grid):
                return True
            grid[row][col] = 0
    
    return False

def solve_sudoku_with_constraints(grid):
    domains = initialize_domains(grid)
    propagate_constraints(grid, domains)

    while update_grid_with_domains(grid, domains):
        propagate_constraints(grid, domains)

    return solve_sudoku(grid)
