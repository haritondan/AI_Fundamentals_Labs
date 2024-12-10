from constraint_propagation import propagate_constraints, initialize_domains, update_grid_with_domains
from utils import check_location_is_safe

def find_mrv_location(domains, grid):
    min_domain_size = 10  
    mrv_cell = None

    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:  
                domain_size = len(domains[(row, col)])
                if 1 < domain_size < min_domain_size:
                    min_domain_size = domain_size
                    mrv_cell = (row, col)
    
    return mrv_cell

def solve_sudoku_with_mrv_and_constraints(grid):
    domains = initialize_domains(grid)
    propagate_constraints(grid, domains)

    while update_grid_with_domains(grid, domains):
        propagate_constraints(grid, domains)


    return mrv_solver(grid, domains)

def mrv_solver(grid, domains):
    mrv_cell = find_mrv_location(domains, grid)
    if mrv_cell is None:
        return True  

    row, col = mrv_cell
    for num in domains[(row, col)]:
        if check_location_is_safe(grid, row, col, num):
            grid[row][col] = num  
            new_domains = propagate_constraints(grid, initialize_domains(grid))
            update_grid_with_domains(grid, new_domains)  

            if mrv_solver(grid, new_domains):
                return True
            grid[row][col] = 0  

    return False

