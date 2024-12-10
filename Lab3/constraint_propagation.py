
def initialize_domains(grid):
    domains = {}
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                domains[(row, col)] = set(range(1, 10))
            else:
                domains[(row, col)] = {grid[row][col]}
    return domains

def propagate_constraints(grid, domains):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:  
                used_values = set()

                used_values.update(grid[row][i] for i in range(9) if grid[row][i] != 0)
                used_values.update(grid[i][col] for i in range(9) if grid[i][col] != 0)

                start_row, start_col = row - row % 3, col - col % 3
                for i in range(3):
                    for j in range(3):
                        if grid[start_row + i][start_col + j] != 0:
                            used_values.add(grid[start_row + i][start_col + j])

                domains[(row, col)] -= used_values

    return domains

def update_grid_with_domains(grid, domains):
    changed = False
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0 and len(domains[(row, col)]) == 1:
                # If a cell has only one possible value in its domain, assign it
                grid[row][col] = domains[(row, col)].pop()
                changed = True
    return changed
