def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end="")
        print()

def read_sudoku_from_file(filename):
    grid = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Remove whitespace, read each character, and convert '*' to 0
                row = [0 if ch == '*' else int(ch) for ch in line.strip()]
                grid.append(row)
        # Validate the grid format
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            raise ValueError("Invalid Sudoku grid format in file")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return grid


def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if arr[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False

def used_in_row(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False

def used_in_col(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False

def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[i + row][j + col] == num:
                return True
    return False

def check_location_is_safe(arr, row, col, num):
    return (not used_in_row(arr, row, num) and
            not used_in_col(arr, col, num) and
            not used_in_box(arr, row - row % 3, col - col % 3, num))


def is_valid_sudoku(grid):
    for i in range(9):
        row = [num for num in grid[i] if num != 0]
        col = [grid[j][i] for j in range(9) if grid[j][i] != 0]
        if len(row) != len(set(row)) or len(col) != len(set(col)):
            return False

    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            subgrid = [
                grid[r][c]
                for r in range(row, row + 3)
                for c in range(col, col + 3)
                if grid[r][c] != 0
            ]
            if len(subgrid) != len(set(subgrid)):
                return False

    return True

def is_solvable_sudoku(grid, solving_function):
    if not is_valid_sudoku(grid):
        return False

    grid_copy = [row[:] for row in grid]  
    return solving_function(grid_copy)