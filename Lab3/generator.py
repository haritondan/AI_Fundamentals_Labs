import random
def is_safe(grid, row, col, num):
    if num in grid[row]:
        return False
    if num in [grid[r][col] for r in range(9)]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if grid[r][c] == num:
                return False
    return True

def fill_grid(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                numbers = list(range(1, 10))
                random.shuffle(numbers)
                for num in numbers:
                    if is_safe(grid, row, col, num):
                        grid[row][col] = num
                        if fill_grid(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

def generate_full_sudoku_grid():
    grid = [[0 for _ in range(9)] for _ in range(9)]
    fill_grid(grid)
    return grid

def create_unsolved_sudoku(grid, clues=30):
    puzzle = [row[:] for row in grid]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    removed = 0
    target_removals = 81 - clues
    for row, col in cells:
        if removed >= target_removals:
            break
        puzzle[row][col] = "*"  

        removed += 1

    return puzzle
