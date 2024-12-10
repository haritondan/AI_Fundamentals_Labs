from solver_heuristic import solve_sudoku_with_mrv_and_constraints
from solver import solve_sudoku_with_constraints
from ac3 import solve_sudoku_with_constraints_ac3
from generator import generate_full_sudoku_grid, create_unsolved_sudoku
from utils import print_grid, read_sudoku_from_file, is_solvable_sudoku

if __name__ == "__main__":

    print("Choose an option:")
    print("1. Solve a Sudoku puzzle using Backtracking")
    print("2. Solve a Sudoku puzzle using MRV heuristic")
    print("3. Generate a full solved Sudoku grid")
    print("4. Solving using Backtracking and AC-3")
    # Ask for user input
    choice = input("Enter choice (1-4): ")

    if choice == "1":
        print("Solving using Backtracking with Constraint Propagation")
        grid = read_sudoku_from_file("test.txt")
        if not is_solvable_sudoku(grid, solve_sudoku_with_constraints):
            print("This Sudoku puzzle is invalid or unsolvable.")
        elif solve_sudoku_with_constraints(grid):
            print("Solved Sudoku grid:")
            print_grid(grid)
        else:
            print("No solution exists")

    elif choice == "2":
        print("Solving using MRV heuristic")
        grid = read_sudoku_from_file("test.txt")
        if not is_solvable_sudoku(grid, solve_sudoku_with_mrv_and_constraints):
            print("This Sudoku puzzle is invalid or unsolvable.")
        elif solve_sudoku_with_mrv_and_constraints(grid):
            print("Solved Sudoku grid:")
            print_grid(grid)
        else:
            print("No solution exists")

    elif choice == "3":
        print("Generating an unsolved Sudoku grid:")
        full_grid = generate_full_sudoku_grid()
        print_grid(full_grid)
        clues = int(input("Enter the number of clues (e.g., 30 for medium difficulty): "))
        unsolved_puzzle = create_unsolved_sudoku(full_grid, clues)
        print("Generated Unsolved Sudoku Puzzle:")
        print_grid(unsolved_puzzle)

    elif choice == "4":
        print("Solving using Backtracking and AC-3")
        grid = read_sudoku_from_file("test.txt")
        if not is_solvable_sudoku(grid, solve_sudoku_with_constraints):
            print("This Sudoku puzzle is invalid or unsolvable.")
        elif solve_sudoku_with_constraints_ac3(grid):
            print("Solved Sudoku grid:")
            print_grid(grid)
        else:
            print("No solution exists")
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")




# Sample Input
# input 
# 53**7****
# 6**195***
# *98****6*
# 8***6***3
# 4**8*3**1
# 7***2***6
# *6****28*
# ***419**5
# ****8**79


#  Output
# 534678912
# 672195348
# 198342567
# 859761423
# 426853791
# 713924856
# 961537284
# 287419635
# 345286179