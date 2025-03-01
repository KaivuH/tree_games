class CrosswordEnv:
    def __init__(self, grid, across_clues, down_clues, solution):
        """
        grid: A tuple of tuples representing the current crossword puzzle state (with None for empty cells and '#' for black cells).
        across_clues: A string with the across clues.
        down_clues: A string with the down clues.
        solution: A tuple of tuples representing the solved grid.
        """
        self.initial_grid = grid
        self.grid = grid
        self.solution = solution
        self.across_clues = across_clues
        self.down_clues = down_clues
        self.history = []

    def is_solved(self):
        # if it's fillable (not a black cell), verify it matches the solution.
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] != "#" and self.grid[r][c] != self.solution[r][c]:
                    return False
        return True

    def take_action(self, move: dict) -> bool:
        row = move["row"]
        col = move["col"]
        direction = move["direction"].lower()
        word = move["word"].upper()
        grid_list = [list(r) for r in self.grid]

        if direction == "across":
            if col + len(word) > len(grid_list[0]):
                return False
            for i, letter in enumerate(word):
                if grid_list[row][col + i] == "#":
                    return False
                grid_list[row][col + i] = letter
        elif direction == "down":
            if row + len(word) > len(grid_list):
                return False
            for i, letter in enumerate(word):
                if grid_list[row + i][col] == "#":
                    return False
                grid_list[row + i][col] = letter
        else:
            return False

        self.history.append(self.grid)
        self.grid = tuple(tuple(r) for r in grid_list)
        return True

    def copy(self):
        new_env = CrosswordEnv(self.grid, self.across_clues, self.down_clues, self.solution)
        new_env.history = self.history.copy()
        return new_env

    def get_result(self):
        return str(self)

    def __str__(self):
        return "\n".join(" ".join(cell if cell is not None else "_" for cell in row) for row in self.grid)
