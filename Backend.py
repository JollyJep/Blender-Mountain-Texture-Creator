import numpy as np
from numba import jit, njit, prange
import random
import math
try:
    import cupy as cp
    gpu = True
except:
    gpu = False

class dla:
    def __init__(self, initial_grid_size, final_grid_size, max_height, grid_fill_small, grid_fill_overall, spawn_points, start_coordinates_local, start_coordinates_global, num_steps):
        self.initial_grid_size = initial_grid_size
        self.grid_size = final_grid_size
        self.max_height = max_height
        self.grid_fill_small = grid_fill_small
        self.grid_fill_overall = grid_fill_overall
        self.percentage_gradient = (grid_fill_small - grid_fill_overall) / num_steps
        self.iteration = 0
        self.multiplier = 1
        self.spawn_points = spawn_points
        self.pixel_count = 1
        self.num_steps = num_steps
        self.sharp_grid = np.zeros((initial_grid_size[0], initial_grid_size[1], 5))   #5 Values per pixel. Direction, ID, Noise x, Noise y, height
        self.sharp_grid[start_coordinates_local[0], start_coordinates_local[1], 0] = 5 # Connections of particles, 1 is North, 2 is East, 3 is South, 4 is West, 5 is seed point
        self.sharp_grid[start_coordinates_local[0], start_coordinates_local[1], 1] = 1    #Pixel ID number, 1 is the seed pixel. This is used for height mapping later
        self.resolution = (self.initial_grid_size[0] * self.multiplier, self.initial_grid_size[1] * self.multiplier, 5)
        self.soft_grid = None
        self.done = True
        self.main_control()

    def main_control(self):
        while not self.done:
            self.dla_control()
            self.multiplier *= 2
            self.iteration += 1
            self.next_resolution()

    def next_resolution(self):
        self.resolution = (self.initial_grid_size[0] * self.multiplier, self.initial_grid_size[1] * self.multiplier, 5)
        if self.resolution[0] <= self.grid_size[0] and self.resolution[1] <= self.grid_size[1]:
            new_grid = np.zeros(self.resolution)
            filled_points = np.argwhere(self.sharp_grid[:, :, 0] != 0)
            for coords in filled_points:
                direction = self.sharp_grid[coords[0], coords[1], 0]
                new_coords = coords * 2
                new_grid[new_coords[0], new_coords[1], 0] = direction
                if direction == 1:
                    new_grid[new_coords[0], new_coords[1] + 1, 0] = direction
                elif direction == 2:
                    new_grid[new_coords[0] + 1, new_coords[1] + 1, 0] = direction
                elif direction == 3:
                    new_grid[new_coords[0], new_coords[1] - 1, 0] = direction
                elif direction == 4:
                    new_grid[new_coords[0] - 1, new_coords[1] + 1, 0] = direction
            if gpu:
                x_new, y_new = cp.asarray(np.meshgrid(np.arange(self.resolution[0]), np.arange(self.resolution[1]), indexing='ij'))
                self.soft_grid = self.soft_upscaler_gpu(cp.asarray(self.sharp_grid), x_new, y_new)
            else:
                self.soft_grid =self.soft_upscaler_cpu(self.resolution, self.sharp_grid)
            self.sharp_grid = new_grid

        else:
            self.done = True

    def dla_control(self):
        percentage = self.percentage_gradient * self.iteration + self.grid_fill_small
        max_pixels = int(self.resolution[0] * self.resolution[1] * percentage)
        false_grid = np.full((self.resolution[0] + 1, self.resolution[1] + 1), False)
        false_grid[0, :] = True
        false_grid[:, 0] = True
        false_grid[-1, :] = True
        false_grid[:, -1] = True
        pixel_count = self.pixel_count
        if max_pixels > self.pixel_count:
            for _ in range(max_pixels-self.pixel_count):
                self.sharp_grid = self.run_dla_on_resolution(self.sharp_grid, false_grid, pixel_count + 1)
                pixel_count += 1
            self.pixel_count = pixel_count

    @staticmethod
    @njit
    def run_dla_on_resolution(grid, false_grid, pixel_ID, directions=np.array([[0, 1, 1], [1, 0, 2], [0, -1, 3], [-1, 0, 4]])):
        indices_pixels = np.argwhere(grid[:, :, 0] == 0)
        position = random.choice(indices_pixels)
        connected = False
        directions_limited = directions.copy()
        directions_movement = directions_limited.copy()
        while not connected:
            for direction in directions_limited:
                if not false_grid[position[0] + direction[0], position[1] + direction[1]]:
                    if grid[position[0] + direction[0], position[1] + direction[1], 0] != 0:
                        grid[position[0], position[1], 0] = direction[2]
                        grid[position[0], position[1], 1] = pixel_ID
                        return grid
                else:
                    directions_movement = np.delete(directions_movement, np.where(directions_movement == direction))
            if len(directions_movement) == 0:
                indices_movement = np.argwhere(false_grid == False)
                indices = np.intersect1d(indices_pixels, indices_movement)
                position = random.choice(indices)
                directions_limited = directions.copy()
                directions_movement = directions_limited.copy()
            else:
                movement = random.choice(directions_movement)
                position += movement[:2]
                index_to_delete = np.where(np.all(directions_limited == movement, axis=1))[0]
                directions_limited = np.delete(directions_limited, index_to_delete, axis=0)    # Just the head of the motion
                directions_movement = directions_limited.copy()


    def sharp_height_calculator(self):
        IDs = np.arange(self.pixel_count, 0, -1)
        for ID in IDs:
            pixel = np.argwhere(self.sharp_grid[:, :, 1] == ID)[0]
            if self.sharp_grid[pixel[0], pixel[1], 4] == 0:
                chain_continue = True
                coords = pixel
                height = 1
                while chain_continue:
                    if self.sharp_grid[coords[0], coords[1], 4] < height:
                        self.sharp_grid[coords[0], coords[1], 4] = height
                        if self.sharp_grid[coords[0], coords[1], 1] == 1:
                            coords += np.array([0, 1])
                        elif self.sharp_grid[coords[0], coords[1], 1] == 2:
                            coords += np.array([1, 0])
                        elif self.sharp_grid[coords[0], coords[1], 1] == 3:
                            coords += np.array([0, -1])
                        elif self.sharp_grid[coords[0], coords[1], 1] == 4:
                            coords += np.array([-1, 0])
                        else:
                            chain_continue = False
                    else:
                        chain_continue = False


    @staticmethod
    def soft_upscaler_gpu(sharp_grid, x_new, y_new):
        x_old = x_new / 2
        y_old = y_new / 2

        # Calculate the floor and ceil for each coordinate
        x_floor = cp.floor(x_old).astype(int)
        y_floor = cp.floor(y_old).astype(int)
        x_ceil = cp.ceil(x_old).astype(int)
        y_ceil = cp.ceil(y_old).astype(int)

        # Ensure that we don't go out of bounds
        x_ceil = cp.clip(x_ceil, 0, sharp_grid.shape[0] - 1)
        y_ceil = cp.clip(y_ceil, 0, sharp_grid.shape[1] - 1)

        # Calculate weights for interpolation
        x_weight = x_old - x_floor
        y_weight = y_old - y_floor

        # Perform bilinear interpolation
        soft_grid = (sharp_grid[x_floor, y_floor] * (1 - x_weight) * (1 - y_weight) +
                     sharp_grid[x_ceil, y_floor] * x_weight * (1 - y_weight) +
                     sharp_grid[x_ceil, y_ceil] * x_weight * y_weight +
                     sharp_grid[x_floor, y_ceil] * (1 - x_weight) * y_weight)
        return cp.asnumpy(soft_grid)

    @staticmethod
    @njit(parallel=True)
    def soft_upscaler_cpu(resolution, sharp_grid):
        x_new, y_new = np.meshgrid(np.arange(resolution[0]), np.arange(resolution[1]), indexing='ij')
        x_old = x_new / 2
        y_old = y_new / 2

        # Calculate the floor and ceil for each coordinate
        x_floor = np.floor(x_old).astype(int)
        y_floor = np.floor(y_old).astype(int)
        x_ceil = np.ceil(x_old).astype(int)
        y_ceil = np.ceil(y_old).astype(int)

        # Ensure that we don't go out of bounds
        x_ceil = np.clip(x_ceil, 0, sharp_grid.shape[0] - 1)
        y_ceil = np.clip(y_ceil, 0, sharp_grid.shape[1] - 1)

        # Calculate weights for interpolation
        x_weight = x_old - x_floor
        y_weight = y_old - y_floor

        # Perform bilinear interpolation
        soft_grid = (sharp_grid[x_floor, y_floor] * (1 - x_weight) * (1 - y_weight) +
                     sharp_grid[x_ceil, y_floor] * x_weight * (1 - y_weight) +
                     sharp_grid[x_ceil, y_ceil] * x_weight * y_weight +
                     sharp_grid[x_floor, y_ceil] * (1 - x_weight) * y_weight)
        return soft_grid







