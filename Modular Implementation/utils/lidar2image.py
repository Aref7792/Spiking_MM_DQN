import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt




def draw_centered_vehicle(grid, length_cells, width_cells, heading_rad, value=1):
    """
    Draw a rotated rectangle in the center of the grid.
    """
    H, W = grid.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0  # center cell coords

    j = np.arange(W)[None, :]   # column indices
    i = np.arange(H)[:, None]   # row indices
    x = j - cx
    y = i - cy

    # Rotation
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    x_loc =  c * x + s * y
    y_loc = -s * x + c * y

    # Half-size
    hx = length_cells / 2.0
    hy = width_cells  / 2.0

    mask = (np.abs(x_loc) <= hx) & (np.abs(y_loc) <= hy)
    grid[mask] = value
    return grid
class LidarToOccupancyGrid:
    def __init__(self, output_range, v_max, voxel_size=0.5, x_range=(-60, 60), y_range=(-60, 60), max_distance=60):
        self.voxel_size = voxel_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.max_distance = max_distance
        self.output_range = output_range
        self.v_max = v_max

        self.grid_width = int((self.x_max - self.x_min) / self.voxel_size)
        self.grid_height = int((self.y_max - self.y_min) / self.voxel_size)
        self.occupancy_grid = np.zeros((self.grid_height + 1, self.grid_width + 1), dtype=int)
        print(self.occupancy_grid.shape)

    def process(self, obs, angles, v_vel, heading):
        points = []

        for i, ray in enumerate(obs):
            if ray[0] > 0 and ray[0] != 1:  # presence
                num = int((1 - abs(ray[0])) * 500)
                distance = np.linspace(ray[0], 1, num) * self.max_distance

                x = distance * np.cos(angles[i])
                y = distance * np.sin(angles[i])
                decay = 0.98 ** np.arange(num)
                z = decay * (ray[1]+v_vel)
                #z = decay * distance
                points.append([x, y, z])

        self.occupancy_grid.fill(0)

        # Add ego vehicle's rectangle
        vl = int(2 // self.voxel_size)
        vh = int(5 // self.voxel_size)
        cx, cy = self.grid_width // 2, self.grid_height // 2

        self.occupancy_grid = draw_centered_vehicle(self.occupancy_grid, vl, vh, np.pi - heading, v_vel)


        # Mark LiDAR points
        for pt in points:
            for j in range(len(pt[0])):
                l = int(pt[0][j] // self.voxel_size)
                k = int(pt[1][j] // self.voxel_size)

                l = self.grid_width // 2 + l
                k = self.grid_height // 2 - k

                if 0 <= k < self.grid_height and 0 <= l < self.grid_width:
                    self.occupancy_grid[l, k] =  pt[2][j]

        #occ = self.occupancy_grid[79:163, 99:141]/30
        occ = self.occupancy_grid[self.output_range[0]:self.output_range[1],self.output_range[0]:self.output_range[1]]/ self.v_max
        cmap = plt.get_cmap("viridis")
        rgbocc = np.uint8(cmap(occ)[:,:,:3]*255)



        rgbocc = np.transpose(rgbocc,(2,0,1))



        #occe = np.expand_dims(occ, 0)

        return rgbocc