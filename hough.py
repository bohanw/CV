import numpy as np
import math

def hough_transform(points, theta_resolution=1, rho_resolution=1):
    # Define theta values from 0 to 180 degrees
    theta_values = list(range(90))
    max_rho = 20
    
    # Create the accumulator array
    accumulator = np.zeros((2 * max_rho // rho_resolution, 180), dtype=int)
    
    # Initialize variables to track the best line parameters
    best_rho = 0
    best_theta = 0
    max_votes = 0
    
    # Iterate through each point
    for x, y in points:
        for theta_index, theta in enumerate(theta_values):
            rho = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))) + max_rho

            accumulator[rho // rho_resolution, theta] += 1

    for i in range(2 * max_rho):
        for j in range(len(theta_values)):
            if accumulator[i, j] > 0 and accumulator[i, j ]> max_votes:
                max_votes = accumulator[i, j]
                best_rho = i
                best_theta = j               
    # if accumulator[rho // rho_resolution, theta_index] > max_votes:
    #     max_votes = accumulator[rho // rho_resolution, theta_index]
    #     best_rho = rho
    #     best_theta = theta
    
    print(accumulator)
    return best_rho , best_theta

# Example usage
points = [(2, 0), (0,2)]
best_rho, best_theta = hough_transform(points)

print(f"Best-fitting line parameters: ρ = {best_rho}, θ = {best_theta} degrees")
