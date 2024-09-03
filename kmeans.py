import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(pixels, k):
    # Randomly choose k pixels as initial centroids
    random_indices = np.random.choice(pixels.shape[0], size=k, replace=False)
    centroids = pixels[random_indices]
    return centroids

def assign_clusters(pixels, centroids):
    # Calculate the distance of each pixel from each centroid and assign to the nearest centroid
    distances = np.sqrt(((pixels - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(pixels, clusters, k):
    # Update the centroids by calculating the mean of all pixels assigned to each centroid
    new_centroids = np.array([pixels[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(pixels, k, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(pixels, k)
    
    for _ in range(max_iters):
        clusters = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, clusters, k)
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# Load the image in grayscale
image = cv2.imread('three-color.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale Image')
plt.show()

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 1)

# Define the number of clusters (e.g., 3 clusters)
n_clusters = 3

# Run K-means clustering
clusters, centroids = kmeans(pixels, n_clusters)

# Replace each pixel value with its corresponding centroid value
clustered_pixels = centroids[clusters].reshape(image.shape)

# Display the clustered image
plt.imshow(clustered_pixels, cmap='gray')
plt.title('Clustered Image')
plt.show()
