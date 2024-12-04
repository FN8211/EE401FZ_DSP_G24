import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_atom(image, show_result):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = image_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(pixels)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    def compute_gray_level(centroid):
        return 0.333 * centroid[0] + 0.333 * centroid[1] + 0.333 * centroid[2]

    gray_levels = [compute_gray_level(c) for c in centroids]
    sky_label = np.argmax(gray_levels)
    sky_region = (labels == sky_label).reshape(image_rgb.shape[:2])
    sky_region_connected_to_top = np.zeros_like(sky_region)

    for row in range(image_rgb.shape[0]):
        if row == 0:
            sky_region_connected_to_top[row] = sky_region[row]
        else:
            sky_region_connected_to_top[row] = sky_region[row] & sky_region_connected_to_top[row - 1]

    sky_region = sky_region_connected_to_top

    sky_pixels = image_rgb[sky_region]
    sky_gray_values = np.array([compute_gray_level(pixel) for pixel in sky_pixels])
    average_gray_value = np.mean(sky_gray_values)
    average_atom = average_gray_value / 255.0

    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    sky_pixels_count = np.sum(sky_region)
    sky_percentage = (sky_pixels_count / total_pixels) * 100

    sky_image = np.zeros_like(image_rgb)
    sky_image[sky_region] = image_rgb[sky_region]

    if show_result:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(labels.reshape(image_rgb.shape[:2]), cmap='jet')
        plt.title('KMeans Cluster Labels')

        plt.subplot(1, 3, 3)
        plt.imshow(sky_image)
        plt.title('Sky Region')
        cv2.imwrite('img/haze/results/sky_region.png', sky_image)

        plt.tight_layout()
        plt.show()

    return average_atom, sky_percentage

if __name__ == '__main__':
    image = cv2.imread('img/haze/1baseline/8.png')
    average_A, sky = kmeans_atom(image, True)
    print(average_A, sky)
