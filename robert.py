import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Membaca gambar (dengan mode 'L' untuk grayscale)
image = imageio.imread('Screenshot (2).png', mode='L')

# Operator Roberts (dengan kernel horizontal dan vertikal)
roberts_kernel_x = np.array([[1, 0],
                              [0, -1]])

roberts_kernel_y = np.array([[0, 1],
                              [-1, 0]])

# Menyaring gambar dengan kedua kernel
edges_x = convolve(image, roberts_kernel_x)
edges_y = convolve(image, roberts_kernel_y)

# Menghitung magnitude dari gradien (magnitude = sqrt(dx^2 + dy^2))
edges = np.sqrt(edges_x**2 + edges_y**2)

# Menampilkan gambar
plt.figure(figsize=(10, 5))

# Gambar asli
plt.subplot(1, 2, 1)
plt.title("Gambar Asli")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Gambar hasil deteksi tepi
plt.subplot(1, 2, 2)
plt.title("Deteksi Tepi dengan Operator Roberts")
plt.imshow(edges, cmap='gray')
plt.axis('off')

# Menampilkan hasil
plt.show()