
import numpy as np
import cv2
import sys
from scipy.signal import convolve2d


SOL = cv2.imread(r'C:\\Users\\rinea\\OneDrive\\Desktop\\euclid_construction.jpg')

# reducing image size by 50%
cv2.resize(SOL, (0, 0), fx=0.5, fy=0.5)

grayscale = cv2.cvtColor(SOL, cv2.COLOR_BGR2GRAY)

threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def gameOfLife(threshold):

    matrix = threshold // 255
    alive_count = np.ones((3,3))

    neighbors_count = convolve2d(matrix, alive_count, mode='same', boundary='wrap').astype(np.int64)

    birth = (neighbors_count == 3) & (matrix == 0)
    survival = ((neighbors_count == 2) | (neighbors_count == 3)) & (matrix == 1)
    death = (neighbors_count == 4) & (matrix == 1)

    updated_matrix = np.zeros_like(matrix, dtype=np.uint8)
    updated_matrix[birth | survival | death] = 1

    return updated_matrix


grid_size = 1
# kronecker operation
grid = np.kron(threshold, np.ones((grid_size, grid_size)))

updated_matrix = gameOfLife(grid)
# * 3 for BGR and merging all 3 channels
updated_image = cv2.merge([updated_matrix * 255] * 3)

cv2.imwrite('updated_matrix.png', updated_image)
cv2.waitKey()

np.set_printoptions(threshold=sys.maxsize)
print(updated_matrix)

gaussian_blur = cv2.GaussianBlur(updated_image,(3,3),0)
cv2.imshow('gaussian filter', gaussian_blur)
cv2.waitKey()

b_gaus,g_gaus,r_gaus = cv2.split(gaussian_blur)

b,g,r = cv2.split(updated_image)

b = np.clip(b_gaus.astype(int) + b.astype(int),0, 255).astype(np.uint8)

g = np.clip(g_gaus.astype(int) + g.astype(int),0,255).astype(np.uint8)

r = np.clip(r_gaus.astype(int) + r.astype(int),0, 255).astype(np.uint8)

b [b > 255] = 255
g [g > 255] = 255
r [r > 255] = 255

g = b // 6


merged = cv2.merge([b, g, r])


updated_last = merged


cv2.imwrite('final_SOL.png', updated_last)

glow_strength = 5
glow_radius = 5

blur = cv2.GaussianBlur(updated_last, (glow_radius, glow_radius), 1)
blend = cv2.addWeighted(updated_last, 1, blur, glow_strength, 1)

cv2.imwrite("glow.png", blend)