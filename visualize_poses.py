import numpy as np
import matplotlib.pyplot as plt
import sys

pose_file = sys.argv[1]
poses = np.load(pose_file)
x, y, z = poses[:30, 0, -1], poses[:30, 1, -1], poses[:30, 2, -1]

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("poses")
plt.show()