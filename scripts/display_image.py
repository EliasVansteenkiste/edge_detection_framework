import os
import matplotlib

import matplotlib.pyplot as plt
from PIL import Image

image_name = "train_30154.jpg"
image_dir = "/home/frederic/Downloads/plnt/"

fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
a = a.set_title(image_name)
im =Image.open(os.path.join(image_dir,image_name))
plt.imshow(im)

plt.show()