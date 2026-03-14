import os
import matplotlib.pyplot as plt

cancer_images = len(os.listdir("data/cancerous"))
normal_images = len(os.listdir("data/non-cancerous"))

labels = ["Cancerous", "Normal"]
values = [cancer_images, normal_images]

plt.bar(labels, values)

plt.title("Dataset Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.show()
