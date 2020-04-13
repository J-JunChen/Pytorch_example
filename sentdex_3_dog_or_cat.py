import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data.

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    TESTING = "PetImages/Testing"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):  # 返回指定路径下的文件和文件夹列表。
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1

                    except Exception as e:
                        # pass
                        print(label, f, str(e)) 
        np.random.shuffle(self.training_data)    # 洗牌                     
        np.save("./data/training_data.npy", self.training_data)
        print('Cats: ', self.catcount)
        print("Dogs: ", self.dogcount)

if REBUILD_DATA:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()         

training_data = np.load("./data/training_data.npy", allow_pickle=True)
print(len(training_data))

for data, target in training_data:
    break
x = torch.Tensor(data).view(-1, 50, 50)
y = torch.Tensor(target)

print(y)
plt.imshow(x[0], cmap="gray")
plt.show()