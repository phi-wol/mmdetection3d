
#%%
import mmcv

# %%
book_path = "objectron_processed_book_all/annotations/objectron_train.json"
chair_path = "objectron_processed_chair_all/annotations/objectron_train.json"
data_book = mmcv.load(book_path)
data_chair = mmcv.load(chair_path)
print ("Loaded data! ...")

print('Len book: ', len(data_book['annotations']))
print('Len chair: ', len(data_chair['annotations']))

#%%
# wrong_samples = {}
# for sample in data['annotations']:
#     if abs(sample["bbox_cam3d"][2])>5:
#     #if sample["bbox_cam3d"][2]<0:
#         print(sample["bbox_cam3d"])
#         print(sample['file_name'])
#         wrong_samples[sample['file_name'][:-7]] = 1

# print("Filtered wrong samples: ")
# print(wrong_samples)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def print_stats(data, object_class):
    distances = []
    for sample in data['annotations']:
        distances.append(sample['bbox_cam3d'][2])

    array = np.array(distances)
    mean = array.mean()
    std = array.std()
    print("Mean: ", mean)
    print("Std: ", std)

    return distances

print("Dataset: book")
dist_book = print_stats(data_book, "book")

print("Dataset: chair")
dist_chair = print_stats(data_chair, "chair")

max_range = max([len(dist_book), len(dist_chair)])
distances = pd.DataFrame(data={"dist_book": dist_book[:max_range], "dist_chair": dist_chair[:max_range]})

#sns.set(rc={'figure.figsize':(10,10)})

sns_plot = sns.histplot(data=distances) #, binrange=(-0.5, 4))
sns_plot.set(xlabel='distance to object [m]', ylabel='object count', xlim=(-0.5, 3))

sns_plot.figure.savefig("dataset_statistics/distance_distribution.png")
sns_plot.figure.savefig("dataset_statistics/distance_distribution.pdf")





