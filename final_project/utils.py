
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def preprocess_features(npdata, pca=128):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """

    pca = PCA(n_components=pca, whiten=True)
    npdata=pca.fit_transform(npdata)
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata
    # n , ndim = npdata.shape
    # npdata =  npdata.astype('float32')
    #
    # # Apply PCA-whitening with Faiss
    # mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    #
    # mat.train(npdata)
    #
    # assert mat.is_trained
    #
    # npdata = mat.apply_py(npdata)
    #
    # # L2 normalization
    # row_sums = np.linalg.norm(npdata, axis=1)
    # npdata = npdata / row_sums[:, np.newaxis]



class ReassignedDataset(Dataset):

    def __init__(self,dataset,pseudo_labels):
        self.images=self.make_dataset(dataset,pseudo_labels)

    def make_dataset(self,dataset,pseudolabels):
        #label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}

        choose_index=self.random_choose(pseudolabels)  #选的一部分才丢进去，其他不要

        images = []
        for idx in choose_index:
            data = dataset[idx][0]
            pseudolabel = pseudolabels[idx]
            images.append((data, pseudolabel))

        return images

    def check_pseudolabel_distribution(self):

        label_array=[]
        for i in range(len(self)):
            _,label=self.images[i]
            label_array.append(label)
        print(pd.Series(label_array).value_counts())




    def random_choose(self,labels):
        # 随机选择，使得每个类别的数量大致相同，避免垃圾解。这个labels是labels array，如[0,2,7,4,7,6,10]

        nmb_of_none_zero = len(set(labels))
        label_num = max(labels + 1) #最大类别的序号


        size_per_label = int(len(labels) / nmb_of_none_zero) + 1  #每个label下应该有几个。

        res = np.array([])
        for i in range(label_num):
            # skip empty clusters
            if np.sum(labels == i) == 0:
                continue

            indexes = np.random.choice(
                np.where(labels == i)[0],  #标签为i的那些点的index;
                size_per_label,
                replace= (np.sum(labels==i) <= size_per_label )
            )

            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= len(labels):
           return res[:len(labels)]
        res += res[: (len(labels) - len(res))]
        return res



    def __getitem__(self, index):

        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel
        :param index:
        :return:
        """
        return self.images[index]


    def __len__(self):
        return len(self.images)



