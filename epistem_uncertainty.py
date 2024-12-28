import numpy as np
import torch 
from PIL import Image
import os
import pickle

def npy2png(array, png_path):
    if not os.path.exists(png_path):
        os.makedirs(png_path)

    for i in range(array.shape[0]):
        img = array[i].transpose(1, 2, 0)  # Change shape from (3, 256, 256) to (256, 256, 3)
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(png_path, f'image_{i}.png'))


def average_cosine_similarity(vectors):
    """
    Computes the average cosine similarity between all pairs of vectors in the input array.

    Parameters:
    - vectors (numpy.ndarray): A 2D array of shape (n, d), where n is the number of vectors
      and d is the dimensionality of each vector.

    Returns:
    - float: The average cosine similarity between all pairs of vectors.
    """
    # Normalize the vectors to unit length (L2 norm = 1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    # Compute cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

    # Extract the upper triangle of the cosine similarity matrix (excluding diagonal)
    triu_indices = np.triu_indices_from(cosine_similarity_matrix, k=1)
    pairwise_similarities = cosine_similarity_matrix[triu_indices]

    # Compute the average cosine similarity
    return pairwise_similarities.mean()




M = 4
PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100"

##### 1) numpy to png #####

# with open('logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100.pkl', 'rb') as f:
#     data = pickle.load(f)
# data = data.cpu().numpy()
# for i in range(M):
#     npy2png(data[:, i], f'{PATH_ROOT}/{i}')

# save only one ensemble member's images
m = 0
with open('logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100.pkl', 'rb') as f:
    data = pickle.load(f)
data = data.cpu().numpy()
data = data[:, m]

# save data as numpy array
np.save(f"{PATH_ROOT}/{m}/all_imgs.npy", data)


##### 2) extract inception-net (FID) features #####
## run from command line:
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/0 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/1 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/2 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/3 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000


##### 3) average cosine similarity #####

# fid_features = []
# for model_id in range(M):
#     path = f"{PATH_ROOT}/{model_id}/fid_features.pt"
#     fid_features.append(torch.load(path))

# fid_features = torch.stack(fid_features, dim=0)
# fid_features = np.transpose(fid_features.cpu().numpy(), (1, 0, 2))   
# print(fid_features.shape)

# cos_sim = np.array([average_cosine_similarity(fid_features[i]) for i in range(fid_features.shape[0])])
# print(cos_sim.shape)
# print(cos_sim.mean(), cos_sim.std(), cos_sim.min(), cos_sim.max())
# np.save(f"{PATH_ROOT}/cos_sim.npy", cos_sim)


