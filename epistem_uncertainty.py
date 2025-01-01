import numpy as np
import torch 
from PIL import Image
import os
import pickle
import clip

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

N = 5
# N = 50
# N = 1000

# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100"
# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_b100"
# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1"
# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100_991"
# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100_854"
# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1_854"
# PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1_991"
PATH_ROOT = "/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1_22"

# # #### 1) numpy to png #####

with open(f'{PATH_ROOT}.pkl', 'rb') as f:
    data = pickle.load(f)
data = data.cpu().numpy()
for i in range(M):
    npy2png(data[:, i], f'{PATH_ROOT}/{i}')

# # save only one ensemble member's images
m = 0
with open(f'{PATH_ROOT}.pkl', 'rb') as f:
    data = pickle.load(f)
data = data.cpu().numpy()
data = data[:, m]

np.save(f"{PATH_ROOT}/{m}/all_imgs.npy", data)


##### 2.1) extract inception-net (FID) features #####
## run from command line (from edm directior and using edm conda environment):
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/0 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/1 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/2 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100/3 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000

# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1/0 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1/1 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1/2 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_1/3 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 1000

# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100_991/0 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 50
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100_991/1 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 50
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100_991/2 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 50
# python fid.py calc --images=/ivi/zfs/s0/original_homes/mjazbec/epistem-diff/DECU/logs/bootstrapped_imagenet_5/certain_vs_uncertain/all_samples_100_991/3 --ref=/nvmestore/mjazbec/diffusion/edm/fid-refs/cifar10-32x32.npz --num 50


# # ####### 2.2) CLIP features #######

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda"
# model, preprocess = clip.load("ViT-B/32", device=device)


# for m in range(M):
#     clip_vecs = []
#     for i in range(N):
#         image = preprocess(Image.open(f"{PATH_ROOT}/{m}/image_{i}.png")).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             clip_vecs.append(model.encode_image(image))

#     clip_vecs = torch.concat(clip_vecs, dim=0)
#     print(clip_vecs.shape)
#     torch.save(clip_vecs, f"{PATH_ROOT}/{m}/clip_features.pt")



# # # ##### 3) average cosine similarity #####

# # FEATURES_TYPE = "fid"
# FEATURES_TYPE = "clip"

# features = []
# for model_id in range(M):
#     path = f"{PATH_ROOT}/{model_id}/{FEATURES_TYPE}_features.pt"
#     features.append(torch.load(path))

# features = torch.stack(features, dim=0)
# features = np.transpose(features.cpu().numpy(), (1, 0, 2))   
# print(features.shape)

# cos_sim = np.array([average_cosine_similarity(features[i]) for i in range(features.shape[0])])
# print(cos_sim.shape)
# print(cos_sim.mean(), cos_sim.std(), cos_sim.min(), cos_sim.max())
# np.save(f"{PATH_ROOT}/cos_sim_{FEATURES_TYPE}.npy", cos_sim)



##### 4) misc

# with open(f'{PATH_ROOT}.pkl', 'rb') as f:
#     data = pickle.load(f)
# data = data.cpu().numpy()

# # save as .pkl
# with open(f"{PATH_ROOT}.pkl", 'wb') as f:
#     pickle.dump(data, f)

# with open(f'{PATH_ROOT}.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data.shape)
# print(type(data))



