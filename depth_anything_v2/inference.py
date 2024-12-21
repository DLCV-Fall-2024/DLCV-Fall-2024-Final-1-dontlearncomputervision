import os
import cv2
import glob
import clip
import json
import faiss
import torch
import argparse
import matplotlib
import numpy as np
from PIL import Image
from datasets import load_dataset

from depth_anything_v2.dpt import DepthAnythingV2


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def generate_depth_map_embeddings(input_dir):
    """
    input: testing data download form hugging face (local)
    output: a list of depth map embeddings
    """

    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    depth_anything = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
    depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu')) # TODO: change .pth file's path
    depth_anything = depth_anything.to(DEVICE).eval()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # load dataset from the given path
    # dataset = load_dataset("../data/dlcv_2024_final1", split='test')
    dataset = load_dataset(input_dir, split='train')

    for i, data in enumerate(dataset):

        # ---------------------------------------------------- generate depth map --------------------------------------------------------------
        print(f"Generate {i}th testing image's depth map. :)")

        raw_image = data['image']
        # print(type(raw_image)) # PIL.PngImagePlugin.PngImageFile

        # convert input type to numpy.ndarray
        raw_image = np.array(raw_image)

        depth = depth_anything.infer_image(raw_image, 518)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        # print("depth map shape:", depth.shape) # (720, 1355)
        """
        Each entry of depth is a number in [0, 255], which represents the depth information of the corresponding pixel.
        The larger the value, the closer it is to the camera.
        """

        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        # print(type(depth))  # numpy.ndarray
        # print("depth map shape:", depth.shape) # (720, 1355, 3) 
        """
        Each entry of depth is an array of size 3, which represents the color value (BGR) of the corresponding pixel in the output picture.
        """

        cv2.imwrite(os.path.join("./output", f'depth_image{i}.png'), depth)

        # # convert type to PIL
        # depth = depth.astype(np.uint8)
        # pil_image = Image.fromarray(depth)
        # # print(type(pil_image))  # PIL.Image.Image
        # # print(pil_image.size)   # (1355, 720)


        # ---------------------------------------------------- use CLIP to generate embedding of depth map --------------------------------------------------------------
        # image = preprocess(pil_image).unsqueeze(0).to(DEVICE)  # TEST: the input of preprocess should be PIL.PngImagePlugin.PngImageFile, don't know if PIL.Image.Imageis available or not
        # print(type(image))  # torch.Tensor
        # print(image.shape)  # torch.Size([1, 3, 224, 224])

        image = preprocess(Image.open(f"../rag/output/train/depth_image{i}.png")).unsqueeze(0).to(DEVICE)

    
        with torch.no_grad():
            image_features = model.encode_image(image)
            # print(type(image_features))  # <class 'torch.Tensor'>
            # print(image_features.shape)  # torch.Size([1, 512])

        image_features_2np = image_features.cpu().numpy()  # numpy.ndarray
        print(image_features_2np)

        if i == 0:
            embedding_list = image_features_2np.reshape(1, -1)
        else:
            embedding = image_features_2np.reshape(1, -1)
            embedding_list = np.append(embedding_list, embedding, axis=0)  # numpy.ndarray

    return embedding_list

# # Query vector:
# # (i is the index of data in the test dataset)
# query_vector = embedding_list[i].reshape(1, -1)


e_l = generate_depth_map_embeddings("../data/dlcv_2024_final1")
# print(e_l)




