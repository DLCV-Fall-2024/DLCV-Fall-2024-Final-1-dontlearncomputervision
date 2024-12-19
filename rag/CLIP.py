import clip
import json
import faiss
import torch
from PIL import Image
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

all_embeddings = []

for i in range(5):
    print("round:", i)
    image = preprocess(Image.open(f"./input/image{i}.jpg")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        # print(type(image_features))  # <class 'torch.Tensor'>
        # print(image_features.shape)  # torch.Size([1, 512])

    image_features_2list = image_features.tolist()  # list
    all_embeddings.append(image_features_2list)

    image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

    if i == 0:
        embedding_means = image_features_2np.reshape(1, -1)
    else:
        embedding_mean = image_features_2np.reshape(1, -1)
        embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray


# ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# Define the output JSON file path
output_json_path = "./embeddings_CLIP.json"

# Save embeddings to a JSON file
with open(output_json_path, "w") as json_file:
    json.dump(all_embeddings, json_file)

print(f"Embeddings saved to {output_json_path}")


# ------------------------------------------------------------------------ FAISS ------------------------------------------------------------------------
# Set dimensions and number of data points
dimension = 512
database_size = 5

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)        # L2 distance
# index = faiss.IndexFlatIP(dimension)      # inner product
# index = faiss.IndexBinaryFlat(dimension)  # Hamming distance
print("Is trained:", index.is_trained)

# Add vectors to the index
index.add(embedding_means)
print("Number of vectors in index:", index.ntotal)

# Query vector
query_vector = embedding_means[1].reshape(1, -1)

# Search for the 5 closest vectors
search_closest_num = 2
distances, indices = index.search(query_vector, search_closest_num)

# Print the distances to the nearest neighbors
print("Distances:", distances)
# Print the indices of the nearest neighbors
print("Indices:", indices)
