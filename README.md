# CLIP_Image_Retrieval
Image/Instance Retrieval using CLIP, A self supervised Learning Model

![architecture](assets/Clip_architecture.png)

### Image Similarity

![Retrieval](assets/result.jpg)

## Check out My DinoV2 implementation for Image search [here](https://github.com/purnasai/Dino_V2)

This Repository contained:
- Image Retrival with `Image` as a querry
- Image Retrival with `Text` as a querry
- Code to self organize Images into Directories.

### Ideas:
- we can modify it further to search an object in image, by splitting image into multiple patches, searching for object in image patch, get similiarity of text & patch. patch with highest similarity has object in it.
- We can use it to organize unorganized files into folders.
- Adding other languages embeddings to search with other languages, like [here](https://github.com/clip-italian/clip-italian/)

#### Notes:
- Paralell Processing is not required at Faiss Search Time, since Faiss Already implements it.
- Paralell Processing at Feature creation for Database images is helpful.
- We are currently using Faiss.IndexL2 with Normalized Vectors which is Cosine Similarity, But IVFPQ(Inverted File pointer Quantization) + HNSW Of FAISS can Search Billions Of Points in MilliSeconds & Can be added Later.
- save & load model from locally to quick run.

### Metrics can be used:
- Recall@K
- Precision@K
- F1-Score@K
- Mean Average precision
- Mean Average Recall
