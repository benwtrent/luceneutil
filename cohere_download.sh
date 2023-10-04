#!/bin/sh

# simple
curl -LO https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/resolve/main/data/train-00000-of-00004-1a1932c9ca1c7152.parquet
curl -LO https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/resolve/main/data/train-00001-of-00004-f4a4f5540ade14b4.parquet
curl -LO https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/resolve/main/data/train-00002-of-00004-ff770df3ab420d14.parquet
curl -LO https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/resolve/main/data/train-00003-of-00004-85b3dbbc960e92ec.parquet

# japanese
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-ja-embeddings/parquet/Cohere--wikipedia-22-12-ja-embeddings/train/0.parquet -o 0-ja.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-ja-embeddings/parquet/Cohere--wikipedia-22-12-ja-embeddings/train/1.parquet -o 1-ja.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-ja-embeddings/parquet/Cohere--wikipedia-22-12-ja-embeddings/train/33.parquet -o 2-ja.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-ja-embeddings/parquet/Cohere--wikipedia-22-12-ja-embeddings/train/34.parquet -o 3-ja.parquet
#
##english
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/Cohere--wikipedia-22-12-en-embeddings/train/0.parquet -o 0-en.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/Cohere--wikipedia-22-12-en-embeddings/train/1.parquet -o 1-en.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/Cohere--wikipedia-22-12-en-embeddings/train/251.parquet -o 2-en.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/Cohere--wikipedia-22-12-en-embeddings/train/252.parquet -o 3-en.parquet
#
## german
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-de-embeddings/parquet/Cohere--wikipedia-22-12-de-embeddings/train/0.parquet -o 0-de.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-de-embeddings/parquet/Cohere--wikipedia-22-12-de-embeddings/train/1.parquet -o 1-de.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-de-embeddings/parquet/Cohere--wikipedia-22-12-de-embeddings/train/106.parquet -o 2-de.parquet
#curl -L https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-de-embeddings/parquet/Cohere--wikipedia-22-12-de-embeddings/train/107.parquet -o 3-de.parquet

