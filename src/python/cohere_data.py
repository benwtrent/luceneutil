import numpy as np


DATA_SETS =[
    {"name": "wiki768", "files": [
        "train-00000-of-00004-1a1932c9ca1c7152.parquet",
        "train-00001-of-00004-f4a4f5540ade14b4.parquet",
        "train-00002-of-00004-ff770df3ab420d14.parquet",
        "train-00003-of-00004-85b3dbbc960e92ec.parquet",
    ]},
#    {"name": "wiki768en", "files": [
#       "0-en.parquet",
#            "1-en.parquet",
#            "2-en.parquet",
#            "3-en.parquet",
#    ]},
#    {"name": "wiki768ja", "files": [
#        "0-ja.parquet",
#        "1-ja.parquet",
#        "2-ja.parquet",
#        "3-ja.parquet",
#    ]},
#    {"name": "wiki768de", "files": [
#        "0-de.parquet",
#        "1-de.parquet",
#        "2-de.parquet",
#        "3-de.parquet",
#    ]},
]


def transform_queries(Q):
    n, _ = Q.shape
    return np.concatenate([Q, np.zeros((n, 1))], axis=-1, dtype=np.float32)


def transform_docs(D, norms):
    n, d = D.shape
    max_norm = magnitudes.max()
    flipped_norms = np.copy(norms).reshape(n, 1)
    transformed_data = np.concatenate([D, np.sqrt(max_norm**2 - flipped_norms**2)], axis=-1, dtype=np.float32)
    return transformed_data


def validate_array_match_upto_dim(arr1, arr2, dim_eq_upto):
    assert np.allclose(arr1[:dim_eq_upto], arr2[:dim_eq_upto]), "data sets are different"


def validate_dataset_match_upto_dim(arr1, arr2, dim_eq_upto):
    n1, d1 = arr1.shape
    n2, d2 = arr2.shape
    assert n1 == n2, f"Shape does not map [{arr1.shape}] vs [{arr2.shape}]"
    for i in range(n1):
        validate_array_match_upto_dim(arr1[i], arr2[i], dim_eq_upto)

dataset = DATA_SETS[0]
name = dataset["name"]
tb1 = pq.read_table(dataset["files"][0], columns=['emb'])
tb2 = pq.read_table(dataset["files"][1], columns=['emb'])
tb3 = pq.read_table(dataset["files"][2], columns=['emb'])
tb4 = pq.read_table(dataset["files"][3], columns=['emb'])
np1 = tb1[0].to_numpy()
np2 = tb2[0].to_numpy()
np3 = tb3[0].to_numpy()
np4 = tb4[0].to_numpy()

np_total = np.concatenate((np1, np2, np3, np4))

#Have to convert to a list here to get
#the numpy ndarray's shape correct later
#There's probably a better way...
flat_ds = list()
for vec in np_total:
    flat_ds.append(vec)
np_flat_ds = np.array(flat_ds)
# we want to mix ja and en
np.random.shuffle(np_flat_ds)
row_count = np_flat_ds.shape[0]
query_count = 10_000
training_rows = row_count - query_count
print(f"{name} num rows: {training_rows}")

with open(f"{name}.test", "w") as out_f:
    np_flat_ds[training_rows:-1].tofile(out_f)

# write the same thing but with each vector normalized by their magnitude
with open(f"{name}.test.norm", "w") as out_f:
    magnitudes = np.linalg.norm(np_flat_ds[training_rows:-1], axis=1)
    # normalize the vectors
    np_flat_ds[training_rows:-1] = np_flat_ds[training_rows:-1] / magnitudes[:, np.newaxis]
    np_flat_ds[training_rows:-1].tofile(out_f)


magnitudes = np.linalg.norm(np_flat_ds[0:training_rows], axis=1)
indices = np.argsort(magnitudes)
np_flat_ds_sorted = np_flat_ds[indices]

with open(f"{name}.train", "w") as out_f:
    np_flat_ds[0:training_rows].tofile(out_f)
with open(f"{name}.ordered.train", "w") as out_f:
    np_flat_ds_sorted.tofile(out_f)

# write the same thing but with each vector normalized by their magnitude
with open(f"{name}.train.norm", "w") as out_f:
    magnitudes = np.linalg.norm(np_flat_ds[0:training_rows], axis=1)
    # normalize the vectors
    np_flat_ds[0:training_rows] = np_flat_ds[0:training_rows] / magnitudes[:, np.newaxis]
    np_flat_ds[0:training_rows].tofile(out_f)
