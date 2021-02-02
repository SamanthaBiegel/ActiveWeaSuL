# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

path_prefix = "../"

# +
# %load_ext autoreload
# %autoreload 2

import csv
import json
import pickle
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath("../activelearning"))
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
# -

# ### Load Visual Genome relationships

with open(path_prefix + 'data/visual_genome/relationships.json') as f:
  visgen_rels = json.load(f)

# ### Filter and process data

# +
pred_list = ['carrying',
 'covered in',
 'covering',
 'eating',
 'flying in',
 'growing on',
 'hanging from',
 'lying on',
 'mounted on',
 'painted on',
 'parked on',
 'playing',
 'riding',
 'says',
 'sitting on',
 'standing on',
 'using',
 'walking in',
 'walking on',
 'watching',
 'wearing']

pred_action = "sitting on"

# +
visgen_df = pd.json_normalize(visgen_rels, record_path=["relationships"], meta="image_id", sep="_")
visgen_df["predicate"] = visgen_df["predicate"].str.lower()

# Filter for selected predicates
visgen_df_actions = visgen_df[visgen_df["predicate"].isin(pred_list)]
# -

# Assign binary target labels to selected predicates
visgen_df_actions.loc[:, "y"] = visgen_df_actions.loc[:, "predicate"].apply(lambda x: 1 if x == pred_action else 0)

# Select relevant features and rename
df_vis = (
visgen_df_actions.loc[:,["image_id", "predicate", "object_name", "object_h", "object_w", "object_y", "object_x", "subject_name", "subject_h", "subject_w", "subject_y", "subject_x", "y"]]
                 .rename(columns={"object_name": "object_category", "subject_name": "subject_category", "image_id": "source_img"})
)

# Filter for valid data points
df_vis = df_vis.dropna()
df_vis = df_drop_duplicates(df_vis)
df_vis = balance_dataset(df_vis)

# Extract image channels
df_vis.source_img = df_vis.source_img.astype(str) + ".jpg"
df_vis["channels"] = df_vis["source_img"].apply(lambda x: len(np.array(Image.open(path_prefix + "data/visual_genome/VG_100K" + "/" + x)).shape))

# +
# Process bounding box coordinates
df_vis["object_x_max"] = df_vis["object_x"] + df_vis["object_w"]
df_vis["object_y_max"] = df_vis["object_y"] + df_vis["object_h"]
df_vis["subject_x_max"] = df_vis["subject_x"] + df_vis["subject_w"]
df_vis["subject_y_max"] = df_vis["subject_y"] + df_vis["subject_h"]

df_vis["object_bbox"] = tuple(df_vis[["object_y", "object_y_max", "object_x", "object_x_max"]].values)
df_vis["subject_bbox"] = tuple(df_vis[["subject_y", "subject_y_max", "subject_x", "subject_x_max"]].values)
# -

word_embs = pd.read_csv(
            path_prefix + "data/word_embeddings/glove.6B.100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        ).T
word_embs = list(word_embs.columns)

# +
# Filter for valid embeddings and image channels
valid_embeddings = (df_vis["channels"] == 3) & df_vis.object_category.isin(word_embs) & df_vis.subject_category.isin(word_embs) & ~df_vis["object_category"].str.contains(" ") & ~df_vis["subject_category"].str.contains(" ")

df_vis_final = df_vis[valid_embeddings]
df_vis_final.index = list(range(len(df_vis_final)))
# -

# ### Split data

np.random.seed(633)
indices_shuffle = np.random.permutation(df_vis_final.shape[0])

# +
split_nr = int(np.ceil(0.7*df_vis_final.shape[0]))
train_idx, test_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]

df_train = df_vis_final.iloc[train_idx]
df_test = df_vis_final.iloc[test_idx]
df_train.index = list(range(len(df_train)))
df_test.index = list(range(len(df_test)))
# -

df_train.to_csv(path_prefix + "data/VG_train.csv", index=False)
df_test.to_csv(path_prefix + "data/VG_test.csv", index=False)

# ### Compute embeddings

# +
dataset_train = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_train,
                      Y=df_train.y.values)

dl_train = DataLoader(dataset_train, shuffle=False, batch_size=256)

final_model = VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix, soft_labels=False)

feature_tensor_train = torch.Tensor([])

for batch_features, batch_labels in dl_train:
    feature_tensor_train = torch.cat((feature_tensor_train, final_model.extract_concat_features(batch_features).to("cpu")))

# +
dataset_test = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_test,
                      Y=df_test.y.values)

dl_test = DataLoader(dataset_test, shuffle=False, batch_size=256)

final_model = VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix, soft_labels=False)

feature_tensor_test = torch.Tensor([])

for batch_features, batch_labels in dl_test:
    feature_tensor_test = torch.cat((feature_tensor_test, final_model.extract_concat_features(batch_features).to("cpu")))
# -

feature_tensor_train.shape

feature_tensor_test.shape

torch.save(feature_tensor_train, "../data/train_embeddings.pt")

torch.save(feature_tensor_test, "../data/test_embeddings.pt")






