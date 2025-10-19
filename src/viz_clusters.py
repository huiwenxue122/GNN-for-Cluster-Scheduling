# src/viz_clusters.py
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 可选：消除 OpenMP 冲突警告
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def load_nodes(path_nodes):
    nodes = pd.read_csv(path_nodes, skipinitialspace=True)
    nodes.columns = [c.strip() for c in nodes.columns]
    # 只保留需要的列
    nodes = nodes[["node", "City"]].copy()
    nodes["node"] = pd.to_numeric(nodes["node"], errors="coerce").astype(int)
    nodes["City"] = nodes["City"].astype(str).str.strip()
    nodes = nodes.sort_values("node").reset_index(drop=True)
    # 城市编码（用于稳定调色）
    city2id = {c:i for i,c in enumerate(sorted(nodes["City"].unique()))}
    nodes["CityId"] = nodes["City"].map(city2id)
    return nodes, city2id

def plot_scatter(emb, color_vals, title, out_path, label_names=None):
    plt.figure(figsize=(6,5))
    sc = plt.scatter(emb[:,0], emb[:,1], c=color_vals, s=36, alpha=0.9)
    plt.xticks([]); plt.yticks([])
    plt.title(title)
    if label_names is not None and len(label_names) <= 12:
        # 只在类别不多时放图例
        handles, _ = sc.legend_elements()
        labels = [label_names[i] for i in range(len(handles))]
        plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def reduce_dim(emb, method="umap", random_state=42):
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, emb.shape[0]//3)), random_state=random_state)
        e2 = reducer.fit_transform(emb)
    else:
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=min(15, max(5, emb.shape[0]-2)), min_dist=0.1, metric="euclidean", random_state=random_state)
        e2 = reducer.fit_transform(emb)
    return e2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="../data/node_feature.csv")
    ap.add_argument("--emb", default="../results/node_embeddings.npy")
    ap.add_argument("--clusters", default="../results/clusters_k4.csv")
    ap.add_argument("--method", choices=["umap","tsne"], default="umap")
    ap.add_argument("--out_city", default="../results/emb_city.png")
    ap.add_argument("--out_cluster", default="../results/emb_cluster.png")
    args = ap.parse_args()

    # 读取
    nodes, city2id = load_nodes(args.nodes)
    emb = np.load(args.emb)
    clus = pd.read_csv(args.clusters)
    # 对齐排序
    clus = clus.sort_values("node").reset_index(drop=True)
    assert emb.shape[0] == len(nodes) == len(clus), "embedding/节点/聚类数量不一致，请检查。"

    # 降维到2D
    emb2 = reduce_dim(emb, method=args.method)

    # 图1：按城市上色
    plot_scatter(emb2, nodes["CityId"].values,
                 title=f"Node embeddings colored by City ({args.method})",
                 out_path=args.out_city,
                 label_names=[c for c,_ in sorted(city2id.items(), key=lambda x:x[1])])

    # 图2：按聚类上色
    k = clus["cluster"].nunique()
    plot_scatter(emb2, clus["cluster"].values,
                 title=f"Node embeddings colored by Cluster (k={k}, {args.method})",
                 out_path=args.out_cluster,
                 label_names=[f"C{i}" for i in range(k)])

    print("Saved:", args.out_city)
    print("Saved:", args.out_cluster)

if __name__ == "__main__":
    main()
