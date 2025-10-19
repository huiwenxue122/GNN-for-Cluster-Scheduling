# src/train_gnn.py
import os, math, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULT_DIR = '../results'
os.makedirs(RESULT_DIR, exist_ok=True)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------- data loader (与 build_graph.py 一致的读取逻辑) ----------
def load_graph():
    # nodes
    nodes = pd.read_csv('../data/node_feature.csv', skipinitialspace=True)
    nodes.columns = [c.strip() for c in nodes.columns]
    nodes["node"] = pd.to_numeric(nodes["node"], errors="coerce").astype("Int64")
    nodes["City"] = nodes["City"].astype(str).str.strip()
    nodes["Capability"] = pd.to_numeric(nodes["Capability"].astype(str).str.replace(" ", ""), errors="coerce")
    nodes["Memory"] = pd.to_numeric(nodes["Memory"], errors="coerce")
    nodes = nodes.dropna(subset=["node", "Capability", "Memory"]).copy()
    nodes["node"] = nodes["node"].astype(int)
    nodes = nodes.sort_values("node").reset_index(drop=True)
    # encode city
    city_codes, city_uniques = pd.factorize(nodes["City"])
    nodes["CityCode"] = city_codes
    feat = nodes[["CityCode", "Capability", "Memory"]].fillna(0.0).values
    x = torch.tensor(feat, dtype=torch.float32)

    # edges
    edges = pd.read_csv("../data/edge.txt", sep=r"[,\s]+", engine="python",
                        header=None, names=["src", "dst", "weight"])
    edges["src"] = pd.to_numeric(edges["src"], errors="coerce").astype("Int64")
    edges["dst"] = pd.to_numeric(edges["dst"], errors="coerce").astype("Int64")
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges = edges.dropna(subset=["src","dst"]).copy()
    edges["src"] = edges["src"].astype(int)
    edges["dst"] = edges["dst"].astype(int)
    edges["weight"] = edges["weight"].fillna(0.0)
    edges = edges[edges["src"] != edges["dst"]].drop_duplicates()
    # 无向图：双向化
    edges_rev = edges.rename(columns={"src":"dst","dst":"src"})
    edges = pd.concat([edges, edges_rev], ignore_index=True).drop_duplicates()
    # 相似度作为 edge_weight
    edges["sim"] = 1.0 / (1.0 + edges["weight"])
    edge_index = torch.as_tensor(edges[["src","dst"]].values.T, dtype=torch.long)
    edge_weight = torch.as_tensor(edges["sim"].values, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    return data

# ---------- model ----------
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, out=32, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid, add_self_loops=True, normalize=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = GCNConv(hid, out, add_self_loops=True, normalize=True)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.act(x)
        x = self.drop(x)
        z = self.conv2(x, edge_index, edge_weight=edge_weight)
        return z  # [N, out]

def linkpred_loss(z, pos_edge_index, neg_edge_index):
    # 点积解码
    def score(ei):
        src, dst = ei
        return (z[src] * z[dst]).sum(dim=-1)
    pos = score(pos_edge_index)
    neg = score(neg_edge_index)
    # 二元交叉熵 (BCE-with-logits)
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    return pos_loss + neg_loss

def train_linkpred(data, epochs=200, lr=1e-3, wd=1e-4, amp=True):
    data = data.to(DEVICE)
    model = GCNEncoder(in_dim=data.x.size(1)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and DEVICE=='cuda'))

    edge_index = data.edge_index
    edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

    # 只用单个 batch（整图训练），每轮做一次负采样
    best = (math.inf, None)  # (loss, state_dict)
    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(amp and DEVICE=='cuda')):
            z = model(data.x, edge_index, edge_weight=edge_weight)
            # 正样本：原图边（无向图会重复，OK）
            pos_ei = edge_index
            # 负样本：与正边同数目的随机无连接边
            neg_ei = negative_sampling(edge_index=edge_index,
                                       num_nodes=data.num_nodes,
                                       num_neg_samples=pos_ei.size(1))
            loss = linkpred_loss(z, pos_ei, neg_ei)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if loss.item() < best[0]:
            best = (loss.item(), {k: v.detach().cpu() for k, v in model.state_dict().items()})

        if epoch % 20 == 0 or epoch == 1:
            print(f"[{epoch:03d}] loss={loss.item():.4f}")

    # 还原最优
    if best[1] is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best[1].items()})
    model.eval()
    with torch.no_grad():
        z = model(data.x, edge_index, edge_weight=edge_weight).detach().cpu().numpy()
    np.save(os.path.join(RESULT_DIR, "node_embeddings.npy"), z)
    return z

def kmeans_cluster(z, k=4):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    y = km.fit_predict(z)
    pd.DataFrame({"node": np.arange(len(y)), "cluster": y}).to_csv(
        os.path.join(RESULT_DIR, f"clusters_k{k}.csv"), index=False)
    print("Cluster counts:", np.bincount(y))
    return y

if __name__ == "__main__":
    print("Device:", DEVICE)
    t0 = time.time()
    data = load_graph()
    print(data)

    # 训练（你可以把 epochs 改为 100~300 之间）
    z = train_linkpred(data, epochs=200, lr=1e-3, wd=1e-4, amp=True)
    # 聚类个数你可以按任务数/城市数调整；先给 4 作演示
    y = kmeans_cluster(z, k=4)

    # 方便快速查看几个节点的聚类结果
    print("Preview:", list(zip(range(10), y[:10])))
    print("Saved to:", RESULT_DIR)
    print(f"Done in {time.time()-t0:.1f}s")
