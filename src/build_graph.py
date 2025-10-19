# src/build_graph.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# -------- 1) 读取节点特征（根据你的CSV格式） --------
nodes = pd.read_csv('../data/node_feature.csv', skipinitialspace=True)

# 统一列名
nodes.columns = [c.strip() for c in nodes.columns]
assert set(nodes.columns) >= {"node", "City", "Capability", "Memory"}, \
    f"Unexpected columns: {nodes.columns}"

# 类型清洗
nodes["node"] = pd.to_numeric(nodes["node"], errors="coerce").astype("Int64")
nodes["City"] = nodes["City"].astype(str).str.strip()

# Capability 可能带空格/字符串，转 float
nodes["Capability"] = (
    nodes["Capability"].astype(str).str.strip()
    .str.replace("+", "", regex=False)  # 以防万一
    .str.replace(" ", "", regex=False)
)
nodes["Capability"] = pd.to_numeric(nodes["Capability"], errors="coerce")

# Memory 转数字
nodes["Memory"] = pd.to_numeric(nodes["Memory"], errors="coerce")

# 去掉任何解析失败的行
nodes = nodes.dropna(subset=["node", "Capability", "Memory"]).copy()
nodes["node"] = nodes["node"].astype(int)
nodes = nodes.sort_values("node").reset_index(drop=True)

# City 做类别编码（并保存映射，便于解释）
city_codes, city_uniques = pd.factorize(nodes["City"])
nodes["CityCode"] = city_codes
city_map = {int(i): c for i, c in enumerate(city_uniques.astype(str))}

# 组装特征矩阵（尽量简单：CityCode, Capability, Memory）
feat_df = nodes[["CityCode", "Capability", "Memory"]].fillna(0)
x = torch.tensor(feat_df.values, dtype=torch.float32)

# 节点 id 校验（必须从0开始连续）
expected_n = nodes["node"].max() + 1
assert expected_n == len(nodes), \
    f"node id 应该从0连续编号。现在最大是 {nodes['node'].max()}，但行数是 {len(nodes)}。请检查CSV。"

# -------- 2) 读取边（edge.txt: src dst weight，空格/逗号都支持） --------
edges = pd.read_csv(
    "../data/edge.txt",
    sep=r"[,\s]+", engine="python",
    header=None, names=["src", "dst", "weight"]
)

edges["src"] = pd.to_numeric(edges["src"], errors="coerce").astype("Int64")
edges["dst"] = pd.to_numeric(edges["dst"], errors="coerce").astype("Int64")
edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
edges = edges.dropna(subset=["src", "dst"]).copy()
edges["src"] = edges["src"].astype(int)
edges["dst"] = edges["dst"].astype(int)
edges["weight"] = edges["weight"].fillna(0.0)

# 去自环、去重复，并双向化（当无向图用）
edges = edges[edges["src"] != edges["dst"]].drop_duplicates()
edges_rev = edges.rename(columns={"src": "dst", "dst": "src"})
edges = pd.concat([edges, edges_rev], ignore_index=True).drop_duplicates()

# 边特征：把“时延”转相似度（更适合聚合，数值范围 0~1）
edges["sim"] = 1.0 / (1.0 + edges["weight"])
edge_index = torch.as_tensor(edges[["src", "dst"]].values.T, dtype=torch.long)
edge_attr  = torch.as_tensor(edges[["sim"]].values, dtype=torch.float32)

# -------- 3) 组装 PyG Data --------
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

print(data)  # 例如：Data(x=[N,3], edge_index=[2,E], edge_attr=[E,1])
print(f"x dtype={data.x.dtype}, edge_attr mean={float(edge_attr.mean()):.4f}")
print(f"#nodes={data.num_nodes}, #edges={data.num_edges}")
print("City code map:", city_map)

# 保险检查：src/dst 范围
assert data.edge_index.min() >= 0 and data.edge_index.max() < data.num_nodes, \
    "边里出现了越界的节点编号，请核对 edge.txt 和 node_feature.csv"
