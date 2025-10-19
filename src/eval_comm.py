import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_edges(path):
    # 支持空格或逗号分隔：src dst weight(ms)
    e = pd.read_csv(path, sep=r"[,\s]+", engine="python",
                    header=None, names=["src","dst","lat_ms"])
    # 清洗
    e["src"] = pd.to_numeric(e["src"], errors="coerce").astype("Int64")
    e["dst"] = pd.to_numeric(e["dst"], errors="coerce").astype("Int64")
    e["lat_ms"] = pd.to_numeric(e["lat_ms"], errors="coerce")
    e = e.dropna(subset=["src","dst","lat_ms"]).astype({"src":int,"dst":int})
    e = e[e["src"] != e["dst"]].copy()

    # 视为无向图：把 (u,v) 与 (v,u) 合并，只保留更小的时延
    e["a"] = np.minimum(e["src"], e["dst"])
    e["b"] = np.maximum(e["src"], e["dst"])
    e = e.sort_values(["a","b","lat_ms"]).drop_duplicates(["a","b"], keep="first")
    e = e[["a","b","lat_ms"]].rename(columns={"a":"src","b":"dst"})
    return e

def load_clusters(path):
    c = pd.read_csv(path)
    # 兼容不同列名
    if "cluster" not in c.columns:
        raise ValueError("clusters csv需要包含列名 'cluster'")
    if "node" not in c.columns:
        c["node"] = np.arange(len(c))
    return c[["node","cluster"]].astype({"node":int,"cluster":int})

def compute_cost(edges, clusters, msg_units=1.0):
    # cost = Σ(latency_ms × message_units) 仅统计存在的边
    clu_map = dict(zip(clusters["node"], clusters["cluster"]))

    # 基线：全图（所有边）
    cost_all = (edges["lat_ms"] * msg_units).sum()
    m_all = len(edges)

    # 仅组内
    same = edges.apply(lambda r: clu_map.get(r["src"], -1) == clu_map.get(r["dst"], -1), axis=1)
    edges_intra = edges[same]
    cost_intra = (edges_intra["lat_ms"] * msg_units).sum()
    m_intra = len(edges_intra)

    # 组间（被“切断”的高延迟通信）
    cost_cross = cost_all - cost_intra
    m_cross = m_all - m_intra

    reduction = (cost_cross / cost_all) if cost_all > 0 else 0.0
    return dict(
        cost_all=cost_all, cost_intra=cost_intra, cost_cross=cost_cross,
        m_all=m_all, m_intra=m_intra, m_cross=m_cross, reduction=reduction
    )

def plot_bar(res, out_path):
    labels = ["All-edges", "Grouped (intra)"]
    vals = [res["cost_all"], res["cost_intra"]]

    plt.figure(figsize=(5,4))
    plt.bar(labels, vals)
    plt.ylabel("Communication cost (ms × units)")
    plt.title("Communication cost before vs after grouping")
    # 标注下降比例
    plt.text(0.5, max(vals)*0.9, f"↓ {res['reduction']*100:.1f}% total",
             ha="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="../data/edge.txt")
    ap.add_argument("--clusters", default="../results/clusters_k4.csv")
    ap.add_argument("--msg_units", type=float, default=1.0,
                    help="每条边上的消息单位量（可当作batch大小比例系数）")
    ap.add_argument("--out", default="../results/comm_cost_k4.png")
    args = ap.parse_args()

    edges = load_edges(args.edges)
    clusters = load_clusters(args.clusters)
    res = compute_cost(edges, clusters, msg_units=args.msg_units)

    print(f"#edges(all)={res['m_all']} | #edges(intra)={res['m_intra']} | #edges(cross)={res['m_cross']}")
    print(f"Cost(all)={res['cost_all']:.2f} | Cost(intra)={res['cost_intra']:.2f} | "
          f"Reduced={res['reduction']*100:.1f}%")

    plot_bar(res, args.out)
    print("Saved plot to:", args.out)

if __name__ == "__main__":
    main()
