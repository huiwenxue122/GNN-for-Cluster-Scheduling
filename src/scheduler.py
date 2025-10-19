# src/scheduler.py
import argparse, os
import pandas as pd, numpy as np

def load_nodes_edges():
    nodes = pd.read_csv("../data/node_feature.csv", skipinitialspace=True)
    nodes.columns = [c.strip() for c in nodes.columns]
    nodes["node"] = pd.to_numeric(nodes["node"], errors="coerce").astype(int)
    nodes["Capability"] = pd.to_numeric(nodes["Capability"], errors="coerce")
    nodes["Memory"] = pd.to_numeric(nodes["Memory"], errors="coerce")
    nodes = nodes.sort_values("node").reset_index(drop=True)

    e = pd.read_csv("../data/edge.txt", sep=r"[,\s]+", engine="python",
                    header=None, names=["src","dst","lat"])
    e = e.dropna().astype({"src":int,"dst":int,"lat":float})
    # 无向去重：保留更小lat
    e["a"]=np.minimum(e.src,e.dst); e["b"]=np.maximum(e.src,e.dst)
    e=e.sort_values(["a","b","lat"]).drop_duplicates(["a","b"], keep="first")[["a","b","lat"]]
    e=e.rename(columns={"a":"src","b":"dst"}).reset_index(drop=True)
    return nodes, e

def load_clusters(path=None, k=None, n=None):
    """优先读取 clusters_k*.csv；没有则按City聚类/或全体一组。"""
    if path and os.path.exists(path):
        c = pd.read_csv(path)
        if "node" not in c.columns: c["node"]=np.arange(len(c))
        return c.sort_values("node")[["node","cluster"]].reset_index(drop=True)
    # fallback：单组
    return pd.DataFrame({"node": np.arange(n), "cluster": 0})

def group_stats(nodes, edges, clu):
    """计算每个组的显存总量、平均算力、组内平均时延等指标。"""
    stats = []
    for g, sub in nodes.merge(clu, on="node").groupby("cluster"):
        Ns = sub["node"].tolist()
        mem = sub["Memory"].sum()
        cap = sub["Capability"].mean()
        # 组内边
        mask = (edges.src.isin(Ns) & edges.dst.isin(Ns))
        lat_intra = edges[mask]["lat"].mean() if mask.any() else np.inf
        stats.append({"cluster": int(g), "nodes": Ns, "mem_sum": float(mem),
                      "cap_mean": float(cap), "lat_intra": float(lat_intra)})
    stat_df = pd.DataFrame(stats).sort_values(["lat_intra","cap_mean"], ascending=[True,False]).reset_index(drop=True)
    return stat_df

def inter_group_latency(edges, A_nodes, B_nodes):
    m = edges[(edges.src.isin(A_nodes) & edges.dst.isin(B_nodes)) |
              (edges.src.isin(B_nodes) & edges.dst.isin(A_nodes))]
    return m["lat"].mean() if len(m) else np.inf

def schedule(tasks, nodes, edges, clu):
    """Algorithm 1 的简化实现：不足→合并最近组。"""
    gstats = group_stats(nodes, edges, clu)
    used_groups=set()
    assign={}
    for t in tasks:  # 任务按优先级顺序
        need_mem = t["min_mem_gb"]
        # 候选：未使用组，按“高算力、低时延”排序
        cand = gstats[~gstats["cluster"].isin(used_groups)].copy()
        if cand.empty:
            assign[t["name"]]={"groups":[], "nodes":[],"mem":0}
            continue
        # 先挑一个最佳起始组
        best = cand.sort_values(["lat_intra","cap_mean"], ascending=[True,False]).iloc[0]
        chosen_nodes=set(best["nodes"]); chosen_groups=[best["cluster"]]; total_mem=best["mem_sum"]
        used_groups.add(best["cluster"])

        # 不足阈值 → 合并“最近的下一组”
        while total_mem < need_mem:
            rest = cand[~cand["cluster"].isin(chosen_groups)]
            if rest.empty: break
            # 与已选集合的“平均跨组时延”最小者
            rest = rest.copy()
            rest["lat_cross"] = rest["nodes"].apply(lambda ns: 
                np.mean([inter_group_latency(edges, list(chosen_nodes), ns)]) )
            pick = rest.sort_values(["lat_cross","cap_mean"], ascending=[True,False]).iloc[0]
            chosen_groups.append(pick["cluster"])
            chosen_nodes.update(pick["nodes"])
            total_mem += pick["mem_sum"]
            used_groups.add(pick["cluster"])

        assign[t["name"]]={"groups": chosen_groups, "nodes": sorted(chosen_nodes), "mem": total_mem}
    return assign

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clusters", default="../results/clusters_k4.csv")
    ap.add_argument("--out", default="../results/schedule.json")

    args=ap.parse_args()

    nodes, edges = load_nodes_edges()
    clu = load_clusters(args.clusters, n=len(nodes))

    # 可以按论文规模自定义任务（参数只是示意）
    tasks = [
        {"name":"GPT2-1.5B", "min_mem_gb": 8*6},     # 例如需要 ~48GB 可用显存
        {"name":"BERT-large", "min_mem_gb": 8*3},    # ~24GB
        {"name":"T5-11B", "min_mem_gb": 8*8},        # ~64GB
    ]
    res = schedule(tasks, nodes, edges, clu)

    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    #with open(args.out,"w",encoding="utf-8") as f: 
        #json.dump(res, f, indent=2)
    # ---- 解决 np.int64 / np.float32 无法序列化的问题 ----
    def _to_py(o):
        if isinstance(o, (np.generic,)):       # numpy 标量类型
            return o.item()
        if isinstance(o, set):
            return list(o)
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_py(v) for v in o]
        return o
    # -----------------------------------------------------

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_to_py(res), f, indent=2, ensure_ascii=False)
    print("=== Schedule ===")
    for k,v in res.items():
        print(k, "groups:", v["groups"], "| #nodes:", len(v["nodes"]), "| mem_sum:", f'{v["mem"]:.1f}GB')
    print("Saved:", args.out)

if __name__=="__main__":
    main()
