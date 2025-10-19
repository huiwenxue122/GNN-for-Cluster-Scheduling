# src/eval_systems.py
import os, json, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_nodes_edges():
    nodes = pd.read_csv("../data/node_feature.csv", skipinitialspace=True)
    nodes.columns=[c.strip() for c in nodes.columns]
    nodes["node"]=pd.to_numeric(nodes["node"], errors="coerce").astype(int)
    nodes["Capability"]=pd.to_numeric(nodes["Capability"], errors="coerce")
    nodes["Memory"]=pd.to_numeric(nodes["Memory"], errors="coerce")
    nodes=nodes.sort_values("node").reset_index(drop=True)

    e = pd.read_csv("../data/edge.txt", sep=r"[,\s]+", engine="python",
                    header=None, names=["src","dst","lat"])
    e = e.dropna().astype({"src":int,"dst":int,"lat":float})
    e["a"]=np.minimum(e.src,e.dst); e["b"]=np.maximum(e.src,e.dst)
    e=e.sort_values(["a","b","lat"]).drop_duplicates(["a","b"], keep="first")[["a","b","lat"]]
    e=e.rename(columns={"a":"src","b":"dst"}).reset_index(drop=True)
    return nodes,e

def edge_sum_latency(edges, nodes_used, only_intra=False, groups=None):
    S=set(nodes_used)
    if only_intra and groups is not None:
        # 仅组内：分组字典 node->gid
        gid = {n:g for g,ns in groups.items() for n in ns}
        m = edges[(edges.src.isin(S) & edges.dst.isin(S) &
                   (edges.apply(lambda r: gid.get(r.src,-1)==gid.get(r.dst,-1), axis=1)))]
    else:
        m = edges[(edges.src.isin(S) & edges.dst.isin(S))]
    return float(m["lat"].sum())

def compute_time(params_b, caps_sum, alpha=1.0):
    return alpha * params_b / max(caps_sum, 1e-6)

def simulate_systems(tasks, nodes, edges, schedule_path, betas, alpha=1.0):
    # Hulk：从 schedule.json 读取每任务的 nodes
    with open(schedule_path,"r",encoding="utf-8") as f:
        sch=json.load(f)
    # groups dict for intra check
    hulk_groups={t:sch[t]["nodes"] for t in sch}
    node2group={}
    for gi,(t,ns) in enumerate(hulk_groups.items()):
        for n in ns: node2group[n]=gi

    results={}
    for name, conf in tasks.items():
        params = conf["params_b"]
        # ---- System A: Data Parallel（用该任务分到的节点）
        nodes_A = sch.get(name,{}).get("nodes",[])
        caps_A = nodes[nodes["node"].isin(nodes_A)]["Capability"].sum()
        comm_A = betas["DP"] * edge_sum_latency(edges, nodes_A, only_intra=False)
        comp_A = compute_time(params, caps_A, alpha)
        results.setdefault("SystemA-DP",[]).append(dict(task=name, comp=comp_A, comm=comm_A))

        # ---- System B: Pipeline（同样用该任务节点，但只统计“组内边”（近似流水段））
        comm_B = betas["PP"] * edge_sum_latency(edges, nodes_A, only_intra=True,
                                                groups={0:nodes_A})
        comp_B = compute_time(params, caps_A, alpha)
        results.setdefault("SystemB-PP",[]).append(dict(task=name, comp=comp_B, comm=comm_B))

        # ---- System C: Tensor（用任务节点，统计所有边但β更小）
        comm_C = betas["TP"] * edge_sum_latency(edges, nodes_A, only_intra=False)
        comp_C = compute_time(params, caps_A, alpha)
        results.setdefault("SystemC-TP",[]).append(dict(task=name, comp=comp_C, comm=comm_C))

        # ---- Hulk：仅统计“组内通信”，跨组不需要
        groups_for_task = {gi:ns for gi,(t,ns) in enumerate(hulk_groups.items()) if t==name}
        comm_H = betas["H"] * edge_sum_latency(edges, nodes_A, only_intra=True, groups=groups_for_task)
        comp_H = compute_time(params, caps_A, alpha)
        results.setdefault("Hulk",[]).append(dict(task=name, comp=comp_H, comm=comm_H))
    return results

def summarize_and_plot(results, out_png="../results/systems_compare.png"):
    systems=list(results.keys())
    # 聚合所有任务（求和）
    sums=[]
    for s in systems:
        comp=sum(x["comp"] for x in results[s])
        comm=sum(x["comm"] for x in results[s])
        sums.append((s, comp, comm, comp+comm))
    df=pd.DataFrame(sums, columns=["system","compute","communication","total"]).sort_values("total")
    print("\n=== Total time (lower is better) ===")
    print(df.to_string(index=False, formatters={"compute":"{:,.2f}".format,
                                               "communication":"{:,.2f}".format,
                                               "total":"{:,.2f}".format}))
    # plot
    x=np.arange(len(df)); w=0.6
    plt.figure(figsize=(7,5))
    plt.bar(x, df["compute"], width=w, label="Compute")
    plt.bar(x, df["communication"], width=w, bottom=df["compute"], label="Communication")
    plt.xticks(x, df["system"])
    plt.ylabel("Relative time (arb. units)")
    plt.title("System comparison (lower is better)")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print("Saved plot to:", out_png)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--schedule", default="../results/schedule.json")
    ap.add_argument("--out", default="../results/systems_compare.png")
    args=ap.parse_args()

    nodes, edges = load_nodes_edges()
    # 任务参数（可按需要调整数量级/相对规模）
    tasks={
        "GPT2-1.5B":{"params_b":1.5},
        "BERT-large":{"params_b":0.34},
        "T5-11B":{"params_b":11.0},
    }
    # 并行策略的“单位消息量系数” β（可调，但保持 DP > TP > PP ≈ H）
    betas={"DP":1.0, "PP":0.33, "TP":0.83, "H":0.085}
    # 计算时间比例系数（影响绝对值，不影响“谁更小”的排序）
    #alpha=80.0
    # ——自动标定 alpha：让平均计算时间 ≈ 平均通信时间×target_ratio——
    # 先用一个临时 alpha 算出每个任务的 params/caps
    tmp_alpha = 1.0
    nodes, edges = load_nodes_edges()
    tasks = {
        "GPT2-1.5B":{"params_b":1.5},
        "BERT-large":{"params_b":0.34},
        "T5-11B":{"params_b":11.0},
    }
    with open("../results/schedule.json","r",encoding="utf-8") as f:
        sch = json.load(f)
    caps_list, comm_list = [], []
    for name, conf in tasks.items():
        nodes_A = sch.get(name,{}).get("nodes",[])
        caps = nodes[nodes["node"].isin(nodes_A)]["Capability"].sum()
        caps_list.append(conf["params_b"]/max(caps,1e-6))
        comm = edge_sum_latency(edges, nodes_A, only_intra=False)
        comm_list.append(comm)

    target_ratio = 0.8  # 希望“平均计算时间”≈“平均通信时间”的80%
    mean_params_over_caps = np.mean(caps_list)
    mean_comm = np.mean(comm_list)
    alpha = target_ratio * mean_comm / max(mean_params_over_caps, 1e-9)
    print(f"[auto] alpha calibrated to {alpha:.1f}")


    res=simulate_systems(tasks, nodes, edges, args.schedule, betas, alpha=alpha)
    summarize_and_plot(res, args.out)

if __name__=="__main__":
    main()
