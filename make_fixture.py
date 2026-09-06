"""Builds a fake logs/<folder>/topology.top + per-client state files, with
IRREGULAR leaf depths (mirrors the real tree shown: some leaf clusters hang
directly off an internal node while siblings go another 2 levels deep),
purely to sanity-check verify_influence_math.py's loader against the exact
save_metadata format."""
import os
import torch
from collections import OrderedDict

os.makedirs("logs/fixture_run", exist_ok=True)
folder = "logs/fixture_run"

DIM = 6
_next_cid = [1000]


def new_cid():
    _next_cid[0] += 1
    return _next_cid[0]


def client_dict(cid):
    state = OrderedDict(w=torch.randn(DIM))
    fname = f"client_{cid}.pt"
    with open(f"{folder}/{fname}", "wb") as f:
        torch.save(state, f)
    return {"cid": cid, "type": "Client", "model_state_file": fname}


def cluster_dict(cid, member_dicts):
    # cluster center = plain average, just for the fixture
    states = []
    for m in member_dicts.values():
        if m["type"] == "Client":
            with open(f"{folder}/{m['model_state_file']}", "rb") as f:
                states.append(torch.load(f))
        else:
            states.append(m["model_state"])
    avg = OrderedDict(w=sum(s["w"] for s in states) / len(states))
    return {"cid": cid, "type": "Cluster", "model_state": avg, "members": member_dicts}


# Irregular tree:
#           root (3 children)
#     ├── leaf-cluster: clients 0,1,2         <- depth 1 leaf cluster
#     ├── mid (2 children)
#     │     ├── leaf-cluster: 3,4
#     │     └── leaf-cluster: 5,6,7           <- depth 2 leaf clusters
#     └── deep (2 children)
#           ├── leaf-cluster: 8,9
#           └── deeper (2 children)
#                 ├── leaf-cluster: 10,11
#                 └── leaf-cluster: 12,13,14  <- depth 4 leaf clusters

leaf_a = cluster_dict(new_cid(), OrderedDict((c["cid"], c) for c in [client_dict(i) for i in [0, 1, 2]]))
leaf_b = cluster_dict(new_cid(), OrderedDict((c["cid"], c) for c in [client_dict(i) for i in [3, 4]]))
leaf_c = cluster_dict(new_cid(), OrderedDict((c["cid"], c) for c in [client_dict(i) for i in [5, 6, 7]]))
mid = cluster_dict(new_cid(), OrderedDict([(leaf_b["cid"], leaf_b), (leaf_c["cid"], leaf_c)]))

leaf_d = cluster_dict(new_cid(), OrderedDict((c["cid"], c) for c in [client_dict(i) for i in [8, 9]]))
leaf_e = cluster_dict(new_cid(), OrderedDict((c["cid"], c) for c in [client_dict(i) for i in [10, 11]]))
leaf_f = cluster_dict(new_cid(), OrderedDict((c["cid"], c) for c in [client_dict(i) for i in [12, 13, 14]]))
deeper = cluster_dict(new_cid(), OrderedDict([(leaf_e["cid"], leaf_e), (leaf_f["cid"], leaf_f)]))
deep = cluster_dict(new_cid(), OrderedDict([(leaf_d["cid"], leaf_d), (deeper["cid"], deeper)]))

root = cluster_dict(new_cid(), OrderedDict([(leaf_a["cid"], leaf_a), (mid["cid"], mid), (deep["cid"], deep)]))

with open(f"{folder}/topology.top", "wb") as f:
    torch.save(root, f)

print(f"Fixture written to {folder}/  (15 clients, depths 1..4)")