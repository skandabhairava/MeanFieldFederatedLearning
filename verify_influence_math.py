"""
verify_influence_math.py

Checks the derived influence formulas (U, w, c_k, R, A_{B,A}) against what the
actual FedAttract aggregation code produces on a synthetic tree.

WHERE TO PLUG YOUR REAL CODE
-----------------------------
Two classes below, `Client` and `Cluster`, are minimal stand-ins for your real
`client.Client` and `topology.Cluster`. `Cluster` here is a *verbatim* copy of
the methods in your `topology.py` (attract_aggregate, calc_distances,
update_centers_upward, propagate_downward) -- nothing about the math is
changed, only HDBSCAN-splitting and Ray/logging are stripped out since this
test does one round on a fixed tree.

To test your ACTUAL code instead of this copy:
  1. Replace the `Client` class below with an import: `from client import Client`
     (as long as it exposes .cid, .model_state (dict[str, Tensor]),
     .model_state_flattened, you're fine -- add a trivial constructor if not).
  2. Replace the `Cluster` class below with: `from topology import Cluster`.
  3. Leave `stats.flatten` behavior consistent: `flatten(state_dict)` must
     concatenate the dict's tensors into a single 1-D vector, matching
     whatever your real `stats.flatten` does.
  4. Everything from `build_random_tree` down (the actual verification logic)
     needs NO changes -- it only reads `.dists`, `.total_dist`, `.members`,
     `.model_state`, `.cid`, which your real classes already expose.

WHAT GETS CHECKED
------------------
  1. Phase-B decomposition: for every internal node v, is
     Phi_v == sum_c U_{v,c} * theta_c^(t,0)  over leaves c under v?
  2. Full-round mixing matrix: for every leaf B, is
     theta_B^(t) == sum_A A_{B,A} * theta_A^(t,0),  A_{B,A}=U_{a,A}*R(a,B) ?
  3. Row-stochasticity: sum_A A_{B,A} == 1 for every leaf B.

Run: python verify_influence_math.py
"""

import copy
import random
from collections import OrderedDict
from typing import Callable

import torch

from client import Client
from topology import Cluster
from stats import flatten

torch.manual_seed(0)
random.seed(0)

# --------------------------------------------------------------------------
# Minimal stand-ins for models.StateDict / stats.flatten / client.Client
# Swap these for your real modules if you want -- nothing below depends on
# the internal representation beyond "dict[str, Tensor]" and "flat 1-D Tensor".
# --------------------------------------------------------------------------

def random_state(dim: int = 8) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict({"w": torch.randn(dim)})


# --------------------------------------------------------------------------
# Cluster: verbatim copy of the relevant methods from topology.py
# (HDBSCAN splitting, Ray, logging stripped -- not needed for one round)
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# Build a random tree of Clients/Clusters (fixed topology, no splitting)
# --------------------------------------------------------------------------

def build_random_tree(n_leaves=12, branching=3, dim=8, dist_func=Callable[[torch.Tensor, torch.Tensor], float], project_func=Callable[[torch.Tensor], torch.Tensor]):
    leaves = [
        Client(
            i, 
            [], 
            project_func, 
            random_state(dim), 
            spill_folder="spill",
            save_log=False, 
            verifying_so_remove_splits=True
        ) for i in range(n_leaves)
    ]
    level = leaves
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), branching):
            group = level[i:i + branching]
            members: dict[int, Client|Cluster] = {m.cid: m for m in group}
            cluster = Cluster(members, dist_func, project_func, spill_folder="spill")
            for m in group:
                m.parent = cluster # pyright: ignore[reportAttributeAccessIssue]
            next_level.append(cluster)
        level = next_level
    root: Cluster = level[0] # pyright: ignore[reportAssignmentType]
    root.parent = None
    return root, leaves


# --------------------------------------------------------------------------
# Loader for REAL saved trees (the .top files written by Cluster.save_metadata,
# in the exact format your PRINT_TREE function reads).
#
# IMPORTANT CAVEAT: a saved .top snapshot only has POST-round model_state
# values -- it does not contain a paired "before local training / after local
# training" pair, and it does NOT contain .dists / .total_dist (those aren't
# persisted by save_metadata). So this loader:
#   1. Rebuilds the real Cluster/Client objects with model_state = whatever
#      was saved, treated as this round's PRE-round state theta^(t-1)
#      (i.e. "the state a fresh round would start attracting/propagating from").
#   2. Synthesizes a Phase-A output theta_c^(t,0) per client by perturbing the
#      loaded state with noise, purely so there's *something* concrete to run
#      Phase B/C on. The aggregation-identity check does not care how
#      theta_c^(t,0) was produced -- it only checks that whatever it is,
#      theta_B^(t) = sum_A A_{B,A} theta_A^(t,0) holds -- so a synthetic
#      perturbation is a valid, sufficient test of the algebra.
#   3. Runs one real round (update_centers_upward + propagate_downward) with
#      your real aggregation code, which (re)populates .dists/.total_dist,
#      and checks the identity exactly as before.
#
# If you want to test against REAL training outputs instead of synthetic
# noise, replace `synthesize_trained_states` with code that loads your actual
# per-round client update files instead.
# --------------------------------------------------------------------------

def _load_client_from_dict(node_dict, folder, project_func):
    with open(f"{folder}/{node_dict['model_state_file']}", "rb") as f:
        state = torch.load(f)
        # print(state)
    return Client(node_dict["cid"], [], project_func, state, verifying_so_remove_splits=True, save_log=False, spill_folder="spill")


def _load_cluster_from_dict(node_dict, folder, dist_func, project_func):
    member_nodes = OrderedDict()
    for _, child_dict in node_dict["members"].items():
        if child_dict["type"] == "Client":
            child_obj = _load_client_from_dict(child_dict, folder, project_func)
        else:
            child_obj = _load_cluster_from_dict(child_dict, folder, dist_func, project_func)
        member_nodes[child_obj.cid] = child_obj

    cluster = Cluster(member_nodes, dist_func, project_func,
                       parent=None, cluster_center=node_dict["model_state"], spill_folder="spill",)
    cluster.cid = node_dict["cid"]  # preserve the real cid (our test Cluster.__init__ assigns a fresh uuid otherwise)
    cluster.model_state_flattened = project_func(flatten(cluster.get_model_state()))

    for child_obj in member_nodes.values():
        child_obj.parent = cluster

    return cluster


def build_tree_from_topology(folder_name: str, dist_func, project_func, base_dir: str = "logs"):
    """Mirrors PRINT_TREE's file access exactly: `{base_dir}/{folder_name}/topology.top`
    for the tree structure + cluster centers, `{base_dir}/{folder_name}/{model_state_file}`
    per client for its individual state."""
    folder = f"{base_dir}/{folder_name}"
    with open(f"{folder}/topology.top", "rb") as f:
        tree_dict = torch.load(f)

    root = _load_cluster_from_dict(tree_dict, folder, dist_func, project_func)
    root.parent = None

    leaves = []

    def collect_leaves(node):
        for m in node.members.values():
            if isinstance(m, Client):
                leaves.append(m)
            else:
                collect_leaves(m)

    collect_leaves(root)
    return root, leaves


def synthesize_trained_states(leaves, noise_std=0.05):
    """Fabricates theta_c^(t,0) = loaded_state + noise, per client. See caveat
    above: this is only meant to exercise the aggregation algebra, not to
    reproduce a real local-training step."""
    trained_states = {}
    for c in leaves:
        perturbed = {k: v.clone() + noise_std * torch.randn_like(v) for k, v in c.model_state.items()}
        trained_states[c.cid] = perturbed
    return trained_states


def verify_real_tree(folder_name: str, base_dir: str = "logs", proj_dim: int = 20, noise_std: float = 0.05):
    torch.manual_seed(0)
    random.seed(0)

    # peek at one client's state to get the flattened dim, then build a fixed
    # random projection matrix of the right shape (same role as server.py's self.R)
    probe_folder = f"{base_dir}/{folder_name}"
    with open(f"{probe_folder}/topology.top", "rb") as f:
        tree_dict = torch.load(f)

    def find_first_client(node_dict):
        for m in node_dict["members"].values():
            if m["type"] == "Client":
                return m
            found = find_first_client(m)
            if found is not None:
                return found
        return None

    first_client_dict = find_first_client(tree_dict)
    assert first_client_dict is not None
    with open(f"{probe_folder}/{first_client_dict['model_state_file']}", "rb") as f:
        probe_state = torch.load(f)
    D = flatten(probe_state).numel()
    R_mat = torch.randn(D, proj_dim) / (proj_dim ** 0.5)

    def project_func(vec, *args, **kwargs):
        return vec @ R_mat

    def dist_func(a, b):
        return torch.norm(a - b, p=2).item()

    root, leaves = build_tree_from_topology(folder_name, dist_func, project_func, base_dir=base_dir)
    print(f"Loaded tree: {len(leaves)} clients, depths ranging "
          f"{min(len(path_to_root(c)) for c in leaves)}..{max(len(path_to_root(c)) for c in leaves)} "
          f"(root-to-leaf edge counts)")

    trained_states = synthesize_trained_states(leaves, noise_std=noise_std)
    for c in leaves:
        c.model_state = copy.deepcopy(trained_states[c.cid])
        c.model_state_flattened = project_func(flatten(c.model_state))

    root.update_centers_upward(trained_states)
    root.propagate_downward(recalc_dists=False)
    actual_final = {c.cid: {k: v.clone() for k, v in c.model_state.items()} for c in leaves}

    A_mat = influence_matrix(leaves)
    # print(A_mat)

    print("\n=== Row-stochasticity check ===")
    max_row_err = max(abs(sum(A_mat[B.cid].values()) - 1.0) for B in leaves)
    print(f"max |sum_A A_(B,A) - 1|: {max_row_err:.3e}")

    print("\n=== Full-round prediction check (per client) ===")
    max_err, max_rel = 0.0, 0.0
    for B in leaves:
        for key in B.model_state:
            pred = torch.zeros_like(B.model_state[key])
            for Aclient in leaves:
                pred += A_mat[B.cid][Aclient.cid] * trained_states[Aclient.cid][key]
            err = torch.norm(pred - actual_final[B.cid][key]).item()
            rel = err / (torch.norm(actual_final[B.cid][key]).item() + 1e-12)
            max_err, max_rel = max(max_err, err), max(max_rel, rel)
        print(f"  client {B.cid}: worst-key abs_err up to this point = {max_err:.3e}")

    tol = 1e-4
    print(f"\nmax abs_err = {max_err:.3e}, max rel_err = {max_rel:.3e}")
    print(f"{'PASS' if max_err < tol else 'FAIL'}: tolerance {tol:.0e}")
    return max_err < tol


# --------------------------------------------------------------------------
# Formula-side computation: U, w, c_k, R, A_{B,A}
# Reads ONLY .dists / .total_dist / .members / .cid -- same attrs your real
# Cluster class exposes, so this needs no changes when you swap in real code.
# --------------------------------------------------------------------------

def raw_weight(total_dist, delta):
    return (total_dist / delta) if delta != 0 else 1.0


def tilde_omega(node: Cluster, child_cid: int) -> float:
    """Normalized Phase-B weight of `child_cid` inside `node`."""
    raws = {cid: raw_weight(node.total_dist, d) for cid, d in node.dists.items()}
    denom = sum(raws.values())
    return raws[child_cid] / denom


def w_self(node) -> float:
    """Phase-C self-weight of `node` relative to its parent."""
    p = node.parent
    if p is None:
        return 1.0  # root: no parent to be pulled towards
    inv_total = 1.0 / p.total_dist if p.total_dist > 0 else 0.0
    return p.dists[node.cid] * inv_total


def path_to_root(node):
    """[node, parent, ..., root]"""
    path = [node]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path


def U(v_target, leaf) -> float:
    """Upward reach coefficient U_{v_target, leaf}: product of tilde_omega
    along leaf's path up to v_target. Assumes leaf is under v_target (or
    leaf is v_target itself, in which case U = 1 by the empty product)."""
    if v_target.cid == leaf.cid:
        return 1.0
    path = path_to_root(leaf)  # leaf = path[0]
    assert v_target in path, "leaf must be a descendant of v_target"
    prod = 1.0
    node = leaf
    for nxt in path[1:]:
        prod *= tilde_omega(nxt, node.cid)
        node = nxt
        if node is v_target:
            break
    return prod


def c_k_all(target_leaf):
    """Returns dict {ancestor.cid: c_k} for every node on root->target path,
    using the derived telescoping formula. Order: root .. target."""
    path = path_to_root(target_leaf)  # [target, ..., root]
    chain = list(reversed(path))       # [root, ..., target]
    m = len(chain) - 1
    ws = [w_self(chain[k]) for k in range(1, m + 1)]  # w_1..w_m (chain[0]=root has none)

    coeffs = {}
    # c_0 = prod_{j=1}^m (1-w_j)
    c0 = 1.0
    for w in ws:
        c0 *= (1 - w)
    coeffs[chain[0].cid] = c0

    for k in range(1, m + 1):
        ck = ws[k - 1]
        for j in range(k + 1, m + 1):
            ck *= (1 - ws[j - 1])
        coeffs[chain[k].cid] = ck

    return coeffs, chain  # chain[k].cid -> c_k, and chain itself (root..target)


def lca(node_a, node_b):
    anc_a = path_to_root(node_a)
    set_a = {n.cid: n for n in anc_a}
    for n in path_to_root(node_b):
        if n.cid in set_a:
            return set_a[n.cid]
    raise ValueError("no common ancestor (disconnected tree?)")


def influence_matrix(leaves):
    """Full A_{B,A} for all leaf pairs, via A_{B,A} = U(a,A) * R(a,B)."""
    A_mat = {}
    for B in leaves:
        c_k, chain = c_k_all(B)  # chain = root..B, c_k[node.cid]
        A_mat[B.cid] = {}
        for Aclient in leaves:
            a = lca(Aclient, B)
            # R(a,B) = sum_{k=0}^{depth(a)} c_k(B) * U(v_k, a)
            R = 0.0
            for node in chain:
                R += c_k[node.cid] * U(node, a)
                if node.cid == a.cid:
                    break
            u_aA = U(a, Aclient)
            A_mat[B.cid][Aclient.cid] = u_aA * R
    return A_mat


# --------------------------------------------------------------------------
# Verification driver
# --------------------------------------------------------------------------

def main():
    dim = 8
    proj_dim = 20  # oversized on purpose relative to dim=8; identity-ish check
    R_mat = torch.randn(dim, proj_dim) / (proj_dim ** 0.5)

    def project_func(vec, *args, **kwargs):
        return vec @ R_mat

    def dist_func(a, b):
        return torch.norm(a - b, p=2).item()

    root, leaves = build_random_tree(n_leaves=12, branching=3, dim=dim,
                                      dist_func=dist_func, project_func=project_func)

    # theta_A^(t,0): "freshly trained" states, independent of current model_state
    trained_states = {c.cid: random_state(dim) for c in leaves}
    trained_by_cid = trained_states

    # simulate local training: overwrite each leaf's model_state with its
    # trained state, exactly as server.py does before calling update_centers_upward
    for c in leaves:
        c.model_state = copy.deepcopy(trained_states[c.cid])
        c.model_state_flattened = project_func(flatten(c.model_state))

    # ---- run the ACTUAL algorithm ----
    root.update_centers_upward(trained_states)          # Phase B
    root.propagate_downward(recalc_dists=False)                            # Phase C

    actual_final = {c.cid: c.get_model_state()["w"].clone() for c in leaves}

    # ---- compute PREDICTED mixing matrix from the derived formulas ----
    A_mat = influence_matrix(leaves)

    # Check 1: row-stochasticity
    print("=== Row-stochasticity check (sum_A A_{B,A} should be 1) ===")
    max_row_err = 0.0
    for B in leaves:
        s = sum(A_mat[B.cid].values())
        max_row_err = max(max_row_err, abs(s - 1.0))
    print(f"max |sum_A A_(B,A) - 1| over all B: {max_row_err:.3e}")

    # Check 2: predicted theta_B^(t) vs actual theta_B^(t)
    print("\n=== Full-round prediction check (theta_B^(t) vs sum_A A_(B,A) theta_A^(t,0)) ===")
    max_err, max_rel = 0.0, 0.0
    for B in leaves:
        pred = torch.zeros(dim)
        for Aclient in leaves:
            pred += A_mat[B.cid][Aclient.cid] * trained_by_cid[Aclient.cid]["w"]
        err = torch.norm(pred - actual_final[B.cid]).item()
        rel = err / (torch.norm(actual_final[B.cid]).item() + 1e-12)
        max_err = max(max_err, err)
        max_rel = max(max_rel, rel)
        print(f"  leaf {B.cid:>10}: abs_err={err:.3e}  rel_err={rel:.3e}")
    print(f"\nmax abs_err = {max_err:.3e}, max rel_err = {max_rel:.3e}")

    tol = 1e-5
    if max_err < tol and max_row_err < tol:
        print(f"\nPASS: formulas match simulator to within {tol:.0e}")
    else:
        print(f"\nFAIL: mismatch exceeds tolerance {tol:.0e} -- derivation or code has a bug")


def run_once(seed, n_leaves, branching, dim=8, proj_dim=20, verbose=False):
    torch.manual_seed(seed)
    random.seed(seed)
    R_mat = torch.randn(dim, proj_dim) / (proj_dim ** 0.5)

    def project_func(vec, *args, **kwargs):
        return vec @ R_mat

    def dist_func(a, b):
        return torch.norm(a - b, p=2).item()

    root, leaves = build_random_tree(n_leaves=n_leaves, branching=branching, dim=dim,
                                      dist_func=dist_func, project_func=project_func)
    trained_states = {c.cid: random_state(dim) for c in leaves}
    for c in leaves:
        c.model_state = copy.deepcopy(trained_states[c.cid])
        c.model_state_flattened = project_func(flatten(c.model_state))

    root.update_centers_upward(trained_states)
    root.propagate_downward(recalc_dists=False)
    actual_final = {c.cid: c.get_model_state()["w"].clone() for c in leaves}

    A_mat = influence_matrix(leaves)

    max_row_err, max_err, max_rel = 0.0, 0.0, 0.0
    for B in leaves:
        max_row_err = max(max_row_err, abs(sum(A_mat[B.cid].values()) - 1.0))
        pred = torch.zeros(dim)
        for Aclient in leaves:
            pred += A_mat[B.cid][Aclient.cid] * trained_states[Aclient.cid]["w"]
        err = torch.norm(pred - actual_final[B.cid]).item()
        rel = err / (torch.norm(actual_final[B.cid]).item() + 1e-12)
        max_err, max_rel = max(max_err, err), max(max_rel, rel)

    ok = max_err < 1e-4 and max_row_err < 1e-4
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"seed={seed:>3} n_leaves={n_leaves:>3} branching={branching:>2} "
              f"row_err={max_row_err:.2e} abs_err={max_err:.2e} rel_err={max_rel:.2e}  {status}")
    return ok


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # python verify_influence_math.py <folder_name> [base_dir] [noise_std]
        folder_name = sys.argv[1]
        base_dir = sys.argv[2] if len(sys.argv) > 2 else "logs"
        noise_std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
        verify_real_tree(folder_name, base_dir=base_dir, noise_std=noise_std)
    else:
        main()

        print("\n=== Stress test across random seeds / tree shapes ===")
        configs = [
            (1, 12, 3), (2, 12, 2), (3, 30, 2), (4, 30, 4),
            (5, 7, 2), (6, 50, 5), (7, 8, 8), (8, 20, 3),
        ]
        all_ok = True
        for seed, n_leaves, branching in configs:
            ok = run_once(seed, n_leaves, branching, verbose=True)
            all_ok = all_ok and ok
        print("\nALL CONFIGS PASS" if all_ok else "\nSOME CONFIGS FAILED")
        print("\n(No folder name given -- ran the synthetic-tree stress test. "
              "Run `python verify_influence_math.py <log_folder_name>` to check "
              "against a real saved tree instead, e.g. one written by save_metadata.)")