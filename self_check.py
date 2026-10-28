"""
Self-check: asserts that core logic works correctly.
Fails loud -- no test framework, just asserts + a final print.
One runnable check that breaks if the logic breaks.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chain import ChainReplica, FabricGateway, InProcessChain, NodeKeyPair, get_chain
from chain.crypto import NodeKeyPair as KP
from data_split import split_for_hospitals
from datasets import load_dataset
from fl_server import fedavg
from hospital_node import ConsentError, HospitalNode, hash_weights


def check_blockchain():
    chain = get_chain("inprocess")
    assert isinstance(chain, InProcessChain)
    assert len(chain) == 1, "Genesis block missing"

    chain.add_record("Hospital_A", "abc123", {"n_records": 100})
    chain.log_event("fl_round", "coordinator", {"round": 1})
    chain.grant_access("Hospital_A", "Hospital_B", "abc123")

    assert len(chain) == 4, f"Expected 4 blocks, got {len(chain)}"
    assert chain.verify_chain(), "Valid chain should verify"
    assert chain.has_access("Hospital_B", "abc123")
    assert not chain.has_access("Hospital_C", "abc123")

    # Signatures present on non-genesis blocks
    for blk in chain.chain[1:]:
        assert blk.signature, "Block must be signed"
        assert blk.signature_valid(), "Block signature must verify"

    # Tamper a block's data in-place -- hash should no longer match
    chain.chain[2].data["event"] = "tampered"
    assert not chain.verify_chain(), "Tampered chain must be detected as invalid"

    # Fabric mock mode must work without a live network
    fabric = get_chain("fabric")
    assert isinstance(fabric, FabricGateway)
    fabric.add_record("H1", "hash1", {})
    assert fabric.verify_chain()
    assert len(fabric) >= 2

    print("  [OK] blockchain -- genesis / signatures / consent / tamper / fabric-mock")


def check_replicas():
    auth = InProcessChain(seed="replica-test")
    auth.add_record("Hospital_A", "r1", {})
    auth.log_event("ping", "coordinator", {})

    keys = NodeKeyPair.generate("Hospital_A", seed="replica-test")
    rep = ChainReplica("Hospital_A", keys=keys)
    status = rep.sync_from(auth)
    assert status["last_sync_ok"], f"replica sync failed: {status}"
    assert status["synced_height"] == len(auth)

    # Reject unsigned garbage block
    from chain.inprocess import Block

    bad = Block(len(rep.local), {"event": "evil"}, rep.local.chain[-1].hash)
    # leave unsigned
    assert not rep.try_accept_block(bad), "unsigned peer block must be rejected"
    print("  [OK] replicas   -- sync / reject unsigned peer block")


def check_fedavg():
    w1 = {"coef": np.array([[1.0, 2.0]]), "intercept": np.array([0.5])}
    w2 = {"coef": np.array([[3.0, 4.0]]), "intercept": np.array([1.5])}

    result = fedavg([(w1, 100), (w2, 100)])
    assert np.allclose(result["coef"], [[2.0, 3.0]]), f"Expected [[2,3]], got {result['coef']}"
    assert np.allclose(result["intercept"], [1.0])

    result = fedavg([(w1, 100), (w2, 300)])
    assert np.allclose(result["coef"][0][0], 1.0 * 0.25 + 3.0 * 0.75)

    h = hash_weights(w1)
    assert len(h) == 64
    print("  [OK] fedavg     -- equal weights / weighted average / update hash")


def check_consent_and_split():
    df, _ = load_dataset("pima")
    names = ["Hospital_A", "Hospital_B", "Hospital_C"]
    shards_iid, _, info_iid = split_for_hospitals(df, names, mode="iid")
    shards_ni, _, info_ni = split_for_hospitals(df, names, mode="noniid")
    assert info_iid["mode"] == "iid"
    assert info_ni["mode"] == "noniid"
    assert sum(len(s) for s in shards_iid) == info_iid["n_train"]

    chain = InProcessChain(seed="consent")
    node_b = HospitalNode("Hospital_B", shards_iid[1], chain)
    node_b.register_data()
    ref = "secret_record_ref"
    try:
        node_b.use_record_ref(ref)
        raise AssertionError("consent gate should deny without grant")
    except ConsentError:
        pass
    chain.grant_access("Hospital_A", "Hospital_B", ref)
    assert node_b.use_record_ref(ref) is True
    print("  [OK] consent    -- grant gate + iid/noniid splits")


def check_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "diabetes.csv")
    df, meta = load_dataset("pima")
    assert os.path.exists(data_path) or meta["n_rows"] >= 500
    assert len(df) >= 500, f"Expected >=500 rows, got {len(df)}"
    assert df.shape[1] >= 9, f"Expected >=9 columns, got {df.shape[1]}"
    label_vals = set(df.iloc[:, -1].unique())
    assert label_vals <= {0, 1}, f"Last column should be binary, got values {label_vals}"

    # MIMIC stub must not require PhysioNet
    mdf, mmeta = load_dataset("mimic")
    assert mmeta.get("is_synthetic"), "mimic loader should use synthetic stub"
    assert len(mdf) >= 50
    print(f"  [OK] dataset    -- pima {len(df)} rows; mimic synthetic {len(mdf)} rows")


def check_crypto():
    a = KP.generate("A", seed="x")
    b = KP.generate("A", seed="x")
    assert a.public_bytes_hex() == b.public_bytes_hex(), "seeded keys must be deterministic"
    msg = b"hello-block"
    sig = a.sign(msg)
    assert KP.verify(a.public_bytes_hex(), msg, sig)
    assert not KP.verify(a.public_bytes_hex(), b"other", sig)
    print("  [OK] crypto     -- ed25519 sign / verify / deterministic seed")


if __name__ == "__main__":
    print("Running self-check...\n")
    check_crypto()
    check_blockchain()
    check_replicas()
    check_fedavg()
    check_consent_and_split()
    check_data()
    print("\nALL CHECKS PASSED")
