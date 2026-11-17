"""
Self-check: asserts that core logic works correctly.
Fails loud -- no test framework, just asserts + a final print.
One runnable check that breaks if the logic breaks.
"""
import os
import time
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


def check_chain_reload():
    c = InProcessChain(seed="reload")
    c.add_record("Hospital_A", "h1", {})
    c.grant_access("Hospital_A", "Hospital_B", "h1")
    c.log_event("ping", "coordinator", {})
    c2 = InProcessChain.from_blocks(c.to_dict())
    assert c2.verify_chain() and len(c2) == len(c)
    assert c2.has_access("Hospital_B", "h1") and not c2.has_access("Hospital_C", "h1")
    c2.chain[1].data["record_hash"] = "x"
    assert not c2.verify_chain(), "tampered reloaded chain must fail"
    print("  [OK] reload     -- chain_state round-trip / verify / tamper")


def check_integrity():
    import json
    import shutil
    import tempfile

    import integrity

    saved = (integrity.DATA_DIR, integrity.CHAIN_PATH, integrity.FL_STATE_PATH)
    tmp = tempfile.mkdtemp()
    try:
        integrity.DATA_DIR = os.path.join(tmp, "hospital_data")
        integrity.CHAIN_PATH = os.path.join(tmp, "chain_state.json")
        integrity.FL_STATE_PATH = os.path.join(tmp, "fl_state.json")
        hs = integrity.HOSPITALS
        df, _ = load_dataset("pima")
        shards = {h: df.iloc[i * 50:(i + 1) * 50].reset_index(drop=True) for i, h in enumerate(hs)}
        hashes = integrity.write_hospital_files(shards, "pima", "iid")
        assert integrity.read_meta() == {"dataset": "pima", "split": "iid"}

        # register like run_simulation does, then write the ledger
        chain = InProcessChain()
        for h in hs:
            node = HospitalNode(h, shards[h], chain)
            _, rec = node.register_data()
            assert rec == hashes[h], "file hash must equal registered hash"
        integrity._write_json(integrity.CHAIN_PATH, chain.to_dict())
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "complete", "chain_length": len(chain), "chain_valid": True})

        def status(h):
            return next(r for r in integrity.scan() if r["hospital"] == h)

        assert [r["status"] for r in integrity.scan()] == ["match"] * 3
        A = "Hospital_A"
        # saving the editor with no edits keeps the hash
        assert integrity.save_shard(A, integrity.load_shard(A)) == hashes[A]
        assert status(A)["status"] == "match"

        # edit one cell
        edited = integrity.load_shard(A)
        old_val = edited.iloc[3, 1]
        edited.iloc[3, 1] = old_val + 7
        integrity.save_shard(A, edited)
        r = status(A)
        assert r["status"] == "changed" and r["live_hash"] != r["chain_hash"] and r["values_identical"] is False
        cc = integrity.changed_cells(A)
        assert cc["modified"] == [{"row": 3, "column": edited.columns[1], "old": str(old_val), "new": str(old_val + 7)}], cc
        assert not cc["values_identical"] and cc["rows_added"] == 0 and cc["rows_removed"] == 0

        # undo, then re-save with Windows line endings (Excel-style): bytes differ, values do not
        integrity.restore_original(A)
        assert status(A)["status"] == "match"
        with open(integrity._path(A), "rb") as f:
            raw = f.read()
        with open(integrity._path(A), "wb") as f:
            f.write(raw.replace(b"\n", b"\r\n"))
        r = status(A)
        assert r["status"] == "changed" and r["values_identical"] is True
        assert integrity.changed_cells(A)["modified"] == []
        integrity.restore_original(A)

        # record a new version: signed by the hospital, chain still verifies after reload
        integrity.save_shard(A, edited)
        blk = integrity.record_new_version(A)
        assert status(A)["status"] == "match"
        c2 = InProcessChain.from_blocks(json.load(open(integrity.CHAIN_PATH)))
        assert c2.verify_chain() and len(c2) == len(chain) + 1
        last = c2.chain[-1]
        assert last.signer_id == A and last.data["event"] == "data_update"
        assert last.data["previous_record_hash"] == hashes[A] and blk["index"] == last.index
        assert last.public_key == KP.generate(A, seed="demo-seed").public_bytes_hex()
        assert json.load(open(integrity.FL_STATE_PATH))["chain_length"] == len(c2)
        assert integrity.chain_record(A)["event"] == "data_update"

        # restore_original returns the snapshot (now the recorded version)
        integrity.save_shard(A, shards[A])
        assert status(A)["status"] == "changed"
        assert integrity.restore_original(A) == status(A)["chain_hash"]

        # missing / unreadable never raise
        B = "Hospital_B"
        os.remove(integrity._path(B))
        assert status(B)["status"] == "missing"
        with open(integrity._path(B), "wb") as f:
            f.write(b"\xff\xfe\x00garbage\x80")
        assert status(B)["status"] == "unreadable"
        try:
            integrity.record_new_version(B)
            raise AssertionError("unreadable file must not be recorded")
        except RuntimeError:
            pass
        integrity.restore_original(B)

        # refuse while a simulation is running; ledger untouched
        before = open(integrity.CHAIN_PATH).read()
        integrity.save_shard(A, shards[A])
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "training"})
        try:
            integrity.record_new_version(A)
            raise AssertionError("must refuse while training")
        except RuntimeError as exc:
            assert "simulation is running" in str(exc)
        assert open(integrity.CHAIN_PATH).read() == before
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "complete"})

        # a stale "training" status (crashed run, file untouched for >60 s) does not block recording
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "training"})
        old_t = time.time() - 120
        os.utime(integrity.FL_STATE_PATH, (old_t, old_t))
        integrity.record_new_version(A)
        assert status(A)["status"] == "match"
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "complete"})
        # recording an unchanged file is refused (no duplicate blocks)
        try:
            integrity.record_new_version(A)
            raise AssertionError("no-op record must be refused")
        except RuntimeError as exc:
            assert "nothing new" in str(exc)

        # a forged (unsigned) record block appended to chain_state.json is never trusted
        good = json.load(open(integrity.CHAIN_PATH))
        forged_file = shards[A].copy()
        forged_file.iloc[0, 1] += 99
        forged_hash = integrity.save_shard(A, forged_file)
        fake = dict(good[-1], index=len(good), previous_hash=good[-1]["hash"], hash="f" * 64,
                    data={"event": "data_update", "hospital": A, "record_hash": forged_hash})
        integrity._write_json(integrity.CHAIN_PATH, good + [fake])
        assert [r["status"] for r in integrity.scan()] == ["chain_invalid"] * 3
        integrity._write_json(integrity.CHAIN_PATH, good)
        integrity.restore_original(A)

        # added columns are reported, not silently "same values"
        wide = integrity.load_shard(A)
        wide["Extra"] = 1
        integrity.save_shard(A, wide)
        assert integrity.changed_cells(A)["columns_added"] == ["Extra"]
        integrity.restore_original(A)
        # undo without a registered copy is a readable error
        os.remove(integrity._snap(B))
        try:
            integrity.restore_original(B)
            raise AssertionError("restore without snapshot must raise RuntimeError")
        except RuntimeError:
            pass

        # old-format chain (truncated keys only) cannot be extended
        old = [{k: v for k, v in b.items() if not k.endswith("_full")} for b in json.load(open(integrity.CHAIN_PATH))]
        integrity._write_json(integrity.CHAIN_PATH, old)
        try:
            integrity.record_new_version(A)
            raise AssertionError("must refuse chain without full keys")
        except RuntimeError as exc:
            assert "Run the simulation again" in str(exc)
        assert json.load(open(integrity.CHAIN_PATH)) == old
    finally:
        integrity.DATA_DIR, integrity.CHAIN_PATH, integrity.FL_STATE_PATH = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [OK] integrity  -- scan / edit / CRLF / new version / undo / missing / running / old chain")



def check_consent_ops():
    """Consent tab operations on the stored ledger: grant, revoke, request, matrix."""
    import json
    import shutil
    import tempfile

    import integrity

    saved = (integrity.DATA_DIR, integrity.CHAIN_PATH, integrity.FL_STATE_PATH)
    tmp = tempfile.mkdtemp()
    try:
        integrity.DATA_DIR = os.path.join(tmp, "hospital_data")
        integrity.CHAIN_PATH = os.path.join(tmp, "chain_state.json")
        integrity.FL_STATE_PATH = os.path.join(tmp, "fl_state.json")
        hs = integrity.HOSPITALS
        A, B, C = hs
        df, _ = load_dataset("pima")
        shards = {h: df.iloc[i * 40:(i + 1) * 40].reset_index(drop=True) for i, h in enumerate(hs)}
        integrity.write_hospital_files(shards, "pima", "iid")
        chain = InProcessChain()
        for h in hs:
            HospitalNode(h, shards[h], chain).register_data()
        integrity._write_json(integrity.CHAIN_PATH, chain.to_dict())
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "complete"})

        def access(req, owner):
            return integrity.consent_matrix()["access"][req][owner]

        def ledger():
            c = InProcessChain.from_blocks(json.load(open(integrity.CHAIN_PATH)))
            assert c.verify_chain()
            return c

        assert not access(C, A) and not access(B, A)
        blk = integrity.grant_consent(A, C)  # owner grants -> signed by the owner
        assert blk["signer_id"] == A and blk["data"]["event"] == "grant_access" and access(C, A)
        assert not access(B, A), "a grant to C must not open access for B"

        r = integrity.request_access(C, A)  # every request is logged, signed by the requester
        assert r["allowed"] is True
        last = ledger().chain[-1]
        assert last.data["event"] == "access_request" and last.signer_id == C
        assert last.data["details"]["allowed"] is True and last.data["details"]["owner"] == A
        r = integrity.request_access(B, A)
        assert r["allowed"] is False and "no consent" in r["reason"]
        assert ledger().chain[-1].data["details"]["allowed"] is False

        integrity.revoke_consent(A, C)  # revoke ends access
        assert not access(C, A) and ledger().chain[-1].data["event"] == "revoke_access"
        for bad in (lambda: integrity.revoke_consent(A, C),   # nothing left to revoke
                    lambda: integrity.grant_consent(A, A),    # own data
                    lambda: integrity.grant_consent("Hospital_X", A)):
            try:
                bad()
                raise AssertionError("must be refused")
            except RuntimeError:
                pass
        integrity.grant_consent(A, C)  # grant again after a revoke
        assert access(C, A)

        # consent is for a specific version: a new recorded version needs a new grant
        edited = integrity.load_shard(A)
        edited.iloc[0, 1] += 1
        integrity.save_shard(A, edited)
        integrity.record_new_version(A)
        assert not access(C, A)

        # refused while a simulation is running; ledger untouched
        before = open(integrity.CHAIN_PATH).read()
        integrity._write_json(integrity.FL_STATE_PATH, {"status": "training"})
        try:
            integrity.grant_consent(B, C)
            raise AssertionError("must refuse while training")
        except RuntimeError as exc:
            assert "simulation is running" in str(exc)
        assert open(integrity.CHAIN_PATH).read() == before
    finally:
        integrity.DATA_DIR, integrity.CHAIN_PATH, integrity.FL_STATE_PATH = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [OK] consent ops -- grant / request log / revoke / owner-only / per-version / running")


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


def check_chain_security():
    from chain.inprocess import Block

    def key(n):
        return KP.generate(n, seed="demo-seed")

    A, B = key("Hospital_A"), key("Hospital_B")

    def fresh():
        c = InProcessChain()
        c.add_record("Hospital_A", "hA", {}, signer=A)
        c.log_event("ping", "coordinator", {})
        c.log_event("pong", "coordinator", {})
        assert c.verify_chain()
        return c

    # 1. rewrite history and re-sign everything with a non-member key
    c = fresh()
    evil = KP.generate("coordinator", seed="attacker")
    c.chain[1].data["record_hash"] = "forged"
    for i in range(1, len(c.chain)):
        blk = c.chain[i]
        blk.previous_hash = c.chain[i - 1].hash
        blk.sign_with(evil)
    assert not c.verify_chain(), "re-signed chain with non-member key must fail"

    # 2. relabel last block's signer_id and recompute hash
    c = fresh()
    c.chain[-1].signer_id = "Hospital_B"
    c.chain[-1].hash = c.chain[-1]._compute_hash()
    assert not c.verify_chain(), "signer relabel must fail"

    # 3. genesis tamper (with hash recomputed and relinked)
    c = fresh()
    c.chain[0].data["description"] = "evil"
    c.chain[0].hash = c.chain[0]._compute_hash()
    c.chain[1].previous_hash = c.chain[0].hash
    assert not c.verify_chain(), "genesis tamper must fail"

    # 4. index swapped
    c = fresh()
    c.chain[1], c.chain[2] = c.chain[2], c.chain[1]
    assert not c.verify_chain(), "swapped blocks must fail"

    # 5. replica / peer rejects unknown-key block; malformed grant never raises
    auth = fresh()
    rep = ChainReplica("Hospital_A", keys=A)
    assert rep.sync_from(auth)["last_sync_ok"]
    h = rep.local.chain[-1].hash
    forged = Block(len(rep.local), {"event": "grant_access", "from": "Hospital_A", "to": "Hospital_C", "record_hash": "hA"}, h)
    forged.sign_with(evil)
    assert not rep.try_accept_block(forged), "unknown signer must be rejected"
    nokey = Block(len(rep.local), {"event": "grant_access", "from": "Hospital_A", "record_hash": "hA"}, h)
    nokey.sign_with(A)
    assert rep.try_accept_block(nokey) in (True, False)  # must not raise

    # 6. consent is bound to the record owner
    c = InProcessChain()
    c.add_record("Hospital_A", "hA", {}, signer=A)
    try:
        c.grant_access("Hospital_B", "Hospital_C", "hA", signer=B)
        raise AssertionError("non-owner grant must raise")
    except (PermissionError, ValueError):
        pass
    try:
        c.grant_access("Hospital_A", "Hospital_C", "hA", signer=B)  # B cannot sign for A
        raise AssertionError("grant signed by someone else must raise")
    except (PermissionError, ValueError):
        pass
    assert not c.has_access("Hospital_C", "hA")
    c.grant_access("Hospital_A", "Hospital_B", "hA", signer=A)
    assert c.has_access("Hospital_B", "hA") and not c.has_access("Hospital_C", "hA")
    # forged grant block appended directly (B pretends A's record is its own to share)
    fb = Block(len(c.chain), {"event": "grant_access", "from": "Hospital_B", "to": "Hospital_C", "record_hash": "hA"}, c.chain[-1].hash)
    fb.sign_with(B)
    c.chain.append(fb)
    assert not c.has_access("Hospital_C", "hA"), "grant by non-owner must not count"
    c.chain.pop()
    # tamper the grant block -> access disappears
    c.chain[-1].data["to"] = "Hospital_C"
    assert not c.has_access("Hospital_B", "hA") and not c.has_access("Hospital_C", "hA")
    print("  [OK] security   -- membership / re-sign / relabel / genesis / index / replica / owner-bound consent")


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
    chain.add_record("Hospital_A", ref, {})  # only the registered owner can grant
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
    # a near-empty positive class makes "predict nobody dies" score 100% — meaningless accuracy
    pos_rate = mdf["Outcome"].mean()
    assert 0.08 <= pos_rate <= 0.30, f"mimic positive rate {pos_rate:.3f} is unrealistic"
    _, mtest, _ = split_for_hospitals(mdf, ["A", "B", "C"], mode="iid")
    assert mtest["Outcome"].sum() >= 3, "mimic test set needs positive cases to measure anything"
    print(f"  [OK] dataset    -- pima {len(df)} rows; mimic synthetic {len(mdf)} rows ({pos_rate:.0%} positive)")


def check_crypto():
    a = KP.generate("A", seed="x")
    b = KP.generate("A", seed="x")
    assert a.public_bytes_hex() == b.public_bytes_hex(), "seeded keys must be deterministic"
    msg = b"hello-block"
    sig = a.sign(msg)
    assert KP.verify(a.public_bytes_hex(), msg, sig)
    assert not KP.verify(a.public_bytes_hex(), b"other", sig)
    print("  [OK] crypto     -- ed25519 sign / verify / deterministic seed")


def check_federated_scaler():
    from fl_client import HospitalClient
    from fl_server import FedAvgServer, combine_stats

    rng = np.random.default_rng(0)
    shards = [
        pd.DataFrame(np.c_[rng.normal(m, sd, (n, 3)), rng.integers(0, 2, n)])
        for m, sd, n in [(0, 1, 50), (5, 3, 80), (-2, 0.5, 30)]
    ]
    clients = [HospitalClient(f"H{i}", sh) for i, sh in enumerate(shards)]
    mean, std = combine_stats([c.local_feature_stats() for c in clients])
    allX = np.vstack([sh.iloc[:, :-1].values for sh in shards])
    assert np.allclose(mean, allX.mean(axis=0)), "pooled mean must equal mean of concatenated data"
    assert np.allclose(std, allX.std(axis=0)), "pooled std must equal std of concatenated data"

    # Server evaluation transforms the test set with the GLOBAL scaler (nothing fit on test)
    test = pd.DataFrame(np.c_[rng.normal(100, 7, (20, 3)), rng.integers(0, 2, 20)])
    srv = FedAvgServer(test, mean, std)
    assert np.allclose(srv.X_test, (test.iloc[:, :-1].values - mean) / std)
    assert abs(srv.X_test.mean()) > 1, "test set must not be re-standardised on itself"
    for c in clients:
        c.set_scaler(mean, std)
    assert np.allclose(np.vstack([c.X for c in clients]), (allX - mean) / std)
    print("  [OK] fed scaler -- pooled mean/std == concatenated; server eval uses global scaler")


def check_poison_traceability():
    import run_experiments as rx

    df, _ = load_dataset("pima")
    shards, test_df, _ = split_for_hospitals(df, rx.HOSPITAL_NAMES, mode="iid")

    # weight_poison: hash on chain == hash of the (poisoned) update that was aggregated
    audit = {}
    rx.run_fedavg(shards, test_df, 2, poison="weight_poison", audit=audit)
    chain_hashes = {
        (b.data["actor"], b.data["details"]["round"]): b.data["details"]["update_hash"]
        for b in audit["chain"].chain
        if b.data.get("event") == "fl_update_hash"
    }
    assert audit["aggregated"], "no updates aggregated"
    for a in audit["aggregated"]:
        assert chain_hashes[(a["hospital"], a["round"])] == a["hash"], "on-chain hash != aggregated update"

    # transit_tamper: altered after commit -> server detects and rejects it
    clean = rx.run_fedavg(shards, test_df, 2)
    t = rx.run_fedavg(shards, test_df, 2, poison="transit_tamper")
    assert t["tamper_detected"] and len(t["updates_rejected"]) == 2
    assert all(r["hospital"] == "Hospital_C" for r in t["updates_rejected"])
    assert t["accuracy"] > 0.5 and clean["auc"] is not None
    print("  [OK] poison     -- poisoned update hash == chain hash; transit tamper rejected")


def check_heart_centres():
    df, meta = load_dataset("heart")
    counts = pd.read_csv(meta["path"])["center"].value_counts()
    assert set(counts.index) == {"cleveland", "hungarian", "switzerland", "va"}, counts
    assert (counts > 100).all(), f"every centre must keep most of its rows: {counts.to_dict()}"
    assert not df.isna().any().any()
    print(f"  [OK] heart      -- all 4 centres present {counts.to_dict()}")


if __name__ == "__main__":
    print("Running self-check...\n")
    check_crypto()
    check_blockchain()
    check_chain_reload()
    check_replicas()
    check_integrity()
    check_chain_security()
    check_fedavg()
    check_consent_and_split()
    check_data()
    check_federated_scaler()
    check_poison_traceability()
    check_heart_centres()
    check_consent_ops()
    print("\nALL CHECKS PASSED")
