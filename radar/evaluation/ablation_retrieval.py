#!/usr/bin/env python3
"""Four-arm retrieval ablation for the SSC 2026 reference library.

  A  hand         人工手写书签表（v0.1 的那版）
  B  runtime      Hema 方案：运行时按患者概念在图上游走，无主题过滤，N 跳扩展
  C  offline      本项目方案：离线图生成 ∩ 节点主题，冻结
  D  similarity   向量 top-k（此处用 TF-IDF 余弦；打分器可替换为 BGE/E5/MedCPT）

The comparison deliberately does NOT score arms by agreement with C — that would
be circular, since C is our own design. It scores them on properties that are
objectively checkable and clinically consequential:

  1. size            how much text gets injected
  2. branch partners for the three actions the guideline both recommends AND
                     recommends against, does the arm return BOTH sides?
                     Returning only the "for" side of MDR coverage makes the
                     agent over-treat; only the "against" side makes it under-treat.
  3. overrides       does it return stmt_014 (MAP 60-65, age>=65) whenever it
                     returns stmt_013 (MAP 65)?
  4. temporal        does it return the antibiotic-timing statements alongside
                     the blood-culture statement they must follow?
  5. remark survival under the chunking a similarity index normally uses, do the
                     22 statements whose remark carries an independent action
                     still deliver that remark?
"""
from radar import paths
import json, pathlib, re, sys
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

H = pathlib.Path(__file__).parent
ROOT = paths.ROOT
def find(name):
    """Resolve a data file wherever it lives in the repo layout.

    Parsed statements sit in knowledge/parsed, graph artifacts in knowledge/graph.
    """
    for d in (paths.GRAPH, paths.PARSED, paths.FIXTURES):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found under {ROOT}")


G  = json.loads(find("ssc2026_graph.json").read_text(encoding="utf-8"))
SC = json.loads(find("scope_tables.json").read_text(encoding="utf-8"))["scope"]
ST = {s["id"]: s for s in json.loads(find("ssc2026_statements.json").read_text(encoding="utf-8"))}
TR = {r["id"]: r for r in json.loads(find("triples_all.json").read_text(encoding="utf-8"))["statements"]}
GR = json.loads(find("agent1_decision_graph.json").read_text(encoding="utf-8"))
E  = G["edges"]

by_rel = defaultdict(list)
for e in E: by_rel[e["rel"]].append(e)
applies = defaultdict(set)
for e in by_rel["APPLIES_WHEN"]: applies[e["dst"]].add(e["src"])
out_edges = defaultdict(list)
for e in E: out_edges[e["src"]].append(e)
in_edges = defaultdict(list)
for e in E: in_edges[e["dst"]].append(e)
disj_down, disj_up = defaultdict(set), defaultdict(set)
for e in by_rel["IS_DISJUNCTION_OF"]:
    disj_down[e["src"]].add(e["dst"]); disj_up[e["dst"]].add(e["src"])

# ---- the objectively checkable structures -------------------------------
recs = defaultdict(list)
for e in by_rel["RECOMMENDS"]: recs[e["dst"]].append((e["src"], e["polarity"]))
BRANCH_PAIRS = []
for a, lst in recs.items():
    if len({p for _, p in lst}) > 1:
        BRANCH_PAIRS.append((a, sorted({s for s, _ in lst})))
OVERRIDE_PAIRS = [(e["src"], e["dst"]) for e in by_rel["OVERRIDDEN_BY"]]
TEMPORAL = []
for pr in by_rel["MUST_PRECEDE"]:
    before = [s for s, _ in recs[pr["src"]]]
    after  = [s for s, _ in recs[pr["dst"]]]
    TEMPORAL.append((pr["src"], pr["dst"], before, after))
# A remark is "load-bearing" if it does anything other than give context: adds an
# action, restricts who the statement applies to, supplies the numbers, defines the
# terms, or states a preference the body does not. All 46 remarks are now classified.
LOADBEARING = {r["id"] for r in TR.values()
               if any(x["kind"] != "implementation_note" for x in r["refined_by"])}
REMARK_ACTION = {r["id"] for r in TR.values()
                 if any(x.get("adds_action") for x in r["refined_by"])}
REMARK_BEARING_ALL = {sid for sid in ST if ST[sid]["remarks"]}
KIND_OF = {r["id"]: sorted({x["kind"] for x in r["refined_by"]
                            if x["kind"] != "implementation_note"})
           for r in TR.values()}
assert not REMARK_BEARING_ALL - {r["id"] for r in TR.values() if r["refined_by"]}, \
    "a statement with remarks has no refined_by — extraction gap is back"

# ---- arms ---------------------------------------------------------------
HAND = {"R1": ["stmt_001","stmt_002","stmt_003","stmt_004","stmt_005","stmt_006"],
        "R2": ["stmt_004","stmt_005","stmt_008","stmt_009"],
        "R3a":["stmt_005","stmt_006"], "R3b":["stmt_005","stmt_006"],
        "R4": ["stmt_009","stmt_010","stmt_011","stmt_013"]}


def arm_runtime(cond_seeds, hops=2):
    """Hema-style: match concepts, then follow connected relationships outward.

    No topic filter and no frozen table — this is the design described as
    'find matching nodes in the SSC graph, follow connected relationships,
    retrieve the complete clinical pathway'.
    """
    wide = set(cond_seeds)
    for c in list(cond_seeds):
        wide |= disj_down.get(c, set()) | disj_up.get(c, set())
    frontier = set()
    for c in wide:
        frontier |= applies.get(c, set())
    seen = set(frontier)
    for _ in range(hops - 1):
        nxt = set()
        for sid in frontier:
            for e in out_edges[sid]:            # statement -> action/condition/target
                for e2 in in_edges[e["dst"]]:   # ...and back to every other statement
                    if e2["src"].startswith("stmt_"):
                        nxt.add(e2["src"])
        nxt -= seen; seen |= nxt; frontier = nxt
        if not frontier: break
    return sorted(seen, key=lambda s: ST[s]["number"])


_corpus_ids = sorted(ST, key=lambda s: ST[s]["number"])


def _doc(sid, with_remarks=True):
    s = ST[sid]
    return s["statement"] + ((" " + " ".join(s["remarks"])) if with_remarks and s["remarks"] else "")


_vec_full = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
_M_full = _vec_full.fit_transform([_doc(s) for s in _corpus_ids])
# a similarity index normally chunks; simulate the common case where the remark
# becomes its own chunk and can therefore be retrieved without its statement
_chunks, _chunk_owner, _chunk_kind = [], [], []
for sid in _corpus_ids:
    _chunks.append(ST[sid]["statement"]); _chunk_owner.append(sid); _chunk_kind.append("body")
    for i, r in enumerate(ST[sid]["remarks"]):
        _chunks.append(r); _chunk_owner.append(sid); _chunk_kind.append(f"remark{i}")
_vec_ch = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
_M_ch = _vec_ch.fit_transform(_chunks)


def arm_similarity(query, k=8, chunked=False):
    """Top-k similarity. Swap this function for BGE / E5 / MedCPT to get the
    neural numbers; the structural findings below do not depend on the scorer."""
    if chunked:
        sims = cosine_similarity(_vec_ch.transform([query]), _M_ch)[0]
        order = sims.argsort()[::-1][:k]
        return [(_chunk_owner[i], _chunk_kind[i], float(sims[i])) for i in order]
    sims = cosine_similarity(_vec_full.transform([query]), _M_full)[0]
    order = sims.argsort()[::-1][:k]
    return [(_corpus_ids[i], "full", float(sims[i])) for i in order]


# ---- scoring ------------------------------------------------------------
def structural_score(ids):
    ids = set(ids)
    r = {"n": len(ids)}
    r["chars"] = sum(len(_doc(s)) for s in ids if s in ST)
    br_ok = br_half = 0
    for a, members in BRANCH_PAIRS:
        hit = ids & set(members)
        if hit:
            if hit == set(members): br_ok += 1
            else: br_half += 1
    r["branch_complete"], r["branch_half"] = br_ok, br_half
    ov_ok = ov_half = 0
    for gen, spec in OVERRIDE_PAIRS:
        if gen in ids or spec in ids:
            if gen in ids and spec in ids: ov_ok += 1
            else: ov_half += 1
    r["override_complete"], r["override_half"] = ov_ok, ov_half
    tp_ok = tp_half = 0
    for _, _, before, after in TEMPORAL:
        b, a = ids & set(before), ids & set(after)
        if b or a:
            if b and a: tp_ok += 1
            else: tp_half += 1
    r["temporal_complete"], r["temporal_half"] = tp_ok, tp_half
    return r


if __name__ == "__main__":
    print("=" * 96)
    print("四臂对比 · Agent 1 的五个决策节点")
    print("=" * 96)
    rows = []
    for nid, node in GR["nodes"].items():
        seeds = node["scope_seeds"]["condition"]
        q = node["question"]
        arms = {
            "A 手写":       [s for s in HAND[nid]],
            "B 运行时游走": arm_runtime(seeds, hops=2),
            "C 离线∩主题":  [s for s in node["scope"] if s.startswith("stmt_")],
            "D 相似度 k=8": [s for s, _, _ in arm_similarity(q, k=8)],
        }
        print(f"\n[{nid}] {node['title']}")
        for name, ids in arms.items():
            sc = structural_score(ids)
            rows.append((nid, name, sc))
            print(f"   {name:<14}{sc['n']:>4} 条   "
                  f"分支对 完整{sc['branch_complete']}/半{sc['branch_half']}   "
                  f"覆盖对 完整{sc['override_complete']}/半{sc['override_half']}   "
                  f"时序对 完整{sc['temporal_complete']}/半{sc['temporal_half']}")

    print("\n" + "=" * 96)
    print("汇总（五个节点合计）")
    print("=" * 96)
    agg = defaultdict(lambda: defaultdict(int))
    for _, name, sc in rows:
        for k, v in sc.items(): agg[name][k] += v
    hdr = ("臂", "注入条数", "注入字符", "分支完整", "分支半", "覆盖完整", "覆盖半", "时序完整", "时序半")
    print(f"{hdr[0]:<14}" + "".join(f"{h:>9}" for h in hdr[1:]))
    for name in ("A 手写", "B 运行时游走", "C 离线∩主题", "D 相似度 k=8"):
        a = agg[name]
        print(f"{name:<14}{a['n']:>9}{a['chars']:>9}{a['branch_complete']:>9}{a['branch_half']:>9}"
              f"{a['override_complete']:>9}{a['override_half']:>9}"
              f"{a['temporal_complete']:>9}{a['temporal_half']:>9}")
    orc = structural_score(list(ST))
    print(f"{'全给（天花板）':<12}{orc['n']*5:>9}{orc['chars']*5:>9}"
          f"{orc['branch_complete']:>9}{orc['branch_half']:>9}{orc['override_complete']:>9}"
          f"{orc['override_half']:>9}{orc['temporal_complete']:>9}{orc['temporal_half']:>9}"
          "   ← 每个节点都把 129 条全塞进去")
    print("""
  这三列怎么读（重要，别读反了）：
  · B 的「完整」列全满，不是因为它聪明，是因为它几乎把整本指南拉了进来。
    对比最后一行「全给」：B 每节点 71-106 条，天花板是 129 条 —— B ≈ 全给。
    在一个「返回全部就满分」的指标上得满分，说明不了检索质量，只说明它没在筛。
  · A 和 C 的「完整」数低，主要是因为 Agent 1 的五个节点都是诊断性问题
    （有没有感染 / 有没有器官功能障碍 / 有没有休克），本来就不涉及抗生素、
    碳酸氢钠这类有正反两面的治疗动作。分支对这一项要到 Agent 2 才真正吃劲。
  · 因此在 Agent 1 这一层，唯一非空的结构检验是 R4 的覆盖对 stmt_013 / stmt_014
    （MAP 65 vs 老年人 60-65）—— 见下。""")

    print("\n" + "=" * 96)
    print("R4 覆盖对：stmt_013（MAP≥65）与 stmt_014（≥65 岁可放宽到 60-65）")
    print("=" * 96)
    for name, ids in (("A 手写", HAND["R4"]),
                      ("B 运行时游走", arm_runtime(GR["nodes"]["R4"]["scope_seeds"]["condition"], 2)),
                      ("C 离线∩主题", [s for s in GR["nodes"]["R4"]["scope"] if s.startswith("stmt_")]),
                      ("D 相似度 k=8", [s for s, _, _ in arm_similarity(GR["nodes"]["R4"]["question"], 8)])):
        s = set(ids)
        got = ("两条都给" if {"stmt_013", "stmt_014"} <= s else
               "只给 stmt_013 → 对 ≥65 岁患者会按 MAP 65 过度升压" if "stmt_013" in s else
               "只给 stmt_014" if "stmt_014" in s else "两条都没给")
        print(f"   {name:<14}{len(s):>4} 条   {got}")

    print("\n" + "=" * 96)
    print("三对「同一动作、方向相反」在相似度检索下的排名距离")
    print("=" * 96)
    print("若两条排名相邻，top-k 有机会都取回；相隔越远越不可能。")
    for a, members in BRANCH_PAIRS:
        q = ST[members[0]]["statement"]
        sims = cosine_similarity(_vec_full.transform([q]), _M_full)[0]
        rank = {sid: i for i, sid in enumerate(
            [_corpus_ids[j] for j in sims.argsort()[::-1]])}
        rs = {m: rank[m] for m in members}
        print(f"\n  {a}")
        for m in members:
            pol = [p for s, p in recs[a] if s == m][0]
            print(f"     {m} [{pol:<7}] 排名 {rs[m]:>3}   {ST[m]['statement'][:74]}")
        print(f"     → 排名相距 {max(rs.values()) - min(rs.values())} 位"
              f"{'（k=8 取不全）' if max(rs.values()) >= 8 else ''}")

    print("\n" + "=" * 96)
    print("分块检索下 remark 的存活情况")
    print("=" * 96)
    from collections import Counter as _C
    kc = _C(x["kind"] for r in TR.values() for x in r["refined_by"])
    nrem = sum(len(ST[s]["remarks"]) for s in ST)
    print(f"129 条里 {len(REMARK_BEARING_ALL)} 条带 remark，共 {nrem} 条 remark，全部已分类：")
    for k, v in kc.most_common():
        tag = "（纯背景，丢了不影响处置）" if k == "implementation_note" else "（丢了会改变处置）"
        print(f"     {k:<22}{v:>4}  {tag}")
    print(f"\n  会改变处置的 remark 分布在 {len(LOADBEARING)} 条条款上。")
    print(f"  下面用条文正文本身做查询、k=5 分块检索，看这 {len(LOADBEARING)} 条的 remark 还在不在。\n")
    lost = kept = 0
    for sid in sorted(LOADBEARING, key=lambda s: ST[s]["number"]):
        q = ST[sid]["statement"]
        got = arm_similarity(q, k=5, chunked=True)
        has_remark = any(k.startswith("remark") for o, k, _ in got if o == sid)
        if has_remark:
            kept += 1
        else:
            lost += 1
            print(f"  ✗ {sid}  [{'/'.join(KIND_OF[sid])}]  正文命中，remark 未进 top-5")
            print(f"      丢掉的是：{ST[sid]['remarks'][0][:96]}…")
    print(f"\n  {len(LOADBEARING)} 条中：remark 一并取回 {kept} 条，remark 丢失 {lost} 条"
          f"（{lost/len(LOADBEARING):.0%}）。")

    print("\n  （2026-08-07：第 65-129 条此前 refined_by 全空，remark 统计只覆盖了前半本。"
          "\n   已补齐，上面 46 条 remark 是全量。）")
