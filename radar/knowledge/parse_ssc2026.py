#!/usr/bin/env python3
"""Parse the 129 recommendation statements out of SSC 2026 (CCM version, pp. 725-812)."""
from radar import paths
import pypdf, re, json, os, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
OUT = paths.PARSED
OUT.mkdir(exist_ok=True)

# The CCM print version, pages 725-812. Override with SSC2026_PDF if the file
# is kept elsewhere; the page-offset logic assumes this exact pagination.
PDF = os.environ.get(
    "SSC2026_PDF",
    str(paths.SOURCES / "SSC2026" / "SSC2026_full_CCM.pdf"))
if not pathlib.Path(PDF).exists():
    raise SystemExit(
        f"SSC 2026 PDF not found at {PDF}\n"
        "  set SSC2026_PDF=/path/to/the/CCM/pdf and rerun")
JOURNAL_P1 = 725

# ---------- 1. extract, cleaning PER PAGE so offsets stay aligned to pages ----------
def clean(s):
    # soft (layout) hyphen in this PDF is "word -\n"; a real compound hyphen is "word-\n"
    s = re.sub(r'(?<=[A-Za-z]) +-\s*\n\s*(?=[a-z])', '', s)
    s = re.sub(r'(?<=[A-Za-z])-\n\s*(?=[a-z])', '-', s)
    s = s.replace('­', '')
    for junk in (r'Copyright © 2026 by the Society of Critical Care Medicine[^\n]*',
                 r'Prescott et al\s*\d*', r'www\.ccmjournal\.org',
                 r'April 2026 • Volume 54 • Number 4', r'Critical Care Medicine'):
        s = re.sub(junk, ' ', s)
    return s

reader = pypdf.PdfReader(PDF)
parts, ends, total = [], [], 0
for pg in reader.pages:
    c = clean(pg.extract_text() or "") + "\n"
    parts.append(c)
    total += len(c)
    ends.append(total)
txt = "".join(parts)

def page_at(i):
    for k, e in enumerate(ends):
        if i < e:
            return k + 1
    return len(ends)

norm = lambda s: re.sub(r'\s+', ' ', s).strip()

# ---------- 3. anchors (whitespace-tolerant throughout) ----------
GRADE = re.compile(
    r'\(\s*(strong|conditional)\s+recommendation\s*'
    r'(?:[,;]\s*(very\s+low|low|moderate|high)[\s\-]+certainty(?:\s+of)?\s+evidence[^)]{0,40})?\s*\)',
    re.I)
GPS   = re.compile(r'\(\s*good\s+practice\s+statement\s*\)', re.I)
INSUF = re.compile(r'there\s+is\s+[“"]?\s*insufficient\s+evidence\s*[”"]?\s+'
                   r'to\s+(?:make|issue)\s+a\s+recommendation', re.I)

anchors = ([(m.start(), m.end(), 'graded', m) for m in GRADE.finditer(txt)] +
           [(m.start(), m.end(), 'gps',    m) for m in GPS.finditer(txt)] +
           [(m.start(), m.end(), 'insuf',  m) for m in INSUF.finditer(txt)])
anchors.sort()

NUM    = re.compile(r'(?:^|\n)[ \t]*(\d{1,3})[ \t]*\.[ \t]+(?=[A-Z“"])')
TYPE   = re.compile(r'\b(Carryover with new remark|Carryover|Revisited|Revised|New)\b')
REMARK = re.compile(r'Remarks?\s*:\s*(.+?)'
                    r'(?=\s*Remarks?\s*:|\s*(?:Carryover|Revisited|Revised|New)\b|$)', re.S)
AGAINST = re.compile(r'[“"]\s*(?:recommend|suggest)\s+against\s*[”"]', re.I)

records, flags, offsets, gaps = {}, {}, {}, {}
for start, end, kind, m in anchors:
    bs = max(0, start - 1500)
    # +2 so the NUM lookahead has material even when the statement starts at `start`
    nums = [m for m in NUM.finditer(txt[bs:start + 2]) if m.end() <= start - bs]
    if not nums:
        continue
    n = int(nums[-1].group(1))
    if not 1 <= n <= 129:
        continue
    body_from = bs + nums[-1].end()

    if kind == 'insuf':
        seg = norm(txt[body_from:end + 400])
        cut = re.search(r'\.(?=\s*(?:Remark|Carryover|Revisited|Revised|New|[A-Z][a-z]))', seg)
        stmt = seg[:cut.end()] if cut else seg.split('. ')[0] + '.'
        strength, certainty = 'insufficient_evidence', None
        tail = norm(txt[body_from:body_from + len(stmt) + 1100])[len(stmt) - 30:]
    else:
        stmt = norm(txt[body_from:start])
        if kind == 'gps':
            strength, certainty = 'good_practice', None
        else:
            direction = 'against' if AGAINST.search(stmt) else 'for'
            strength = f"{m.group(1).lower()}_{direction}"
            certainty = re.sub(r'\s+', '_', m.group(2).lower()) if m.group(2) else None
        tail = txt[end:end + 1000]

    # a remark belongs to THIS statement only — stop at the next numbered statement
    nxt = re.search(r'(?:^|\n)[ \t]*\d{1,3}[ \t]*\.[ \t]+(?=[A-Z“"])', tail)
    if nxt:
        tail = tail[:nxt.start()]
    tail = norm(tail)

    remarks = [norm(r) for r in REMARK.findall(tail)]
    ty = TYPE.search(tail[:520])

    f = []
    if not (45 <= len(stmt) <= 700):                                   f.append('length')
    if kind == 'graded' and certainty is None:                         f.append('no_certainty')
    if kind != 'insuf' and not re.search(r'[“"](recommend|suggest)', stmt, re.I) \
       and not re.search(r'\bshould\b', stmt, re.I):                   f.append('no_verb')
    if not ty:                                                         f.append('no_type')

    # when a number matches more than one anchor, keep the one whose grade marker sits
    # closest after the statement text (the real box); a stray phrase in the narrative is farther away
    gap = start - body_from
    if n in records and gaps.get(n, 10**9) <= gap:
        continue
    gaps[n] = gap
    offsets[n] = start
    flags[n] = f
    records[n] = dict(id=f"stmt_{n:03d}", number=n, statement=stmt,
                      strength=strength, certainty=certainty,
                      type=ty.group(1).replace(' ', '_').lower() if ty else None,
                      remarks=remarks, pdf_page=page_at(start),
                      journal_page=JOURNAL_P1 + page_at(start) - 1)

# ---------- 4. domains ----------
HEADS = [("SCREENING AND EARLY MANAGEMENT", "screening_early_management", 16),
         ("INFECTION",                      "infection",                  25),
         ("HEMODYNAMIC MANAGEMENT",         "hemodynamic_management",     23),
         ("RESPIRATORY SUPPORT",            "respiratory_support",        14),
         ("ADJUNCTIVE THERAPIES",           "adjunctive_therapies",        8),
         ("ADDITIONAL SUPPORTIVE",          "additional_supportive",      13),
         ("GOALS OF CARE",                  "goals_of_care",               8),
         ("TRANSITIONS OF CARE",            "transitions_of_care",        16),
         ("LONG-TERM OUTCOMES",             "long_term_outcomes",          6)]
# statement numbers run in document order, so cumulative counts give the ranges
lo = 1
RANGES = []
for _, key, cnt in HEADS:
    RANGES.append((lo, lo + cnt - 1, key))
    lo += cnt
for n, rec in records.items():
    rec['domain'] = next((k for a, b, k in RANGES if a <= n <= b), None)
    rec['review_flags'] = flags.get(n, [])

if __name__ == "__main__":
    got = sorted(records)
    missing = [i for i in range(1, 130) if i not in records]
    print(f"captured {len(got)}/129    missing({len(missing)}): {missing}")
    print("\nstrength :", Counter(r['strength'] for r in records.values()).most_common())
    print("certainty:", Counter(r['certainty'] for r in records.values()).most_common())
    print("type     :", Counter(r['type'] for r in records.values()).most_common())
    print("flagged  :", sum(1 for f in flags.values() if f), "of", len(records))
    print("with remark:", sum(1 for r in records.values() if r['remarks']))
    json.dump([records[i] for i in got], open(OUT / "ssc2026_raw.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
