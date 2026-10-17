"""Regenerate thesis/bib/bucket_*.bib from the audited references.bib.

references.bib is the single source of truth (it is what main.tex compiles and
what the Ch1/Ch2 citation audits corrected). The buckets are only a Zotero
import convenience, so they must be derived from it, never the other way round.
"""
import re, pathlib

THESIS = pathlib.Path(__file__).resolve().parents[2] / "thesis"
REF = THESIS / "references.bib"
BIB = THESIS / "bib"

def entries(text):
    out = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        i = text.index('{', m.start()); d = 0
        for j in range(i, len(text)):
            if text[j] == '{': d += 1
            elif text[j] == '}':
                d -= 1
                if d == 0: break
        out[m.group(2).strip()] = text[m.start():j + 1]
    return out

BUCKETS = {
 "A_foundations": """imt2020_6g rappaport2019wireless saad2019vertical rajatheva2020whitepaper
   dang2020what6g liu2020toward6g zhang20216gvision yastrebova2022business6g jiang2021road
   matthaiou2021road schulz2018latency popovski20184g nasrullah2019ultra durisi2016short
   polyanskiy2010finite bennis2018tail wijethilaka2021slicing wu2022networkslicing
   li2021tactile5g aijaz2019private5g varga2020wireless series2015factory gubbi2013internet tanghe2008industrial
   breque2021industry50 nahavandi2019industry5 xu2021industry50 daudu2017industrial5g
   barreto2020industrial antonopoulos2020industrialiot cheng2018industrial holfeld2016wireless
   3gpp_tr38901 3gpp_tr22804 proakis2008digital tse2005fundamentals goldsmith2005wireless""",
 "B_ml_channel_estimation": """oshea2017introduction chen2021deep hoydis2017deep neumann2018deep
   soltani2019learned li2020reesnet ye2018power honkala2021deeprx luan2023channelformer
   balevi2020untrained he2018lce liu2023modeldriven coleri2002estimation ozdemir2007channel
   hoydis2013five studer2011channel shlezinger2020model reis2022aice gao2023dlce
   kim2022towardsdlce zheng2022irsce yu2022role zappone2019wireless zhang2017dncnn""",
 "C_ml_resource_management": """luo2022deep tse1998multiaccess viswanathan2002wireless shi2011wmmse
   tassiulas1992stability mao2016survey shalev2016safe mnih2015human sutton2018reinforcement
   luong2019applications luong2021uav mao2018deep wang2020deep wang2018drlaccess ye2019drlv2v
   meng2020power ju2022energydrl li2022drlresource sun2018learning eisen2019learning
   wang2025rldu alsenwi2021slicing yoo2006capacity mo2000fair jain1984fairness
   neely2010stochastic henderson2018matters garcia2015safe levine2020offline
   shen2021gnn chowdhury2021unfolding""",
 "D_integration": """li2018energy sadeghi2019physics dorner2018deep letaief20184g5g
   chiang2007layering wang2005crosslayer ramly2021crosslayer wang2019thirty
   kawadia2005cautionary""",
 "E_tools": """sionna2022 hoydis2023sionnart abadi2016tensorflow ldpc_dvb""",
}

ref = entries(REF.read_text(encoding="utf-8"))
assigned, missing = set(), {}
for name, keys in BUCKETS.items():
    ks = keys.split()
    have = [k for k in ks if k in ref]
    missing[name] = [k for k in ks if k not in ref]
    assigned.update(have)
    body = "\n\n".join(ref[k] for k in have)
    (BIB / f"bucket_{name}.bib").write_text(
        f"% Bucket {name.split('_')[0]} — generated from thesis/references.bib.\n"
        f"% references.bib is the source of truth; do not hand-edit this file.\n"
        f"% Regenerate: python scripts/tools/resync_thesis_bib_buckets.py\n\n{body}\n",
        encoding="utf-8")
    print(f"bucket_{name}.bib: {len(have)} entries"
          + (f"  (skipped, not in references.bib: {missing[name]})" if missing[name] else ""))

orphans = sorted(set(ref) - assigned)
print("\nIn references.bib but assigned to no bucket:", orphans or "none")
