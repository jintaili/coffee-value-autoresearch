"""Build one shareable HTML report from saved model and serving-pilot results."""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "artifacts" / "jev_serving_benchmark"
OUT = ROOT / "plans" / "jev-results-showcase.html"
PUBLIC_DATA = ROOT / "plans" / "jev-results-data.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    base_report = read_json(ROOT / "artifacts" / "jev" / "training_report.json")
    rating = base_report["tasks"]["rating"]
    base_price = base_report["tasks"]["price"]
    price = read_json(ROOT / "artifacts" / "jev_auction" / "training_report.json")
    bench = read_json(BENCH / "summary.json")
    events = [json.loads(line) for line in (BENCH / "events.jsonl").read_text().splitlines()]
    completed = {(item["url"], item["round"], item["provider"]): item
                 for item in events if item["status"] == "complete"}
    if price["status"] != "complete" or bench["paired"].get("complete_pairs") != 16:
        raise ValueError("the model and serving pilot must be complete")

    r_base = rating["baseline"]["validation"]
    r_jev = rating["hybrid"]["validation"]
    p_base = price["runs"]["incumbent"]["validation"]
    p_jev = price["runs"]["jev_original"]["validation"]
    p_pg = price["runs"]["jev_pg"]["validation"]
    p_auction = price["runs"]["jev_auction"]["validation"]
    rating_interval = base_report["independent_audit"]["tasks"]["rating"]["runs"]["hybrid"]["paired_primary_delta_95pct_interval"]
    price_base_interval = base_report["independent_audit"]["tasks"]["price"]["runs"]["hybrid"]["paired_primary_delta_95pct_interval"]
    price_pg_interval = price["paired_intervals"]["intervals"]["jev_pg"]["rmsle_delta_vs_incumbent_95pct_interval"]
    old = bench["providers"]["openai_app"]
    new = bench["providers"]["jev_ab"]
    p50_reduction = 100 * (1 - new["p50_seconds"] / old["p50_seconds"])
    p95_reduction = 100 * (1 - new["p95_seconds"] / old["p95_seconds"])

    public_events = [
        {key: item[key] for key in (
            "url", "round", "provider", "context_sha256", "context_chars", "cache_hit",
            "started_utc", "status", "model", "latency_seconds", "semantic_fields")}
        for item in sorted(completed.values(), key=lambda item: (item["url"], item["round"], item["provider"]))
    ]
    public_data = {
        "study_date": "2026-09-22",
        "model_validation": {
            "base_contract": price["base_contract"],
            "auction_contract": price["auction_contract"],
            "rating_rows": {"train": rating["train_rows"], "validation": rating["validation_rows"]},
            "price_rows": {"train": base_price["train_rows"], "validation": base_price["validation_rows"]},
            "split_hashes": price["split_hashes"],
            "rating": {"incumbent": r_base, "jev_ab": r_jev},
            "price": {"incumbent": p_base, "jev_ab": p_jev,
                      "jev_panama_gesha": p_pg, "jev_auction": p_auction},
            "exploratory_roaster_cluster_95pct_intervals": {
                "rating_jev_ab_concordance_gain_vs_incumbent": rating_interval,
                "price_jev_ab_rmsle_delta_vs_incumbent": price_base_interval,
                "price_jev_panama_gesha_rmsle_delta_vs_incumbent": price_pg_interval,
            },
        },
        "extraction_pilot": {
            "summary": bench,
            "successful_calls": public_events,
            "note": "Frozen page context text and provider credentials are not included. Product pages may change after capture.",
        },
    }
    PUBLIC_DATA.write_text(json.dumps(public_data, indent=2, ensure_ascii=False))

    page_rows = []
    for snapshot in bench["contexts"]:
        url = snapshot["url"]
        old_times = [completed[(url, round_no, "openai_app")]["latency_seconds"] for round_no in (1, 2)]
        new_times = [completed[(url, round_no, "jev_ab")]["latency_seconds"] for round_no in (1, 2)]
        domain = urlparse(url).netloc.removeprefix("www.")
        page_rows.append(f"<tr><td><a href=\"{html.escape(url, quote=True)}\">"
                         f"{html.escape(snapshot['product_name'])}</a><small>{html.escape(domain)}</small></td>"
                         f"<td>{snapshot['context_chars']:,}</td>"
                         f"<td>{statistics.median(old_times):.2f}</td>"
                         f"<td>{statistics.median(new_times):.2f}</td></tr>")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JEV validation gains and extraction-call pilot</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; background: #f5f5ef; color: #21312a; font: 16px/1.55 system-ui, sans-serif; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 2.5rem 1.2rem 5rem; }}
h1,h2 {{ line-height: 1.2; letter-spacing: -.02em; }} h1 {{ font-size: clamp(2rem,5vw,3rem); margin: .2rem 0 1rem; }}
h2 {{ margin: 2.3rem 0 .7rem; }} p {{ max-width: 78ch; }}
.eyebrow {{ color: #536e5e; font-size: .78rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }}
.lead {{ font-size: 1.18rem; max-width: 76ch; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(230px,1fr)); gap: .9rem; margin: 1.5rem 0; }}
.card {{ background: white; border: 1px solid #d8e2d8; border-radius: 12px; padding: 1rem 1.1rem; }}
.card strong {{ display:block; font-size: 1.65rem; line-height: 1.2; color: #14593d; }}
.card span {{ color:#53665b; font-size:.91rem; }}
.table-wrap {{ overflow-x:auto; background:white; border:1px solid #d8e2d8; border-radius:12px; }}
table {{ width:100%; border-collapse:collapse; min-width:660px; }} th,td {{ padding:.65rem .75rem; text-align:left; border-bottom:1px solid #e5ebe5; vertical-align:top; }}
th {{ background:#ecf3ec; font-size:.86rem; }} tr:last-child td {{ border-bottom:0; }} td small {{ display:block; color:#6a786f; }}
.note {{ border-left:4px solid #c6883a; background:#fff8e8; padding:.7rem 1rem; }}
.muted {{ color:#5b6a60; font-size:.93rem; }} a {{ color:#176345; }}
</style></head><body><main>
<div class="eyebrow">Coffee value · research and local extraction-call pilot · 22 September 2026</div>
<h1>Modest validation gains; faster JEV calls in a small pilot</h1>
<p class="lead">The best price candidate lowers historical validation RMSLE by 2.14% versus the incumbent, and the rating hybrid raises concordance by 0.00148. On eight frozen product pages, the 38-question JEV call took 0.42 seconds at the median, versus 4.81 seconds for the app’s current full extraction call. The timing paths return different amounts of information, so this is a latency observation for the calls tested.</p>
<div class="cards">
  <div class="card"><strong>−2.14%</strong><span>price RMSLE, best JEV hybrid vs incumbent</span></div>
  <div class="card"><strong>+0.00148</strong><span>rating concordance, JEV hybrid vs research winner</span></div>
  <div class="card"><strong>−{p50_reduction:.1f}%</strong><span>JEV semantic call vs current full app extraction call; outputs differ</span></div>
</div>
<h2>Prediction performance</h2>
<div class="table-wrap"><table><thead><tr><th>Task and metric</th><th>Incumbent</th><th>JEV A+B hybrid</th><th>Best tested follow-up</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Rating concordance ↑</td><td>{r_base['val_concordance']:.6f}</td><td>{r_jev['val_concordance']:.6f}</td><td>—</td><td>+{r_jev['val_concordance']-r_base['val_concordance']:.6f}; below the proposed +0.005 target</td></tr>
<tr><td>Rating MAE ↓</td><td>{r_base['val_mae']:.3f}</td><td>{r_jev['val_mae']:.3f}</td><td>—</td><td>Small improvement</td></tr>
<tr><td>Price RMSLE ↓</td><td>{p_base['val_rmsle']:.6f}</td><td>{p_jev['val_rmsle']:.6f}</td><td>{p_pg['val_rmsle']:.6f} with Panama × Gesha</td><td>Best gain: {100*(p_base['val_rmsle']-p_pg['val_rmsle'])/p_base['val_rmsle']:.2f}%; below the proposed 3% target</td></tr>
<tr><td>Price MAE, USD/100 g ↓</td><td>{p_base['val_mae']:.3f}</td><td>{p_jev['val_mae']:.3f}</td><td>{p_pg['val_mae']:.3f}</td><td>MAE rises by about 1%</td></tr>
</tbody></table></div>
<p>The added exact-lot auction question yielded RMSLE {p_auction['val_rmsle']:.6f}, slightly worse than the original JEV hybrid’s {p_jev['val_rmsle']:.6f}. The Panama × Gesha term uses two existing JEV answers and requires no extra provider question. The most expensive price decile remains strongly underpredicted. Exploratory paired intervals cross zero for the rating gain [{rating_interval[0]:+.4f}, {rating_interval[1]:+.4f}] and the best price candidate’s RMSLE change [{price_pg_interval[0]:+.4f}, {price_pg_interval[1]:+.4f}]. These historical validation rows were repeatedly used during research, so the gains need a fresh evaluation. See the <a href="jev-training-results.html">rating and original JEV report</a> and <a href="jev-auction-results.html">price follow-up detail</a>.</p>
<h2>Extraction-call latency on frozen pages</h2>
<div class="table-wrap"><table><thead><tr><th>Measure</th><th>Current app extractor<br>gpt-4o-mini</th><th>JEV A+B<br>jev-1.13.0</th><th>Observed reduction</th></tr></thead><tbody>
<tr><td>p50</td><td>{old['p50_seconds']:.2f} s</td><td>{new['p50_seconds']:.2f} s</td><td>{p50_reduction:.1f}%</td></tr>
<tr><td>p95</td><td>{old['p95_seconds']:.2f} s</td><td>{new['p95_seconds']:.2f} s</td><td>{p95_reduction:.1f}%</td></tr>
<tr><td>Successful calls</td><td>{old['complete']}/16</td><td>{new['complete']}/16</td><td>JEV faster in all {bench['paired']['jev_faster_pairs']} pairs</td></tr>
</tbody></table></div>
<p class="muted">Eight product pages across three roasters, two uncached calls per provider per frozen page context, one call at a time in randomized order on the same local host. The page fetch and context construction were done once before timing. p95 is the maximum of only 16 observations and is unstable. The JEV input-token charge estimate was ${new['estimated_input_cost_usd']:.4f} for 16 successful calls; OpenAI usage was not recorded, so no cost comparison is claimed.</p>
<div class="table-wrap"><table><thead><tr><th>Frozen page</th><th>Context characters</th><th>App median, s</th><th>JEV median, s</th></tr></thead><tbody>{''.join(page_rows)}</tbody></table></div>
<p class="note"><strong>Scope of the timing:</strong> the app’s existing call returns a full PageExtraction, including price, package size, display tasting notes, and page classification. The JEV call returns 38 typed semantic answers for model features. Both saw the same saved page context, but JEV does not yet provide the complete response the app needs. These measurements establish the latency of the semantic call, not an end-to-end serving speedup.</p>
<h2>Extraction quality and serving decision</h2>
<p>All final benchmark calls completed after the shared TypeSafe client was given an application User-Agent. Two TypeSafe preflight calls received HTTP 403 before that fix and are excluded from latency statistics. On two pages whose saved contexts contain no “natural” process claim, JEV nevertheless marked natural processing supported in both rounds. A rotating subscription page also drew inconsistent attributes from historical offerings in the app extractor. Product identity and evidence selection need a matched-page quality check before changing the serving path.</p>
<p>The observed call-latency reductions exceed the numerical extraction targets, but this pilot does not meet the targets' equivalent-output condition. The next serving measurement should run a JEV-based path that also supplies the required commerce and display fields, then time the full appraisal with the same URLs and completeness rules. The price and rating models remain research candidates because their gains miss the proposed prediction targets and no fresh product-page prediction set has been evaluated.</p>
<p class="muted">Inspect the <a href="jev-results-data.json">public evidence file</a> for the aggregate model metrics, page-context hashes, and individual call timings and semantic outputs. It excludes the third-party page text and credentials. The frozen contexts and full research artifacts remain local, so an exact replay requires those files; refetching a page can produce different content. The local harness is <code>scripts/benchmark_jev_serving_extraction.py</code>.</p>
</main></body></html>"""
    OUT.write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()
