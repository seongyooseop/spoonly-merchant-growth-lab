"""Compare every headline estimate with the truth the simulation used (data/truth). Writes outputs/validation.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, TRUTH = ROOT / "outputs", ROOT / "data" / "truth"


def j(n):
    return json.loads((OUT / n).read_text(encoding="utf-8"))


def main():
    t = json.loads((TRUTH / "params.json").read_text(encoding="utf-8"))
    mp = pd.read_csv(TRUTH / "merchant_params.csv")
    mer = pd.read_csv(ROOT / "data" / "raw" / "merchants.csv")
    true_b = mp.merge(mer[["merchant_id", "category"]]).groupby("category").b.mean()
    c1, c2, c3, c4, c5 = j("ch1_segments.json"), j("ch2_coupons.json"), j("ch3_elasticity.json"), j("ch4_pilot.json"), j("ch5_forecast.json")
    checks = []

    def add(name, est, truth, ok, note=""):
        checks.append({"check": name, "estimate": est, "truth": truth, "pass": bool(ok), "note": note})

    lv = c1["levers"]
    add("평점 0.1점당 주문 효과", lv["rating_per_0p1"], np.exp(0.1 * 0.40) - 1, lv["rating_ci"][0] <= np.exp(0.04) - 1 <= lv["rating_ci"][1], "95% 구간 안")
    add("배달 1분 지연당 주문 효과", lv["delivery_per_min"], np.exp(-0.015) - 1, lv["delivery_ci"][0] <= np.exp(-0.015) - 1 <= lv["delivery_ci"][1], "95% 구간 안")
    # truth is a lift per eligible week; the estimate is per customer over 12 weeks, so the ranking is what is compared
    lift = t["reorder_test"]["lift_per_eligible_week"]
    order_est = [r["segment"] for r in sorted([r for r in c2["reorder"] if r["segment"] in lift and r["cost_per_incr"]], key=lambda r: r["cost_per_incr"])]
    add("재주문 쿠폰: 증분 주문당 비용 순위(휴면 최저)", order_est[0], "휴면(60일+)", order_est[0] == "휴면(60일+)")
    heavy = next(r for r in c2["reorder"] if r["segment"] == "주 1회 이상")
    add("재주문 쿠폰: 주 1회 이상 고객 효과 0과 구분 안 됨", heavy["diff"], 0.0, heavy["ci"][0] <= 0 <= heavy["ci"][1] or heavy["diff"] < 0.05)
    fo = c2["first_order"]
    add("첫 주문 쿠폰: 신규 고객 증가율", fo["lift"], t["first_order_test"]["extra_new_share"], abs(fo["lift"] - 0.32) <= 0.06, "±6%p")
    add("첫 주문 쿠폰: 쿠폰으로 늘어난 고객 8주 재주문율", fo["reorder_marginal"], t["first_order_test"]["repeat_marginal"], abs(fo["reorder_marginal"] - 0.22) <= 0.05, "±5%p")
    for r in c3["categories"]:
        tb = true_b[r["category"]]
        add(f"할인 반응(캠페인 기준) {r['category']}", r["b"], float(tb), abs(r["b"] / tb - 1) <= 0.25, "±25%")
    worse = sum(abs(r["b_fe_merchant_disc"] - true_b[r["category"]]) > abs(r["b"] - true_b[r["category"]]) for r in c3["categories"])
    add("매장 자체 할인으로 잰 반응이 더 틀림(함정 1)", worse, len(c3["categories"]), worse >= len(c3["categories"]) - 1, "범주 수 기준")
    tw = pd.read_parquet(TRUTH / "merchant_week_truth.parquet")
    tiers = pd.read_csv(OUT / "merchant_tiers.csv")
    g = tiers[(tiers.tier == "성장") & (tiers.area == "W")].merchant_id
    x = tw[tw.merchant_id.isin(g) & (tw.week >= 30)]
    true_lo = float(np.log(x.lam.sum() / x.lam_no_discount.sum()) - np.log(
        tw[tw.merchant_id.isin(g) & tw.week.between(14, 29)].pipe(lambda y: y.lam.sum() / y.lam_no_discount.sum())))
    did = c4["did"]["lo"]
    add("비용 분담 시범: 주문 효과(DiD)", did["est"], true_lo, did["ci"][0] - 0.03 <= true_lo <= did["ci"][1] + 0.03, "95% 구간 ±3%p")
    add("예측: 행사 주 오차가 단순 평균보다 작음", c5["mape_event_bu"], c5["mape_event_naive"], c5["mape_event_bu"] < c5["mape_event_naive"])
    add("예측: 연말 급증은 놓침(함정 5, 예상된 실패)", c5["year_end_miss"], -0.18, c5["year_end_miss"] < -0.10, "모델에 없는 패턴")
    out = {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks)}
    (OUT / "validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"], round(c["estimate"], 3) if isinstance(c["estimate"], float) else c["estimate"], c["truth"] if not isinstance(c["truth"], float) else round(c["truth"], 3))
    print(f"{out['passed']} / {out['total']}")


if __name__ == "__main__":
    main()
