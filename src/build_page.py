"""Charts (site/img/*.png), the report page (site/index.html), the one-page summary (site/summary.html)
and its PDF (site/SPOONLY_portfolio_summary.pdf, printed with headless Edge or Chrome). Numbers come from outputs/*.json."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, SITE, IMG = ROOT / "outputs", ROOT / "site", ROOT / "site" / "img"
REPO = "https://github.com/seongyooseop/spoonly-merchant-growth-lab"
DEMO = "https://seongyooseop.github.io/spoonly-merchant-growth-lab/site/"
COPY_TO = ROOT.parent / "성유섭_포트폴리오요약_쿠팡이츠_사업전략비즈니스분석.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.unicode_minus": False})


def j(n):
    return json.loads((OUT / n).read_text(encoding="utf-8"))


def won(v, d=0):
    a = abs(v)
    s = "−" if v < 0 else ""
    if a >= 1e8:
        return f"{s}{a / 1e8:,.{max(d, 1)}f}억 원"
    if a >= 1e5:
        return f"{s}{a / 1e4:,.0f}만 원"
    if a >= 1e4:
        return f"{s}{a / 1e4:,.1f}만 원"
    return f"{s}{a:,.0f}원"


def pc(v, d=0, sign=False):
    return (f"{100 * v:+.{d}f}%" if sign else f"{100 * v:.{d}f}%").replace("-", "−")


def save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def charts(c2, c3, c4, c5):
    # 2: cost per incremental order by customer segment
    r = pd.DataFrame([x for x in c2["reorder"] if x["segment"] != "전체"])
    r["cpi"] = r.cost_per_incr.fillna(60000).clip(upper=60000)
    r = r.sort_values("cpi", ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    ax.barh(r.segment, r.cpi / 1e3, color=[BLUE if v <= 6100 else "#b9c9df" for v in r.cpi], height=0.55)
    thr = c2["value_per_incr_order"] / 1e3
    ax.axvline(thr, color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(thr + 0.8, len(r) - 0.45, "증분 주문 1건 가치 4,500원 (가정)", fontsize=8.5, color=INK, va="bottom")
    for y, (v, raw) in enumerate(zip(r.cpi, r.cost_per_incr)):
        ax.text(max(v / 1e3, thr) + 1.0, y, "증분 없음" if pd.isna(raw) else f"{raw:,.0f}원", va="center", fontsize=8.5, color=INK)
    ax.set_ylim(-0.6, len(r) + 0.1)
    ax.set_xlim(0, 70); ax.set_xlabel("증분 주문 1건당 쿠폰 비용 (천 원)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "ch2_cost_per_incremental.png")

    # 3: orders per 1,000,000 won of platform coupon, by category
    e = pd.DataFrame(c3["categories"]).sort_values("orders_per_1m_won")
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.barh(e.category, e.orders_per_1m_won, color=[BLUE if c in c3["top"] else ("#b9c9df" if c in c3["bottom"] else "#8fb3e3") for c in e.category], height=0.6)
    for y, v in enumerate(e.orders_per_1m_won):
        ax.text(v + 5, y, f"{v:,.0f}건", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, e.orders_per_1m_won.max() * 1.18); ax.set_xlabel("쿠폰 100만 원당 늘어나는 주문 (건)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "ch3_orders_per_won.png")

    # 4: event study, share of growth-tier merchants running a discount
    es = pd.DataFrame(c4["event_study"])
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    ax.axhline(0, color=GRID, lw=1); ax.axvline(-0.5, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.errorbar(es.bin, 100 * es.est, yerr=[100 * (es.est - es.lo), 100 * (es.hi - es.est)], fmt="o", color=BLUE, ms=5, capsize=0, lw=1.4)
    ax.text(-0.35, 100 * es.hi.max(), "시범 시작 (30주차)", fontsize=8.5, color=MUTED, va="top")
    ax.set_xticks(es.bin, [f"{4 * b:+d}" for b in es.bin]); ax.set_xlabel("시범 시작 대비 주 (4주 묶음)")
    ax.set_ylabel("할인 참여 매장 비중 차이 (%p)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "ch4_event_study.png")

    # 5: Q4 weekly orders, actual vs forecasts
    w = pd.DataFrame(c5["weekly"])
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(w.week, w.actual / 1e3, color=INK, lw=2, label="실제")
    ax.plot(w.week, w.bottom_up / 1e3, color=BLUE, lw=2, marker="o", ms=4, label="바텀업 예측")
    ax.plot(w.week, w.naive / 1e3, color=ORANGE, lw=2, ls=(0, (4, 2)), label="직전 13주 평균")
    for wk, lab in [(40, "추석"), (44.5, "캠페인"), (51.5, "연말")]:
        ax.text(wk, w.actual.max() / 1e3 * 1.03, lab, ha="center", fontsize=8.5, color=MUTED)
    ax.set_ylim(150, w.actual.max() / 1e3 * 1.09); ax.set_xlabel("2025년 주차"); ax.set_ylabel("주간 주문 (천 건)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "ch5_forecast.png")


CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;--soft:#eef4fc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222220;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--accent:#3987e5;--soft:#1f2a38}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222220;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--accent:#3987e5;--soft:#1f2a38}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;line-height:1.6;font-size:15px}
.wrap{max-width:980px;margin:0 auto;padding:28px 16px 56px}h1{font-size:28px;margin:4px 0 6px;line-height:1.3}h2{font-size:20px;margin:0 0 6px}h3{font-size:15px;margin:14px 0 4px}
.eyebrow{font-size:12px;color:var(--muted);letter-spacing:.04em}.lede{color:var(--muted);max-width:70ch}.note{font-size:13px;color:var(--muted)}
.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--muted);margin:2px 4px 2px 0}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-top:16px}
.memo{border-left:4px solid var(--accent)}.memo ol{padding-left:20px;margin:6px 0}.memo li{margin:4px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:10px}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 12px}.tile b{display:block;font-size:22px;line-height:1.2}.tile span{font-size:12.5px;color:var(--muted)}
.two{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:18px;align-items:start}@media(max-width:820px){.two{grid-template-columns:1fr}}
img{max-width:100%;height:auto;display:block;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:6px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12.5px}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl{overflow-x:auto}a{color:var(--accent)}footer{margin-top:22px;font-size:12.5px;color:var(--muted)}
"""


def page(c0, c1, c2, c3, c4, c5, c6, val):
    y, camp = c0["year"], c0["campaign"]
    lv, op = c1["levers"], c1["opportunity"]
    tg, o2, fo = c2["targeted"], c2["targeted"]["option2"], c2["first_order"]
    allr = next(r for r in c2["reorder"] if r["segment"] == "전체")
    cats = pd.DataFrame(c3["categories"])
    top, bot = cats.iloc[0], cats.iloc[-1]
    ra = c3["realloc"]
    did = c4["did"]
    tier_order = ["핵심", "성장", "유지", "관리"]
    tiers = {t["tier"]: t for t in c1["tiers"]}
    tier_desc = {"핵심": "주문 상위 · 평점 4.4+ · 배달 36분 이내", "성장": "주문 하위 · 평점 4.4+ · 배달 36분 이내",
                 "유지": "주문 상위 · 평점 또는 배달 기준 미달", "관리": "주문 하위 · 평점 또는 배달 기준 미달"}
    tier_act = {"핵심": "매장 부담 할인 + 행사 노출, 플랫폼 보조는 캠페인만", "성장": "플랫폼 매칭(조건부) + 노출 지원",
                "유지": "배달 시간·평점 개선을 먼저, 보조는 축소", "관리": "운영 지표 개선 목표를 걸고 보조 없음"}
    tier_rows = "".join(f"<tr><td><b>{t}</b><br><span class='note'>{tier_desc[t]}</span></td><td class='n'>{tiers[t]['n']:,}</td>"
                        f"<td class='n'>{pc(tiers[t]['gmv_share'])}</td><td class='n'>{tiers[t]['orders_wk']:.0f}</td>"
                        f"<td class='n'>{pc(tiers[t]['repeat'])}</td><td class='n'>{tiers[t]['delivery']:.1f}분</td><td>{tier_act[t]}</td></tr>" for t in tier_order)
    def cpi(r):
        return "증분 없음" if r["cost_per_incr"] is None else f"{r['cost_per_incr']:,.0f}원"

    def share(r):
        return "—" if r["incr_share"] is None else pc(max(r["incr_share"], 0))

    seg_rows = "".join(f"<tr><td>{r['segment']}</td><td class='n'>{r['diff']:+.3f}</td><td class='n'>{r['redemptions']:,}</td>"
                       f"<td class='n'>{r['incremental']:,.0f}</td><td class='n'>{share(r)}</td><td class='n'>{cpi(r)}</td></tr>"
                       for r in sorted(c2["reorder"], key=lambda r: ["주 1회 이상", "월 2~3회", "월 1회 이하", "휴면(60일+)", "전체"].index(r["segment"])))
    cat_rows = "".join(f"<tr><td>{r.category}</td><td class='n'>{r.b_fe_merchant_disc:.2f}</td><td class='n'>{r.b:.2f} [{r.ci[0]:.2f}, {r.ci[1]:.2f}]</td>"
                       f"<td class='n'>{r.aov:,.0f}원</td><td class='n'>{r.orders_per_1m_won:,.0f}건</td></tr>" for r in cats.itertuples())
    val_rows = "".join(f"<tr><td>{c['check']}</td><td class='n'>{c['estimate'] if isinstance(c['estimate'], str) else round(c['estimate'], 3)}</td>"
                       f"<td class='n'>{c['truth'] if isinstance(c['truth'], str) else round(c['truth'], 3)}</td><td>{'통과' if c['pass'] else '미통과'}</td></tr>" for c in val["checks"])
    base, bp = c6["base"], c2["breakpoints"]
    sens_lo, sens_hi = min(x["total_value"] for x in c6["sensitivity"]), max(x["total_value"] for x in c6["sensitivity"])
    fr = c6["full_rollout"]
    memo = [
        f"<b>재주문 쿠폰은 전 고객으로 넓히지 말고 휴면 고객에게만 보낸다.</b> 쿠폰 사용 주문 {allr['redemptions']:,}건 중 실제로 늘어난 주문은 {allr['incremental']:,.0f}건({pc(allr['incr_share'])})이었다. "
        f"시험 방식 그대로 전 고객에 넓히면 분기 쿠폰비 약 {won(fr['coupon_cost'])}, 증분 주문당 {fr['cost_per_incr']:,.0f}원이다. 휴면 고객은 증분 주문당 {bp['휴면(60일+)']:,.0f}원, "
        f"월 1회 이하 고객은 {bp['월 1회 이하']:,.0f}원이라, 증분 주문 가치가 {bp['월 1회 이하']:,.0f}원 이상으로 확인될 때만 월 1회 이하를 더한다.",
        f"<b>플랫폼 캠페인 예산을 할인 반응이 큰 카테고리로 옮기되, 실험한 할인 범위 안에서만 옮긴다.</b> 쿠폰 100만 원당 주문은 {top.category} {top.orders_per_1m_won:,.0f}건, {bot.category} {bot.orders_per_1m_won:,.0f}건으로 약 {top.orders_per_1m_won / bot.orders_per_1m_won:.1f}배 차이. "
        f"하위 3개 카테고리 예산의 {pc(ra['share_of_bottom_moved'])}({won(ra['spend_moved'])})를 옮기면 상위 카테고리 평균 할인율이 실험 최대치 10%에 닿고, 같은 예산으로 캠페인 주문 {pc(ra['rel'], 1, True)}.",
        f"<b>첫 주문 쿠폰은 '8주 안에 다시 주문한 신규 고객당 비용'으로 관리한다.</b> 쿠폰이 신규 고객을 {pc(fo['lift'])} 늘렸지만(무작위화 추론 p = {fo['p_permutation']:.3f}), 쿠폰으로 늘어난 고객의 8주 재주문율은 {pc(fo['reorder_marginal'])}(자연 유입 {pc(fo['reorder_organic'])}). 재주문 고객 1명당 약 {won(fo['cost_per_incr_retained'])}.",
        f"<b>비용 분담 매칭은 비율을 낮춰 다시 시험한 뒤 넓힌다.</b> 서울 서부 성장 등급 매장에서 할인 참여 {did['disc_on']['est'] * 100:+.1f}%p, 주문 {pc(did['lo']['est'], 1, True)}였지만 늘어난 주문 1건당 플랫폼 비용이 {c4['cost_per_extra_order']:,.0f}원이다. 증분 주문 가치가 6,000원 이상일 때만 지금 비율로 남는다.",
        f"<b>쿠폰보다 큰 기회는 배달 시간이다.</b> 배달 36분 초과 매장 {op['slow_n']:,}곳을 32분으로 줄이면 하반기 GMV 기준 약 {won(op['gain_delivery_h2'])}에 해당하는 주문이 늘어난다(1분당 주문 {pc(lv['delivery_per_min'], 1)}). 운영 지원 비용이 확인되지 않아 아래 합계에는 넣지 않았다.",
    ]
    impact_rows = "".join(f"<tr><td>{r['lever']}</td><td class='n'>{r['orders']:+,.0f}</td><td class='n'>{won(r['coupon_cost']) if r['coupon_cost'] else '0'}</td>"
                          f"<td class='n'>{won(r['value_won'])}</td></tr>" for r in base["rows"])
    impact_rows += (f"<tr><td><b>합계</b></td><td class='n'><b>{base['total_orders']:+,.0f}</b></td><td class='n'><b>{won(sum(r['coupon_cost'] for r in base['rows']))}</b></td>"
                    f"<td class='n'><b>{won(base['total_value'])}</b></td></tr>")
    seg_label = {(): "보내지 않음", ("휴면(60일+)",): "휴면", ("휴면(60일+)", "월 1회 이하"): "휴면 + 월 1회 이하"}
    sens_rows = "".join(f"<tr><td class='n'>{x['value']:,}원</td><td>{seg_label.get(tuple(sorted(x['segments'], key=lambda z: z != '휴면(60일+)')), ', '.join(x['segments']))}</td>"
                        f"<td>{'남음' if c4['pays_back'].get(str(x['value'])) else '안 남음'}</td><td class='n'>{won(x['total_value'])}</td></tr>" for x in c6["sensitivity"])
    rep_rows = "".join(f"<tr><td>{r['segment']}</td><td class='n'>{pc(next(q['repeat_4w'] for q in c2['repeat_4w'] if q['segment'] == r['segment'] and q['arm'] == '홀드아웃'), 1)}</td>"
                       f"<td class='n'>{pc(next(q['repeat_4w'] for q in c2['repeat_4w'] if q['segment'] == r['segment'] and q['arm'] == '쿠폰'), 1)}</td></tr>"
                       for r in [{"segment": x} for x in ["주 1회 이상", "월 2~3회", "월 1회 이하", "휴면(60일+)"]])
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPOONLY 매장 성장 분석</title><style>{CSS}</style></head><body><div class="wrap">
<div class="eyebrow">SPOONLY · 가상의 배달앱 · 서울 동부·서울 서부·경기 남부 · 2025년 52주</div>
<h1>입점 매장 성장 분석: 할인·쿠폰의 실제 효과를 가르고, 예산·분담 기준·분기 예측까지</h1>
<p class="lede">가상의 배달앱 SPOONLY의 매장 {y['merchants']:,}곳·주문 {y['orders']:,}건(GMV {won(y['gmv'])})을 시뮬레이션해, 사업전략·비즈니스 분석 담당자가 하는 일을 처음부터 끝까지 보여주는 포트폴리오입니다.
KPI 정의 → 매장 세분화 → 쿠폰 증분 실험 → 카테고리별 할인 반응 → 비용 분담 시범 평가 → 분기 매출 예측 → 경영진 메모 순서입니다. 데이터에 함정 다섯 가지를 넣어 두었고, 모든 추정치를 참값과 대조했습니다({val['passed']}개 중 {val['passed']}개 통과).</p>
<span class="badge">모든 데이터는 가상</span><span class="badge">실제 회사·고객 정보 없음</span><span class="badge">SQL(DuckDB) · Python(statsmodels)</span>
<span class="badge"><a href="{REPO}">코드 저장소</a></span>

<section class="memo"><div class="eyebrow">경영진 메모 · 2026 상반기 매장 성장 계획</div>
<h2>쿠폰은 '쓰인 건수'가 아니라 '늘어난 주문'으로 배분하고, 매장 등급마다 분담 기준을 달리한다</h2>
<p><b>측정된 두 권고(1·2번)를 합치면 분기당 쿠폰비 {won(-sum(r['coupon_cost'] for r in base['rows']))}을 줄이고, 증분 주문 가치 기준 {won(base['total_value'])}의 효과가 있습니다</b>
(전 고객 확대안 대비, 증분 주문 1건 = 4,500원 가정). 이 가치를 3,000~7,500원 어느 값으로 잡아도 효과는 {won(sens_lo)}~{won(sens_hi)}으로 양수이고, 달라지는 것은 쿠폰 대상 범위입니다.</p>
<ol>{''.join(f'<li>{m}</li>' for m in memo)}</ol>
<div class="two"><div><h3>분기 효과 (증분 주문 1건 = 4,500원)</h3>
<div class="tbl"><table><thead><tr><th>권고</th><th>주문</th><th>쿠폰비</th><th>가치</th></tr></thead><tbody>{impact_rows}</tbody></table></div></div>
<div><h3>가정을 바꾸면</h3>
<div class="tbl"><table><thead><tr><th>증분 주문 가치</th><th>재주문 쿠폰 대상</th><th>50% 매칭</th><th>분기 효과</th></tr></thead><tbody>{sens_rows}</tbody></table></div></div></div>
<p class="note">증분 주문 1건의 가치 = 수수료(약 1,700원) + 그 주문이 만드는 이후 재주문의 가치. 재주문 쿠폰 효과는 실험 고객 9만 명의 주문 비중으로 플랫폼 전체에 비례 확대했고(약 {c6['scale']:.1f}배), 캠페인 재배분은 분기 1회 캠페인 기준입니다.</p></section>

<section><h2>0. KPI 트리: 무엇이 GMV와 플랫폼 수익을 움직이나</h2>
<p>GMV = 주문 수 × 객단가, 주문 = 신규 + 재주문, 플랫폼 순수익 = GMV × 수수료 10% − 플랫폼 부담 할인. 캠페인 주에는 주문이 늘지만 주문당 순수익이 크게 줄어, '주문 수'만 보면 쿠폰이 늘 좋아 보입니다.</p>
<div class="tiles">
<div class="tile"><b>{won(y['gmv'])}</b><span>연간 GMV · 주문 {y['orders'] / 1e6:.1f}백만 건 · 객단가 {y['aov']:,.0f}원</span></div>
<div class="tile"><b>{pc(y['repeat_share'])}</b><span>재주문 비중 (주문 기준)</span></div>
<div class="tile"><b>{pc(camp['orders_per_week'] / camp['orders_per_week_near'] - 1, 0, True)}</b><span>캠페인 주 주문 (앞뒤 2주 대비)</span></div>
<div class="tile"><b>{camp['net_per_order']:,.0f}원</b><span>캠페인 주 주문당 플랫폼 순수익 (평소 {camp['net_per_order_near']:,.0f}원)</span></div>
</div></section>

<section><h2>1. 매장 세분화와 성장 레버</h2>
<p>상반기 주문 수준(중앙값 기준)과 성장 잠재력(평점 4.4 이상·배달 36분 이내)으로 {c1['n']:,}곳을 네 등급으로 나눴습니다. 가입 1주차 매장 {lv['n_est']:,}곳의 횡단면 회귀(카테고리·지역·객단가 통제)로 레버 크기를 쟀습니다:
평점 0.1점당 주문 {pc(lv['rating_per_0p1'], 1, True)} [{pc(lv['rating_ci'][0], 1)}, {pc(lv['rating_ci'][1], 1)}], 배달 1분 지연당 {pc(lv['delivery_per_min'], 1)} [{pc(lv['delivery_ci'][0], 1)}, {pc(lv['delivery_ci'][1], 1)}].</p>
<div class="tbl"><table><thead><tr><th>등급</th><th>매장</th><th>하반기 GMV 비중</th><th>주간 주문</th><th>재주문 비중</th><th>평균 배달</th><th>운영 방향</th></tr></thead><tbody>{tier_rows}</tbody></table></div>
<p class="note">성장 기회: 배달 36분 초과 {op['slow_n']:,}곳을 32분으로 → 하반기 GMV 약 {won(op['gain_delivery_h2'])} 상당 / 평점 4.3 미만 {op['low_rating_n']:,}곳을 4.3으로 → 약 {won(op['gain_rating_h2'])} 상당.</p></section>

<section><h2>2. 쿠폰 증분: 기존 고객 활성화와 신규 고객 획득</h2>
<div class="two"><div>
<h3>재주문 쿠폰(3,000원) 고객 홀드아웃 · 27~38주차</h3>
<p>고객 90,000명 중 15%를 무작위로 빼고, 나머지에게 마지막 주문 후 3주가 지나면 쿠폰을 보냈습니다. 쿠폰이 쓰인 주문을 모두 효과로 치면 {allr['redemptions']:,}건이지만, 홀드아웃과 비교한 실제 증분은 {allr['incremental']:,.0f}건입니다. 주 1회 이상 주문 고객에게는 효과가 0과 구분되지 않습니다.</p>
</div><div><img src="img/ch2_cost_per_incremental.png" alt="고객군별 증분 주문 1건당 쿠폰 비용"></div></div>
<div class="tbl"><table><thead><tr><th>고객군</th><th>고객당 주문 차이(12주)</th><th>쿠폰 사용 주문</th><th>증분 주문</th><th>증분 비중</th><th>증분 주문당 비용</th></tr></thead><tbody>{seg_rows}</tbody></table></div>
<h3>고객 기준 4주 재주문율 (주문 로그에서 SQL로 계산)</h3>
<div class="two"><div><p>각 주문 뒤 같은 고객의 다음 주문을 <code>LEAD(week) OVER (PARTITION BY customer_id ORDER BY week)</code>로 찾아, 4주 안에 다시 주문한 비율을 냈습니다(4주를 지켜볼 수 있는 34주차까지의 주문만). KPI 트리의 '재주문 비중'(주문 중 재주문의 몫)과 달리, 고객이 돌아오는 속도를 직접 봅니다.
쿠폰이 4주 재주문율을 가장 많이 올린 곳도 휴면 고객입니다.</p></div>
<div class="tbl"><table><thead><tr><th>고객군</th><th>홀드아웃</th><th>쿠폰</th></tr></thead><tbody>{rep_rows}</tbody></table></div></div>
<h3>첫 주문 쿠폰(5,000원) 지역 스위치백 · 40~47주차</h3>
<p>세 지역에서 2주 단위로 쿠폰을 켜고 껐습니다(지역·주 고정효과). 쿠폰이 켜진 주에 신규 고객이 지역·주당 {fo['extra_per_area_week']:,.0f}명({pc(fo['lift'], 0, True)}) 늘었습니다. 다만 쿠폰으로 늘어난 고객의 8주 재주문율은 {pc(fo['reorder_marginal'])}로 자연 유입({pc(fo['reorder_organic'])})보다 낮아,
신규 고객 1명당 {won(fo['cost_per_incr_new'])}, 다시 주문한 신규 고객 1명당 {won(fo['cost_per_incr_retained'])}이 듭니다.</p>
<p class="note">관측치가 지역·주 {fo['n_obs']}개뿐이라 회귀 표준오차를 그대로 믿기 어렵습니다. 그래서 '지역마다 4개 블록 중 2개를 켜는' 가능한 일정 {fo['n_permutations']}개(세 지역이 같은 일정인 경우 제외)로 효과를 다시 계산한 무작위화 추론을 함께 보고합니다: p = {fo['p_permutation']:.3f}.</p></section>

<section><h2>3. 카테고리별 할인 반응과 캠페인 예산</h2>
<div class="two"><div>
<p>매장이 스스로 건 할인으로 효과를 재면 반응이 작게 나옵니다. 매장은 주문이 줄어든 주에 할인을 올리기 때문입니다(함정 1). 그래서 지역×카테고리 단위로 무작위 배정된 플랫폼 캠페인 쿠폰(0·5·10%)만으로 반응을 추정했습니다(매장·지역×주 고정효과).</p>
<p>같은 1원이라도 {top.category}는 {bot.category}보다 약 {top.orders_per_1m_won / bot.orders_per_1m_won:.1f}배 많은 주문을 만듭니다. 44~45주차 캠페인에서 하위 3개 카테고리 예산의 절반을 옮기면 상위 카테고리 평균 할인율이 {pc(ra['rate_top_before'], 1)}에서 약 {pc(ra['rate_top_before'] * (1 + ra['wanted'] / (ra['spend_moved'] / (ra['rate_top_after'] / ra['rate_top_before'] - 1))), 1)}로 올라 실험 범위(0~10%)를 벗어납니다.
그래서 10%에 닿을 때까지만({pc(ra['share_of_bottom_moved'])}, {won(ra['spend_moved'])}) 옮기고, 받는 쪽 효율은 할인율이 오른 구간의 중간값으로 계산했습니다: 같은 예산으로 주문 {ra['orders_delta']:,.0f}건({pc(ra['rel'], 1, True)}).</p>
</div><div><img src="img/ch3_orders_per_won.png" alt="카테고리별 쿠폰 100만 원당 증분 주문"></div></div>
<div class="tbl"><table><thead><tr><th>카테고리</th><th>매장 자체 할인으로 잰 반응</th><th>캠페인 쿠폰으로 잰 반응 [95%]</th><th>객단가</th><th>100만 원당 주문</th></tr></thead><tbody>{cat_rows}</tbody></table></div>
<p class="note">반응 = 할인율 1당 로그 주문 변화(할인 10%p → 주문 약 e^(0.1×반응)배). 서울 서부 30주차 이후(비용 분담 시범)는 추정에서 뺐습니다.</p></section>

<section><h2>4. 비용 분담 시범: 플랫폼이 매장 할인을 50% 매칭하면</h2>
<div class="two"><div>
<p>서울 서부 성장 등급 매장({c4['n_treated']}곳)에 30주차부터 매장 할인의 50%(최대 5%p)를 플랫폼이 부담했습니다. 다른 지역 성장 등급 매장과의 이중차분(매장·주 고정효과, 매장 단위 군집 표준오차):
할인 참여 {did['disc_on']['est'] * 100:+.1f}%p [{did['disc_on']['ci'][0] * 100:.1f}, {did['disc_on']['ci'][1] * 100:.1f}], 주문 {pc(did['lo']['est'], 1, True)} [{pc(did['lo']['ci'][0], 1)}, {pc(did['lo']['ci'][1], 1)}].</p>
<p>참여는 시작 후 8~12주에 걸쳐 늘었고, 시작 전에는 두 집단의 차이가 0 근처였습니다. 하지만 늘어난 주문 약 {c4['extra_orders']:,.0f}건에 플랫폼 할인 {won(c4['platform_cost'])}이 들어, 증분 주문 1건당 {c4['cost_per_extra_order']:,.0f}원입니다. 증분 주문 가치가 6,000원 이상이어야 지금 비율로 남으므로, 매칭 30% 셀을 추가해 비용 곡선을 확인한 뒤 확대를 권합니다.</p>
</div><div><img src="img/ch4_event_study.png" alt="비용 분담 시범 전후 할인 참여 차이"></div></div></section>

<section><h2>5. 분기 매출 예측과 시뮬레이션</h2>
<div class="two"><div>
<p>매장별로 할인을 걷어낸 수요의 추세를 잡고, 명절 효과와 이미 정해진 캠페인 일정 × 카테고리 반응을 더해 주 단위로 합산했습니다. 추세 기간은 Q3 백테스트에서 고른 {c5['window']}주를 Q4에 그대로 적용했습니다(시험 기간을 보고 고르지 않음).</p>
<p>평소 주 오차는 {pc(c5['mape_normal_bu'], 1)}(직전 13주 평균 {pc(c5['mape_normal_naive'], 1)})로 비슷하지만, 추석·캠페인 주 오차는 {pc(c5['mape_event_bu'], 1)}로 단순 평균({pc(c5['mape_event_naive'], 1)})보다 훨씬 작습니다. 40~50주 합계 오차는 {pc(c5['q4_error_w40_50'], 1, True)}.
연말 급증은 학습 기간에 없던 패턴이라 {pc(c5['year_end_miss'], 0)} 놓쳤습니다(함정 5). 작년 데이터가 쌓이면 연간 계절성을 넣어야 합니다.</p>
</div><div><img src="img/ch5_forecast.png" alt="Q4 주간 주문 실제와 예측"></div></div></section>

<section><h2>참값 대조</h2>
<p class="note">시뮬레이션 데이터라 모든 추정치를 생성에 쓴 참값과 비교할 수 있습니다. 분석 코드는 data/raw만 읽고, 참값(data/truth)은 이 표에서만 씁니다.</p>
<div class="tbl"><table><thead><tr><th>점검</th><th>추정</th><th>참값</th><th>결과</th></tr></thead><tbody>{val_rows}</tbody></table></div></section>

<footer>가상 데이터·가상 브랜드로 만든 포트폴리오입니다. 실제 회사의 데이터·화면·수치와 무관합니다. 파이프라인: Python 시뮬레이션 → DuckDB SQL 마트 → statsmodels(고정효과 회귀·이중차분·사건 연구) → 검증 → 이 페이지.
작성: 성유섭 · <a href="{REPO}">{REPO}</a></footer>
</div></body></html>"""
    (SITE / "index.html").write_text(html, encoding="utf-8")
    return memo


def summary(c0, c1, c2, c3, c4, c5, c6, val):
    y = c0["year"]; o2 = c2["targeted"]["option2"]; fo = c2["first_order"]; allr = next(r for r in c2["reorder"] if r["segment"] == "전체")
    cats = pd.DataFrame(c3["categories"]); top, bot = cats.iloc[0], cats.iloc[-1]; ra = c3["realloc"]; did = c4["did"]; lv, op = c1["levers"], c1["opportunity"]
    tiles = [
        ("쿠폰 사용 주문 중 실제 증분", pc(allr["incr_share"]), f"{allr['redemptions']:,}건 중 {allr['incremental']:,.0f}건 (홀드아웃 대비)"),
        ("두 권고의 분기 효과", won(c6["base"]["total_value"]), f"전 고객 확대안 대비 · 증분 주문 가치 3,000~7,500원이면 {won(min(x['total_value'] for x in c6['sensitivity'])).replace(' 원', '')}~{won(max(x['total_value'] for x in c6['sensitivity']))}"),
        ("카테고리별 쿠폰 효율", f"{top.orders_per_1m_won / bot.orders_per_1m_won:.1f}배", f"100만 원당 {top.category} {top.orders_per_1m_won:,.0f}건 vs {bot.category} {bot.orders_per_1m_won:,.0f}건"),
        ("같은 예산 재배분", pc(ra["rel"], 1, True), "캠페인 주문 · 실험 범위(할인 10%) 안에서만 이동"),
        ("행사 주 예측 오차", pc(c5["mape_event_bu"], 1), f"직전 13주 평균 {pc(c5['mape_event_naive'], 1)} · 40~50주 합계 {pc(c5['q4_error_w40_50'], 1, True)}"),
        ("참값 대조", f"{val['passed']} / {val['total']}", "모든 추정치를 생성 참값과 비교"),
    ]
    chapters = [
        ("0 KPI", "무엇이 GMV·순수익을 움직이나", "KPI 트리(주문×객단가, 신규+재주문, 수수료−플랫폼 할인)", f"캠페인 주 주문 +15%, 주문당 순수익 {c0['campaign']['net_per_order']:,.0f}원(평소 {c0['campaign']['net_per_order_near']:,.0f}원)"),
        ("1 세분화", "어떤 매장에 무엇을", "주문 수준×성장 잠재력 4등급, 레버 회귀", f"평점 0.1점 {pc(lv['rating_per_0p1'], 1, True)}, 배달 1분 {pc(lv['delivery_per_min'], 1)}"),
        ("2 쿠폰 증분", "쿠폰이 실제로 늘린 주문은", "고객 홀드아웃, 지역 스위치백(무작위화 추론), 4주 재주문 SQL", f"신규 {pc(fo['lift'], 0, True)}, 쿠폰 유입 재주문 {pc(fo['reorder_marginal'])} vs {pc(fo['reorder_organic'])}"),
        ("3 할인 반응", "같은 1원의 효과는 어디서 큰가", "무작위 캠페인 셀 + 매장·지역×주 고정효과", f"매장 자체 할인으로 재면 과소 추정(함정)"),
        ("4 비용 분담", "매칭 정책은 남는가", "이중차분, 사건 연구", f"참여 {did['disc_on']['est'] * 100:+.1f}%p, 주문 {pc(did['lo']['est'], 1, True)}, 증분 주문당 {c4['cost_per_extra_order']:,.0f}원"),
        ("5 예측", "다음 분기 얼마나", "바텀업 추세+명절+캠페인×반응, Q3로 기간 선택", f"연말 급증 {pc(c5['year_end_miss'], 0)} 놓침 → 연간 계절성 필요"),
    ]
    rows = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in chapters)
    tl = "".join(f"<div class='t'><span>{a}</span><b>{b}</b><em>{c}</em></div>" for a, b, c in tiles)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>SPOONLY 포트폴리오 요약</title><style>
@page{{size:A4;margin:12mm 13mm}}body{{font-family:"Malgun Gothic",sans-serif;color:#0b0b0b;font-size:10.2px;line-height:1.5;margin:0}}
h1{{font-size:17px;margin:0 0 3px}}.sub{{color:#52514e;font-size:10px;margin-bottom:8px}}h2{{font-size:12px;margin:10px 0 4px;color:#1d4f8c}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}.t{{border:1px solid #e4e3df;border-radius:6px;padding:6px 8px}}.t span{{display:block;color:#52514e;font-size:9.2px}}.t b{{display:block;font-size:15px;line-height:1.25}}.t em{{font-style:normal;color:#52514e;font-size:9px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #e4e3df;padding:4px 5px;text-align:left;vertical-align:top}}td:first-child{{white-space:nowrap}}th{{color:#52514e;font-size:9.2px}}
ol{{margin:3px 0;padding-left:16px}}li{{margin:2px 0}}.memo{{border-left:3px solid #2a78d6;padding-left:9px}}.foot{{color:#52514e;font-size:9px;margin-top:8px}}
img{{width:100%}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
</style></head><body>
<h1>SPOONLY 입점 매장 성장 분석 — 포트폴리오 요약</h1>
<div class="sub">성유섭 · 가상의 배달앱(매장 {y['merchants']:,}곳 · 주문 {y['orders'] / 1e6:.1f}백만 건 · 52주) · SQL(DuckDB)·Python · 모든 데이터 가상 · {DEMO}</div>
<div class="grid">{tl}</div>
<h2>경영진 메모: 쿠폰은 '쓰인 건수'가 아니라 '늘어난 주문'으로 배분한다</h2>
<div class="memo"><ol>
<li>재주문 쿠폰은 전 고객 확대 대신 휴면 고객에게만(증분 주문당 {c2['breakpoints']['휴면(60일+)']:,.0f}원). 월 1회 이하({c2['breakpoints']['월 1회 이하']:,.0f}원)는 증분 주문 가치가 그 이상일 때만.</li>
<li>캠페인 예산을 {top.category}·{cats.iloc[1].category} 등 반응이 큰 카테고리로, 실험한 할인 범위 안에서: 같은 예산으로 캠페인 주문 {pc(ra['rel'], 1, True)}.</li>
<li>첫 주문 쿠폰은 '다시 주문한 신규 고객당 비용'(약 {won(fo['cost_per_incr_retained'])})으로 관리.</li>
<li>비용 분담 50% 매칭은 참여·주문을 늘렸지만 증분 주문당 {c4['cost_per_extra_order']:,.0f}원(가치 6,000원 이상일 때만 남음) → 30% 매칭 셀로 다시 시험 후 확대.</li>
<li>배달 36분 초과 {op['slow_n']:,}곳의 배달 시간 개선이 쿠폰보다 큰 기회(하반기 GMV 약 {won(op['gain_delivery_h2'])} 상당).</li>
</ol></div>
<div class="two"><img src="img/ch3_orders_per_won.png"><img src="img/ch5_forecast.png"></div>
<h2>데이터에 넣어 둔 함정 5가지 (분석이 찾아야 하는 것)</h2>
<ol><li>매장은 주문이 줄어든 주에 할인을 올린다 → 매장 자체 할인으로 재면 할인 효과가 작게 보임</li>
<li>재주문 쿠폰 사용 주문 대부분은 어차피 할 주문 → 쿠폰 사용 건수로 재면 효과가 약 3배 부풀려짐</li>
<li>첫 주문 쿠폰으로 늘어난 신규 고객은 덜 재주문한다 → 신규 고객 수만 보면 과대평가</li>
<li>서울 서부 성장 등급 매장에만 30주차부터 비용 분담 시범 → 그대로 섞으면 할인 반응 추정이 오염됨</li>
<li>연말 급증은 학습 기간에 없다 → 어떤 모형도 맞힐 수 없는 패턴을 구분해 보고</li></ol>
<h2>장별 질문 · 방법 · 결과</h2>
<table><thead><tr><th>장</th><th>질문</th><th>방법</th><th>결과</th></tr></thead><tbody>{rows}</tbody></table>
<div class="foot">실제 경험과의 연결: Shopee에서 셀러 50곳 이상을 6개 기준으로 등급화하고, 가격·바우처 탄력성으로 행사 할인율을 조정했으며, 회귀·MMM 기반 바텀업 예측으로 분기 매출을 예측했습니다. 이 포트폴리오는 같은 방법을 배달앱 입점 매장에 옮겨 본 가상 케이스입니다. 코드: {REPO}</div>
</body></html>"""
    (SITE / "summary.html").write_text(html, encoding="utf-8")
    pdf = SITE / "SPOONLY_portfolio_summary.pdf"
    exe = next((b for b in BROWSERS if Path(b).exists()), None)
    if exe:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", (SITE / "summary.html").as_uri()],
                       check=True, capture_output=True, timeout=120)
        shutil.copy(pdf, COPY_TO)
        print(f"{pdf.name}: {pdf.stat().st_size / 1024:.0f} KB, copied to {COPY_TO.name}")


def main():
    c0, c1, c2, c3, c4, c5, val = (j(n) for n in ["ch0_kpi.json", "ch1_segments.json", "ch2_coupons.json", "ch3_elasticity.json",
                                                  "ch4_pilot.json", "ch5_forecast.json", "validation.json"])
    c6 = j("ch6_impact.json")
    charts(c2, c3, c4, c5)
    page(c0, c1, c2, c3, c4, c5, c6, val)
    summary(c0, c1, c2, c3, c4, c5, c6, val)
    print("site/index.html, site/summary.html written")


if __name__ == "__main__":
    main()
