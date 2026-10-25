"""Интерактивный отчёт → docs/index.html. Читает только results/*.csv — исходные данные не нужны.

    python scripts/report.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.stdout.reconfigure(encoding="utf-8")

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
CONFIG = {"displaylogo": False, "locale": "ru", "responsive": True}
NB = " "


def fmt(x, d=3):
    return f"{x:,.{d}f}".replace(",", NB).replace(".", ",").replace("-", "−")


def sfmt(x, d=3):
    return ("+" if x > 0 else "") + fmt(x, d)


def pct(x, d=0, sign=False):
    s = f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"
    return s.replace(".", ",").replace("-", "−") + NB + "%"


def base(fig, title, h=380, x="", y="", yfmt=None):
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK), x=0.01, xanchor="left", y=0.985,
                                 yanchor="top"),
                      height=h, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, separators=", ",
                      font=dict(family=FONT, size=12, color=INK2), margin=dict(l=60, r=16, t=84, b=48),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#fff", bordercolor=AXIS, font=dict(color=INK, family=FONT)))
    fig.update_xaxes(title=x, showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(title=y, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), tickformat=yfmt)
    return fig


def html(fig, key):
    return '<div class="chart">' + fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                                               div_id=f"fig-{key}") + "</div>"


def save_images(figs: dict) -> None:
    """Статичные PNG для README (kaleido не понимает не-ASCII путь к plotly.js — кладём копию во временную папку)."""
    import os
    import shutil
    import tempfile

    import plotly
    import plotly.io as pio
    js = os.path.join(tempfile.gettempdir(), f"plotly-{plotly.__version__}.min.js")
    shutil.copy(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"), js)
    pio.kaleido.scope.plotlyjs = js
    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    for k, f in figs.items():
        f.write_image(out / f"{k}.png", width=1000, scale=1.5)


def main():
    facts = pl.read_csv(RES / "eda_facts.csv").row(0, named=True)
    test = pl.read_csv(RES / "test.csv")
    abl = pl.read_csv(RES / "ablation_val.csv")
    leak = pl.read_csv(RES / "leakage_val.csv")
    imp = pl.read_csv(RES / "importance_test.csv")
    peru = pl.read_csv(RES / "test_peruser.csv")
    tail = pl.read_csv(RES / "eda_longtail.csv")
    played = pl.read_csv(RES / "eda_played.csv")
    figs = {}

    # --- данные
    f = go.Figure(go.Scatter(x=tail["доля треков"], y=tail["доля откликов"], mode="lines", line=dict(color=BLUE, width=2.5),
                             hovertemplate="топ-%{x:.2%} треков дают %{y:.0%} откликов<extra></extra>"))
    t1 = tail.filter(pl.col("доля треков") >= 0.01).row(0, named=True)
    f.add_annotation(x=np.log10(t1["доля треков"]), y=t1["доля откликов"], text=f"1 % треков → {pct(t1['доля откликов'])} откликов",
                     showarrow=True, arrowhead=0, ax=60, ay=30, font=dict(color=INK2))
    figs["tail"] = base(f, "Длинный хвост: какая доля положительных откликов приходится на самые популярные треки", h=380,
                        x="доля самых популярных треков (лог. шкала)", y="накопленная доля откликов", yfmt=".0%")
    figs["tail"].update_xaxes(type="log", tickformat=".1%")

    p = played.with_columns(pl.col("n") / pl.col("n").sum().over("is_organic"))
    f = go.Figure()
    for org, name, c in [(1, "сам нашёл", GREEN), (0, "из рекомендаций", ORANGE)]:
        d = p.filter(pl.col("is_organic") == org).sort("pct")
        f.add_trace(go.Scatter(x=d["pct"], y=d["n"], mode="lines", name=name, line=dict(color=c, width=2),
                               hovertemplate="%{x} %: %{y:.1%}<extra>" + name + "</extra>"))
    figs["played"] = base(f, "Сколько процентов трека дослушивают: пики на 0 % (пропуск) и 100 %", h=360,
                          x="прослушано, % трека (больше 100 — перематывали и слушали снова)", y="доля прослушиваний", yfmt=".0%")
    figs["played"].update_yaxes(type="log")

    # --- модели на тесте
    def models_chart(task):
        d = test.filter(pl.col("задача") == task).sort("ndcg@10")
        best = d["лучший простой"][0]
        col = [BLUE if m == "ранкер" else (ORANGE if m == best else AXIS) for m in d["модель"]]
        g = go.Figure(go.Bar(x=d["ndcg@10"], y=d["модель"], orientation="h", marker=dict(color=col, cornerradius=4),
                             text=[fmt(v) for v in d["ndcg@10"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: NDCG@10 = %{x:.4f}<extra></extra>"))
        base(g, f"«{task}»: NDCG@10 на тестовой неделе", h=300)
        g.update_xaxes(range=[0, d["ndcg@10"].max() * 1.2], showgrid=True, gridcolor=GRID)
        g.update_yaxes(automargin=True, showgrid=False)
        return g

    figs["m_all"] = models_chart("прослушает")
    figs["m_new"] = models_chart("откроет")

    # органика
    f = make_subplots(rows=1, cols=2, subplot_titles=["прослушает", "откроет"], horizontal_spacing=0.1)
    for j, task in enumerate(["прослушает", "откроет"], start=1):
        d = test.filter((pl.col("задача") == task) & pl.col("модель").is_in(["ранкер", "ALS", "ещё раз своё", "исполнители",
                                                                             "популярное"]))
        for col, name, c in [("ndcg@10 сам нашёл", "трек нашли сами", GREEN), ("ndcg@10 из рекомендаций", "трек пришёл из рекомендаций", ORANGE)]:
            f.add_trace(go.Bar(x=d["модель"], y=d[col], name=name, marker=dict(color=c, cornerradius=3), showlegend=j == 1,
                               hovertemplate="%{x}: %{y:.4f}<extra>" + name + "</extra>"), 1, j)
    base(f, "Качество отдельно по откликам на треки, найденные самостоятельно и пришедшие из рекомендаций", h=400, y="NDCG@10")
    f.update_layout(barmode="group", margin=dict(t=110), legend=dict(y=1.1))
    figs["organic"] = f

    # шире точности
    f = go.Figure()
    for task, sym in [("прослушает", "circle"), ("откроет", "diamond")]:
        d = test.filter(pl.col("задача") == task)
        f.add_trace(go.Scatter(x=d["coverage@10"], y=d["ndcg@10"], mode="markers+text", text=d["модель"],
                               textposition="top center", name=task, marker=dict(size=11, symbol=sym,
                                                                                 color=BLUE if task == "прослушает" else ORANGE),
                               customdata=d["head@10"],
                               hovertemplate="%{text}<br>покрытие %{x:.2%}<br>NDCG@10 %{y:.4f}<br>хиты в топ-10: %{customdata:.0%}<extra></extra>"))
    figs["beyond"] = base(f, "Точность против разнообразия: покрытие каталога и NDCG@10", h=420,
                          x="покрытие каталога топ-10 (доля треков, попавших хоть кому-то)", y="NDCG@10")
    tv = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2]
    figs["beyond"].update_xaxes(type="log", tickvals=tv, ticktext=["0,001 %", "0,01 %", "0,1 %", "1 %", "3 %"])
    figs["beyond"].update_traces(textfont=dict(size=10, color=INK2))

    # абляция
    f = make_subplots(rows=1, cols=2, subplot_titles=["прослушает", "откроет"], horizontal_spacing=0.18)
    for j, task in enumerate(["прослушает", "откроет"], start=1):
        d = abl.filter(pl.col("задача") == task).sort("Δ к полному")
        f.add_trace(go.Scatter(x=d["Δ к полному"], y=d["без источника"], mode="markers", marker=dict(size=10, color=BLUE),
                               error_x=dict(type="data", symmetric=False, array=d["ДИ до"] - d["Δ к полному"],
                                            arrayminus=d["Δ к полному"] - d["ДИ от"], color=BLUE),
                               showlegend=False, hovertemplate="без «%{y}»: %{x:+.4f}<extra></extra>"), 1, j)
        f.add_vline(x=0, line=dict(color=AXIS), row=1, col=j)
    base(f, "Абляция на валидации: насколько падает NDCG@10 ранкера без каждого источника кандидатов (95 % ДИ)", h=360,
         x="изменение NDCG@10")
    f.update_yaxes(automargin=True)
    figs["abl"] = f

    # утечка
    f = go.Figure()
    for split, c in [("по времени", BLUE), ("случайно", ORANGE)]:
        d = leak.filter(pl.col("разбиение") == split)
        f.add_trace(go.Bar(x=d["задача"], y=d["recall@100"], name=f"разбиение {split}", marker=dict(color=c, cornerradius=4),
                           text=[fmt(v) for v in d["recall@100"]], textposition="outside"))
    figs["leak"] = base(f, "ALS: Recall@100 при честном разбиении по времени и при случайном", h=360, y="Recall@100")
    figs["leak"].update_layout(barmode="group")

    # важность признаков
    f = make_subplots(rows=1, cols=2, subplot_titles=["прослушает", "откроет"], horizontal_spacing=0.22)
    for j, task in enumerate(["прослушает", "откроет"], start=1):
        d = imp.filter(pl.col("задача") == task).sort("gain", descending=True).head(12).reverse()
        f.add_trace(go.Bar(x=d["gain"], y=d["feature"], orientation="h", marker=dict(color=BLUE, cornerradius=3),
                           showlegend=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"), j and 1, j)
    base(f, "Главные признаки ранкера (доля прироста качества)", h=460)
    f.update_xaxes(tickformat=".0%")
    figs["imp"] = f

    # выигрыш по активности пользователя
    f = go.Figure()
    for task, c in [("прослушает", BLUE), ("откроет", ORANGE)]:
        d = peru.filter(pl.col("задача") == task).with_columns(
            pl.col("история").qcut(5, labels=["1 (мало)", "2", "3", "4", "5 (много)"]).alias("квинтиль"))
        g = d.group_by("квинтиль").agg((pl.col("ранкер").mean() - pl.col("лучший простой").mean()).alias("Δ")).sort("квинтиль")
        f.add_trace(go.Scatter(x=g["квинтиль"].cast(pl.Utf8), y=g["Δ"], mode="lines+markers", name=task,
                               line=dict(color=c, width=2), marker=dict(size=8)))
    f.add_hline(y=0, line=dict(color=AXIS))
    figs["seg"] = base(f, "Выигрыш ранкера над лучшим простым решением по квинтилям активности пользователя", h=360,
                       x="квинтиль по числу событий в истории", y="Δ NDCG@10")

    # --- текст
    T = {t: test.filter(pl.col("задача") == t) for t in ["прослушает", "откроет"]}

    def row(task, model):
        return T[task].filter(pl.col("модель") == model).row(0, named=True)

    ra, rn = row("прослушает", "ранкер"), row("откроет", "ранкер")
    ba, bn = row("прослушает", ra["лучший простой"]), row("откроет", rn["лучший простой"])
    def ratio_text(r, b):
        g_self = r["ndcg@10 сам нашёл"] / max(b["ndcg@10 сам нашёл"], 1e-9) - 1
        g_rec = r["ndcg@10 из рекомендаций"] / max(b["ndcg@10 из рекомендаций"], 1e-9) - 1
        if g_self < 0.05:
            return (f"Почти весь выигрыш ({pct(g_rec, 0, True)}) — на треках, которые показывала старая система: ранкер "
                    f"во многом учится предсказывать прошлые рекомендации, а на самостоятельных находках не лучше простого правила.")
        return (f"Выигрыш есть на обоих типах: {pct(g_self, 0, True)} на самостоятельных находках и {pct(g_rec, 0, True)} "
                f"на треках из рекомендаций — это не только эхо старой системы.")

    bias_all, bias_new = ratio_text(ra, ba), ratio_text(rn, bn)
    ab = {t: abl.filter(pl.col("задача") == t) for t in ["прослушает", "откроет"]}
    needed = {t: [r["без источника"] for r in ab[t].iter_rows(named=True) if r["ДИ до"] < 0] for t in ab}
    useless = {t: [r["без источника"] for r in ab[t].iter_rows(named=True) if r["ДИ от"] <= 0 <= r["ДИ до"]] for t in ab}
    la = leak.filter(pl.col("задача") == "откроет")
    lr = la.filter(pl.col("разбиение") == "случайно")["recall@100"][0] / la.filter(pl.col("разбиение") == "по времени")["recall@100"][0]

    sections = [
        ("data", "1. Данные", f"""
<p><a href="https://huggingface.co/datasets/yandex/yambda">Yambda-50M</a> — открытый датасет Яндекс Музыки:
{fmt(facts['событий'], 0)} событий {fmt(facts['пользователей'], 0)} пользователей с {fmt(facts['треков'], 0)} треками за
{fmt(facts['дней'], 0)} дней — прослушивания с долей прослушанного, лайки и дизлайки. У каждого события есть флаг: нашёл
пользователь трек сам или трек показала рекомендательная система ({pct(facts['доля из рекомендаций'])} прослушиваний).</p>
{html(figs['tail'], 'tail')}
<p><strong>{pct(facts['доля повторных прослушиваний'])} прослушиваний — повторы уже знакомых треков.</strong> Поэтому
оцениваются две задачи: «прослушает» — любые треки, включая любимые, и «откроет» — только треки, которых пользователь
раньше не встречал. Во второй и состоит работа рекомендаций: напомнить любимое умеет и простое правило.</p>
{html(figs['played'], 'played')}
<p>Положительный отклик — трек прослушан больше чем наполовину (определение авторов датасета) или лайкнут.
{pct(facts['доля пропусков (≤10 %)'])} прослушиваний обрываются на первых 10 % трека.</p>"""),
        ("method", "2. Как проверяем", """
<p>План записан до обучения моделей (<code>PLAN.md</code>). Разбиение только по времени: модели учатся на истории,
проверяются на следующей неделе. Ранкер учится на одной неделе, подбирается на следующей (валидация), а финальные цифры
ниже — на последней неделе (тест), которая открывалась один раз. Метрика — NDCG@10; разницы моделей — 95 %
доверительные интервалы бутстрепом по пользователям (парные: одни и те же пользователи).</p>
<p><strong>Двухэтапная схема, как в индустрии:</strong> несколько источников кандидатов (популярное, «ещё раз своё»,
треки любимых исполнителей, «слушают вместе», ALS) дают около 500 треков на пользователя, затем LightGBM
(LambdaRank) переранжирует их по признакам трека, пользователя и пары.</p>"""),
        ("results", "3. Результат на тестовой неделе", f"""
{html(figs['m_all'], 'm_all')}
{html(figs['m_new'], 'm_new')}
<div class="check"><b>«Прослушает»:</b> ранкер {fmt(ra['ndcg@10'])} против «{ra['лучший простой']}» {fmt(ba['ndcg@10'])}:
{sfmt(ra['Δ ndcg@10'])} (95 % ДИ [{sfmt(ra['ДИ от'])}; {sfmt(ra['ДИ до'])}]).<br>
<b>«Откроет»:</b> ранкер {fmt(rn['ndcg@10'])} против {rn['лучший простой']} {fmt(bn['ndcg@10'])}:
{sfmt(rn['Δ ndcg@10'])} [{sfmt(rn['ДИ от'])}; {sfmt(rn['ДИ до'])}], то есть {pct(rn['ndcg@10'] / bn['ndcg@10'] - 1, 0, True)}.
В кандидаты попадает {pct(ra['полнота кандидатов'])} и {pct(rn['полнота кандидатов'])} будущих откликов — это потолок,
выше которого ранкер подняться не может.</div>
{html(figs['seg'], 'seg')}"""),
        ("bias", "4. Смещение обратной связи", f"""
<p>Модели учатся на логах, которые сами частично создали: половина откликов — на треки, показанные прошлой
рекомендательной системой. Если модель лишь воспроизводит старые рекомендации, она выиграет на откликах «из рекомендаций»
и проиграет на треках, которые люди находят сами.</p>
{html(figs['organic'], 'organic')}
<div class="check"><b>«Прослушает»:</b> на треках, найденных самостоятельно, ранкер {fmt(ra['ndcg@10 сам нашёл'])}
против «{ra['лучший простой']}» {fmt(ba['ndcg@10 сам нашёл'])}, на пришедших из рекомендаций — {fmt(ra['ndcg@10 из рекомендаций'])}
против {fmt(ba['ndcg@10 из рекомендаций'])}. {bias_all}<br>
<b>«Откроет»:</b> сам нашёл — {fmt(rn['ndcg@10 сам нашёл'])} против {fmt(bn['ndcg@10 сам нашёл'])} у {rn['лучший простой']},
из рекомендаций — {fmt(rn['ndcg@10 из рекомендаций'])} против {fmt(bn['ndcg@10 из рекомендаций'])}. {bias_new}</div>"""),
        ("beyond", "5. Шире точности", f"""
{html(figs['beyond'], 'beyond')}
<p>Популярное почти ничего не покрывает: всем показывается один и тот же топ. Ранкер в задаче «откроет» показывает
{pct(rn['coverage@10'], 1)} каталога, а доля «хитов» (топ-1 % треков) в его топ-10 — {pct(rn['head@10'])}.</p>
{html(figs['abl'], 'abl')}
<p><strong>Абляция:</strong> значимо нужны только «{'», «'.join(sorted(set(needed['прослушает'] + needed['откроет'])))}»
— без них качество падает. Источники «{'», «'.join(sorted(set(useless['прослушает']) & set(useless['откроет'])))}»
сверх них ничего не добавляют (интервал накрывает ноль): систему можно упростить, а вклад исполнителей сохраняется
через признаки пары «пользователь — исполнитель».</p>
{html(figs['imp'], 'imp')}"""),
        ("leak", "6. Ловушка: случайное разбиение", f"""
<p>Частая ошибка — делить взаимодействия на обучение и проверку случайно: модель видит будущие прослушивания соседей
и «предсказывает» то, что уже знает.</p>
{html(figs['leak'], 'leak')}
<p>Та же ALS со случайным разбиением показывает Recall@100 в {fmt(lr, 1)} раза выше в задаче «откроет». Эффект меньше,
чем часто пишут, но достаточно, чтобы выбрать не ту модель.</p>"""),
        ("decision", "7. Что нести в A/B-тест", f"""
<ul>
<li><strong>Что:</strong> ранкер для блока открытий (задача «откроет») — значимо лучше лучшего простого решения
({pct(rn['ndcg@10'] / bn['ndcg@10'] - 1, 0, True)} NDCG@10), и выигрыш есть на треках, которые люди находят сами.
Для «ещё раз любимое» хватает простого правила: выигрыш ранкера там в основном на предсказании старых рекомендаций.</li>
<li><strong>Главная метрика теста:</strong> число новых для пользователя треков, прослушанных больше чем наполовину,
на пользователя в неделю.</li>
<li><strong>Защитные метрики:</strong> доля пропусков и дизлайков в блоке, время прослушивания в день, возвращаемость
через неделю, покрытие каталога (не должны скатиться в «одни хиты»).</li>
<li><strong>Упрощение до теста:</strong> оставить кандидатов из ALS и «ещё раз своё» — абляция показывает, что
остальные источники не добавляют качества, а время на подготовку кандидатов сокращается.</li>
<li><strong>Чего офлайн не скажет:</strong> офлайн-метрики считаются по логам старой системы; реальный эффект новых
рекомендаций покажет только эксперимент.</li>
</ul>"""),
    ]
    toc = "".join(f'<a href="#{i}">{t}</a>' for i, t, _ in sections)
    body = "".join(f'<section id="{i}"><h2>{t}</h2>{b}</section>' for i, t, b in sections)
    page = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Рекомендации музыки</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly-locale-ru.js"></script>
<style>
:root {{ color-scheme: light; }} * {{ box-sizing: border-box; }}
body {{ margin: 0; background: #f9f9f7; color: {INK}; font: 16px/1.6 {FONT}; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 72px; }}
h1 {{ font-size: clamp(26px, 4vw, 34px); line-height: 1.2; margin: 0 0 8px; }} h2 {{ font-size: 22px; margin: 0 0 12px; }}
section {{ margin-top: 40px; padding-top: 28px; border-top: 1px solid {GRID}; }}
p {{ max-width: 76ch; }} .lead {{ color: {INK2}; font-size: 17px; }} a {{ color: #1c5cab; }}
code {{ background: #efeee9; padding: 1px 5px; border-radius: 4px; font-size: 14px; }}
.chart {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 10px; padding: 6px; margin: 16px 0; overflow: hidden; }}
.check {{ background: {SURFACE}; border-left: 3px solid {BLUE}; border-radius: 6px; padding: 12px 16px; margin: 16px 0; max-width: 80ch; }}
nav.toc {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 20px; font-size: 14px; }} nav.toc a {{ text-decoration: none; }}
footer {{ margin-top: 48px; color: {MUTED}; font-size: 13px; }}
</style></head><body><main>
<h1>Рекомендации музыки: двухэтапная система на данных Яндекс Музыки</h1>
<p class="lead">Отбор кандидатов из пяти источников и ранжирование LightGBM на открытом датасете Yambda-50M.
Честная проверка по времени, две задачи — «прослушает» и «откроет», доверительные интервалы, смещение обратной связи.
Отчёт собран {dt.date.today().strftime('%d.%m.%Y')}; код — <a href="https://github.com/Alex91132/music-recsys">GitHub</a>.</p>
<nav class="toc">{toc}</nav>{body}
<footer>Данные: Yandex, Yambda (Apache 2.0; опубликованы для научных и исследовательских целей).</footer>
</main></body></html>"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")
    save_images({k: figs[k] for k in ("m_new", "organic", "abl")})
    print(f"docs/index.html: {len(page) / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
