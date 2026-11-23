"""
Імітаційне моделювання методики верифікації OSINT-повідомлень
(групування за походженням, зважене агрегування, правило утримання та вето).

Відтворення:  python3 osint_sim.py            -> results.json, fig_*.png
Усі параметри задано явно нижче; зерно генератора фіксоване.
"""
import json
import numpy as np

# ---------------------------------------------------------------------------
# Параметри первинних джерел (модельні припущення)
#   cov  - імовірність, що джерело спостерігає індикатор
#   acc  - імовірність правильного вердикту за умови спостереження
#   both - чи публікує джерело спростування (-1); False -> лише позитивні сигнали
#   w    - середня якість повідомлення w = q0 * t (оцінка аналітика за табл. 1)
# ---------------------------------------------------------------------------
PRIMARY = [
    # name,                         cov,  acc,  both,  w
    ("sandbox / технічна перевірка", 0.45, 0.93, True,  0.85),
    ("бюлетень CERT / аналітика",    0.25, 0.95, True,  0.80),
    ("спільнотна база",              0.55, 0.88, False, 0.65),
    ("підтвердження власника бренду",0.15, 0.97, True,  0.90),
    ("евристична автоматична стрічка",0.70, 0.75, False, 0.40),
    ("комерційна стрічка",           0.40, 0.87, False, 0.60),
]
# Імовірність, що агрегатор віддзеркалює певне первинне джерело
PARENT_P = np.array([0.05, 0.05, 0.25, 0.00, 0.45, 0.20])
COPY_P = 0.8            # частка повідомлень батьківського джерела, які копіює агрегатор
COPY_W = (0.50, 0.75)   # якість, яку аналітик приписує агрегатору (виглядає «солідно»)
W_SD = 0.05             # розкид оцінки якості окремого повідомлення
PREV = 0.4              # частка фішингових індикаторів серед кандидатів

TH_S, TH_VETO, MIN_GROUPS = 0.60, 0.70, 2


def simulate_reports(rng, n, n_copiers):
    """Повертає мітки та списки повідомлень (y, w, origin, is_copy) для кожного індикатора."""
    labels = rng.random(n) < PREV
    parents = rng.choice(len(PRIMARY), size=n_copiers, p=PARENT_P)
    cw = rng.uniform(*COPY_W, size=n_copiers)
    data = []
    for i in range(n):
        Y = labels[i]
        reps = []          # (y, w, origin_group_true, is_copy)
        prim = {}
        for k, (_, cov, acc, both, w) in enumerate(PRIMARY):
            if rng.random() >= cov:
                continue
            correct = rng.random() < acc
            verdict = (1 if Y else -1) if correct else (-1 if Y else 1)
            if verdict == -1 and not both:
                continue  # стрічка не публікує «чисті» записи
            ww = float(np.clip(w + rng.normal(0, W_SD), 0.05, 1.0))
            prim[k] = verdict
            reps.append((verdict, ww, k, False))
        for j in range(n_copiers):
            k = parents[j]
            if k in prim and rng.random() < COPY_P:
                ww = float(np.clip(cw[j] + rng.normal(0, W_SD), 0.05, 1.0))
                reps.append((prim[k], ww, k, True))
        data.append(reps)
    return labels, data


def decide(reps, mode, delta, rng):
    """mode: 'A' більшість, 'B' зважене без групування, 'C' повна методика, 'Cnv' без вето."""
    if mode == "A":
        npos = sum(1 for r in reps if r[0] == 1)
        nneg = sum(1 for r in reps if r[0] == -1)
        if npos >= MIN_GROUPS and npos > nneg:
            return 1
        if nneg >= MIN_GROUPS and nneg > npos:
            return -1
        return 0
    if mode == "B":
        units = [(y, w) for (y, w, _, _) in reps]
    else:
        groups = {}
        for idx, (y, w, origin, is_copy) in enumerate(reps):
            # копію об'єднують із групою першоджерела лише тоді, коли залежність виявлено
            key = origin if (not is_copy or rng.random() < delta) else ("solo", idx)
            if key not in groups or w > groups[key][1]:
                groups[key] = (y, w)
        units = list(groups.values())
    units = [u for u in units if u[0] != 0]
    if not units:
        return 0
    wp = sum(w for y, w in units if y == 1)
    wn = sum(w for y, w in units if y == -1)
    S = (wp - wn) / (wp + wn)
    npos = sum(1 for y, _ in units if y == 1)
    nneg = sum(1 for y, _ in units if y == -1)
    maxneg = max([w for y, w in units if y == -1], default=0.0)
    maxpos = max([w for y, w in units if y == 1], default=0.0)
    veto = mode != "Cnv"
    if npos >= MIN_GROUPS and S >= TH_S and not (veto and maxneg >= TH_VETO):
        return 1
    if nneg >= MIN_GROUPS and S <= -TH_S and not (veto and maxpos >= TH_VETO):
        return -1
    return 0


def metrics(labels, dec, strong_neg=None):
    labels = np.asarray(labels); dec = np.asarray(dec)
    P = labels.sum(); N = (~labels).sum()
    tp = int(((dec == 1) & labels).sum()); fp = int(((dec == 1) & ~labels).sum())
    fn_hard = int(((dec == -1) & labels).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    sn = np.asarray(strong_neg) & ~labels
    fpr_sn = float(((dec == 1) & sn).sum() / sn.sum()) if sn.sum() else np.nan
    return dict(precision=prec, recall=tp / P, fpr=fp / N,
                abstain=float((dec == 0).mean()), false_refute=fn_hard / P,
                abstain_phish=float(((dec == 0) & labels).sum() / P),
                fpr_strong_neg=fpr_sn)


def run(n=2000, reps=50, n_copiers=8, delta=1.0, modes=("A", "B", "C", "Cnv"), seed=2026):
    out = {m: [] for m in modes}
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        labels, data = simulate_reports(rng, n, n_copiers)
        for m in modes:
            rr = np.random.default_rng(seed * 7 + r)
            dec = [decide(x, m, delta, rr) for x in data]
            strong = [any(y == -1 and w >= TH_VETO for (y, w, _, _) in x) for x in data]
            out[m].append(metrics(labels, dec, strong))
    summary = {}
    for m, lst in out.items():
        summary[m] = {k: (float(np.nanmean([d[k] for d in lst])),
                          float(np.nanstd([d[k] for d in lst], ddof=1)))
                      for k in lst[0]}
    return summary


def avg_reports(n_copiers, seed=2026, n=2000):
    rng = np.random.default_rng(seed)
    labels, data = simulate_reports(rng, n, n_copiers)
    tot = np.array([len(x) for x in data]); cop = np.array([sum(r[3] for r in x) for x in data])
    return float(tot.mean()), float(cop.sum() / max(tot.sum(), 1))


def pct(x):
    """0.1234 -> '12,3' (українська десяткова кома)."""
    return f"{100 * x:.1f}".replace(".", ",")


def print_summary(res):
    print("\n=== Таблиця 5 статті (M = 8; середнє ± ст. відхилення), % ===")
    print(f"{'Підхід':<28}{'Влучність':>14}{'Повнота':>14}{'Хибнопоз.':>14}{'На ручну':>10}")
    rows = [("А – більшість", res["main"]["A"]), ("Б – без групування", res["main"]["B"]),
            ("В – методика, δ = 1", res["main"]["C"]), ("В – методика, δ = 0,5", res["main_d05"]["C"])]
    for name, r in rows:
        f = lambda k: f"{pct(r[k][0])} ± {pct(r[k][1])}"
        print(f"{name:<28}{f('precision'):>14}{f('recall'):>14}{f('fpr'):>14}{pct(r['abstain_phish'][0]):>10}")
    print("\n=== Рис. 2а: частка хибнопозитивних залежно від кількості агрегаторів M (δ = 1), % ===")
    for M, r in res["copiers"].items():
        print(f"M = {M:>2}:  А = {pct(r['A']['fpr'][0]):>5}   Б = {pct(r['B']['fpr'][0]):>5}   В = {pct(r['C']['fpr'][0]):>5}")
    print("\n=== Рис. 2б: методика В залежно від частки виявлених залежностей δ (M = 8), % ===")
    for d, r in res["delta"].items():
        print(f"δ = {d.replace('.', ','):<5} хибнопозитивні = {pct(r['C']['fpr'][0]):>5}   повнота = {pct(r['C']['recall'][0]):>5}")
    print("\n=== Повідомлень на індикатор і частка копій ===")
    for M, (avg, share) in res["reports"].items():
        print(f"M = {M:>2}:  {avg:.2f} повідомлення, копій {pct(share)} %".replace(".", ",", 1))


if __name__ == "__main__":
    import sys, time
    QUICK = "--quick" in sys.argv     # швидка перевірка, що все працює (числа будуть трохи інші)
    EXTRA = "--extra" in sys.argv     # додаткові перевірки для відповіді рецензенту
    R_MAIN, R_THR = (5, 3) if QUICK else (50, 20)
    t0 = time.time()
    step = lambda k, txt: print(f"[{k}/6] {txt} ... ({time.time() - t0:.0f} с)", flush=True)
    res = {}
    step(1, "Основний експеримент, δ = 1")
    res["main"] = run(n_copiers=8, delta=1.0, reps=R_MAIN)
    step(2, "Основний експеримент, δ = 0,5")
    res["main_d05"] = run(n_copiers=8, delta=0.5, reps=R_MAIN)
    step(3, "Чутливість до частки виявлених залежностей δ")
    res["delta"] = {str(d): run(n_copiers=8, delta=d, modes=("B", "C"), reps=R_MAIN) for d in (1.0, 0.75, 0.5, 0.25, 0.0)}
    step(4, "Вплив кількості агрегаторів M")
    res["copiers"] = {str(M): run(n_copiers=M, delta=1.0, modes=("A", "B", "C"), reps=R_MAIN) for M in (0, 4, 8, 16)}
    res["reports"] = {str(M): avg_reports(M) for M in (0, 4, 8, 16)}
    step(5, "Чутливість до порогів θ_S і θ_V")
    sens = {}
    for ts in (0.5, 0.6, 0.7):
        for tv in (0.6, 0.7, 0.8):
            TH_S, TH_VETO = ts, tv
            sens[f"{ts}_{tv}"] = run(n_copiers=8, delta=1.0, modes=("C",), reps=R_THR)["C"]
    TH_S, TH_VETO = 0.60, 0.70
    res["thresholds"] = sens
    if EXTRA:
        step(6, "Додаткові перевірки: частка фішингу 20 % і 60 %, мінімум 3 групи")
        extra = {}
        for prev in (0.2, 0.6):
            PREV = prev
            extra[f"prev_{prev}"] = run(n_copiers=8, delta=1.0, modes=("A", "B", "C"), reps=20)
        PREV = 0.4
        MIN_GROUPS = 3
        extra["min_groups_3"] = run(n_copiers=8, delta=1.0, modes=("C",), reps=20)
        MIN_GROUPS = 2
        res["extra"] = extra
    else:
        step(6, "Додаткові перевірки пропущено (запустіть з --extra)")
    out = "results_quick.json" if QUICK else "results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print_summary(res)
    if EXTRA:
        print("\n=== Додаткові перевірки (20 повторень), % ===")
        for k, r in res["extra"].items():
            print(k, {m: f"хибнопоз. {pct(v['fpr'][0])}, повнота {pct(v['recall'][0])}" for m, v in r.items()})
    print(f"\nГотово за {time.time() - t0:.0f} с. Результати збережено у файл {out}")
    if QUICK:
        print("УВАГА: це швидкий режим (5 повторень). Для чисел зі статті запускайте без --quick.")
