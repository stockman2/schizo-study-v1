# -*- coding: utf-8 -*-
"""
wm_open2b.py — ЗНАКОВЫЙ ТЕСТ, исправленная версия на открытых наборах (ориентация и направление движения) и, для сравнения, на данных Stein.

ИДЕЯ. У каждого человека измеряются два числа.
  lam — куда тянет ответы: плюс — к диагоналям (от кардинальных осей), минус — к осям.
  m   — где разброс больше: m = ln(дисперсия у осей / дисперсия у диагоналей); плюс — у осей, минус — у диагоналей.
Если смещение создаёт «яма» (место притяжения), то там, куда тянет, разброс сжат: знаки lam и m СОВПАДАЮТ
(тянет к диагоналям и разброс больше у осей — либо тянет к осям и разброс больше у диагоналей).
Если ответы уходят ОТ мест, где признак виден точнее (так предсказывает теория эффективного кодирования), то тянет туда,
где разброс больше: знаки lam и m ПРОТИВОПОЛОЖНЫ.
«У диагонали» — стимулы ближе 15 град к диагонали (45, 135, …), «у оси» — ближе 15 град к оси (0, 90, …).

ПРАВИЛО (записано до расчёта). Для каждого опыта считаются средние lam и m по людям с 95 % интервалами (бутстрэп по людям).
  «как яма»   — оба интервала не содержат ноль и знаки совпадают;
  «наоборот»  — оба интервала не содержат ноль и знаки противоположны;
  «не решено» — хотя бы один интервал содержит ноль.
Итог по виду стимула (ориентация; движение): если решённых опытов не меньше трёх и не меньше двух третей из них одного
вида — это и есть ответ; иначе «картина смешанная».
В расчёт входят опыты, где у человека не меньше 150 проб в условии; условия усредняются внутри человека.
ИСПРАВЛЕНИЕ по сравнению с wm_open2.py: перед сравнением разброса из ошибок вычитается гибкая карта смещений (гармоники до 12-й),
а не пила. Раньше плавное схождение карты к нулю у оси попадало в остаток и завышало разброс у осей примерно на 700*lam^2 град².
Повторно записанные наборы пропускаются. Правило то же.
Грубые промахи не отбрасываются: они добавляют разброс всюду поровну, то есть уменьшают |m|, но знак не меняют.

ЗАПУСК
  python wm_open2b.py --selftest   # два искусственных мира с известным ответом (несколько секунд)
  python wm_open2b.py              # таблицы из data_open/ (их уже скачал wm_open1.py); если рядом есть data/behavior.pkl
                                  # и wm_stein2.py, добавится строка по данным Stein
  Результат: папка wm_open2b_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform, glob
import numpy as np

NBOOT = 2000; MINTR = 150; T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)
def ddiag(th): return np.mod(th, 90.0) - 45.0            # расстояние до ближайшей диагонали, град: 0 на диагонали, ±45 на оси


def lam_m(theta, err, motion):
    """Один человек, одно условие. lam — наклон смещения относительно пилы (как раньше). m — логарифм отношения дисперсий
    «у оси / у диагонали», но дисперсии берутся по остатку ПОСЛЕ вычитания гибкой карты смещений (гармоники до 12-й),
    чтобы плавная форма карты не засчитывалась как разброс у осей."""
    d = ddiag(theta); t = np.radians(theta); cols = [d, np.sin(2 * t), np.cos(2 * t), np.ones_like(t)]
    if motion: cols += [np.sin(t), np.cos(t)]
    X = np.stack(cols, 1); lam = -np.linalg.lstsq(X, err, rcond=None)[0][0]
    ks = range(1, 13) if motion else range(2, 13, 2); H = np.stack([f(k * t) for k in ks for f in (np.sin, np.cos)] + [np.ones_like(t)], 1)
    res = err - H @ np.linalg.lstsq(H, err, rcond=None)[0]; ad = np.abs(d); a = res[ad >= 30]; b = res[ad < 15]
    if len(a) < 20 or len(b) < 20: return np.nan, np.nan, np.nan, np.nan
    return lam, float(np.log(a.var() / b.var())), float(b.var()), float(a.var())

def per_experiment(theta, err, obs, cond, motion):
    out = []
    for o in np.unique(obs):
        acc = []
        for c in np.unique(cond[obs == o]):
            k = (obs == o) & (cond == c)
            if k.sum() >= MINTR:
                r = lam_m(theta[k], err[k] - err[k].mean(), motion)
                if np.isfinite(r[0]): acc.append(r)
        if acc: out.append(np.mean(acc, 0))
    return np.array(out)

def summarize(name, stim, P, rng):
    n = len(P); idx = rng.integers(0, n, (NBOOT, n)); ci = lambda j: [float(P[:, j].mean())] + [float(x) for x in np.percentile(P[idx][:, :, j].mean(1), [2.5, 97.5])]
    lam, m = ci(0), ci(1); sl = 0 if lam[1] <= 0 <= lam[2] else np.sign(lam[0]); sm = 0 if m[1] <= 0 <= m[2] else np.sign(m[0])
    v = 'не решено' if sl == 0 or sm == 0 else ('как яма' if sl == sm else 'наоборот')
    return dict(name=name, stimulus=stim, n=n, lam=lam, m=m, v_diag=float(P[:, 2].mean()), v_axis=float(P[:, 3].mean()), same_sign_share=float(np.mean(P[:, 0] * P[:, 1] > 0)), verdict=v)


def load_open(folder):
    import pandas as pd
    for path in sorted(glob.glob(os.path.join(folder, '*.csv'))):
        d = pd.read_csv(path, sep=';')
        if 'expnum' not in d: d['expnum'] = 1
        if 'cond' not in d: d['cond'] = 1
        for ex, g in d.groupby('expnum'):
            g = g[np.isfinite(g['error']) & np.isfinite(g['theta'])]; stim = str(g['stimulus'].iloc[0]) if 'stimulus' in g else 'Orientation'
            name = '%s, %s' % (str(g['study'].iloc[0]) if 'study' in g else os.path.basename(path), str(g['experiment'].iloc[0]) if 'experiment' in g else ex)
            yield name, stim, g['theta'].values.astype(float), g['error'].values.astype(float), g['obs'].astype(str).values, g['cond'].astype(str).values

def synth(kind, rng):
    """Искусственные люди с ПЛАВНОЙ картой смещений (тянет к диагоналям). 'well': разброс больше у осей; 'anti': больше у диагоналей;
    'flat': разброс везде одинаков (прежний вариант теста ошибочно показывал бы здесь «как яма»)."""
    th, er, ob = [], [], []; amp = {'well': 0.6, 'anti': -0.4, 'flat': 0.0}[kind]
    for o in range(20):
        t = rng.uniform(0, 180, 600); d = ddiag(t); s = (np.abs(d) / 45.0) ** 2; sd = 12 * np.sqrt(1 + amp * (s - 1 / 3))
        th.append(t); er.append(-0.18 * 28.65 * np.sin(np.radians(4 * d)) + sd * rng.standard_normal(600) + 2 * np.sin(np.radians(2 * t) + rng.uniform(0, 6))); ob.append(np.full(600, str(o)))
    return np.concatenate(th), np.concatenate(er), np.concatenate(ob)


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--folder', default='data_open'); ap.add_argument('--stein', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--out', default='wm_open2b_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    rng = np.random.default_rng(91)
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА: три искусственных мира по 20 человек, карта смещений плавная')
        for kind, want in (('well', 'как яма'), ('anti', 'наоборот'), ('flat', 'не решено')):
            t, e, o = synth(kind, rng); r = summarize(kind, 'Orientation', per_experiment(t, e, o, np.full(len(t), '1'), False), rng)
            print('  мир «%s»: lam = %+.3f [%+.3f; %+.3f], m = %+.2f [%+.2f; %+.2f] -> %s (ожидалось: %s)' % ({'well': 'яма', 'anti': 'наоборот', 'flat': 'разброс одинаков'}[kind], *r['lam'], *r['m'], r['verdict'], want))
        return
    rows = []
    if os.path.exists(a.stein):
        try:
            import wm_stein2 as W
            d, _ = W.load_real(a.stein)
            for t in (1.0, 3.0):
                k = d['delay'] == t; P = per_experiment(np.degrees(d['target'][k]), d['err'][k], d['subject'][k], np.full(k.sum(), '1'), True)
                rows.append(summarize('Stein и соавт. (2020), положение, пауза %g с — для сравнения' % t, 'Position', P, rng))
            stage('данные Stein добавлены для сравнения')
        except Exception as e: stage('данные Stein пропущены (%s)' % repr(e)[:80])
    seen = {}; dup = []
    for name, stim, th, er, ob, co in load_open(a.folder):
        P = per_experiment(th, er, ob, co, stim == 'Motion'); key = (len(P), round(float(P[:, 0].sum()), 6)) if len(P) else None
        if key is not None and key in seen: stage('%s: ПОВТОР набора «%s» — пропущен' % (name, seen[key])); dup.append(name); continue
        if key is not None: seen[key] = name
        if len(P) >= 4: rows.append(summarize(name, stim, P, rng)); stage('%s: людей %d -> %s' % (name, len(P), rows[-1]['verdict']))
        else: stage('%s: пропущен (людей с достаточным числом проб: %d)' % (name, len(P)))
    L = ['# Знаковый тест (исправленный): тянет ли ответы туда, где разброс меньше\n']
    L.append('- среда: Python %s, numpy %s, %s; время %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    L.append('- lam: плюс — ответы тянет к диагоналям, минус — к осям; m = ln(дисперсия у осей / у диагоналей): плюс — разброс больше у осей')
    L.append('- «как яма»: знаки lam и m совпадают; «наоборот»: противоположны; в скобках 95 % интервал (бутстрэп по людям)\n')
    L.append('| опыт | стимул | людей | lam | m | дисперсия у диагоналей / у осей, град² | доля людей с совпадающими знаками | вердикт |\n|---|---|---|---|---|---|---|---|')
    for r in rows: L.append('| %s | %s | %d | %+.3f [%+.3f; %+.3f] | %+.2f [%+.2f; %+.2f] | %.0f / %.0f | %.2f | **%s** |' % (r['name'], r['stimulus'], r['n'], *r['lam'], *r['m'], r['v_diag'], r['v_axis'], r['same_sign_share'], r['verdict']))
    if dup: L.append('\nПропущены как повторы: ' + '; '.join(dup) + '.')
    L.append('\n## Итог по правилу\n'); tot = {}
    for stim, nm in (('Orientation', 'ориентация'), ('Motion', 'направление движения')):
        rr = [r for r in rows if r['stimulus'] == stim]; w = sum(r['verdict'] == 'как яма' for r in rr); x = sum(r['verdict'] == 'наоборот' for r in rr); dec = w + x
        v = 'решённых опытов меньше трёх' if dec < 3 else ('как яма' if w >= 2 * dec / 3 else 'наоборот' if x >= 2 * dec / 3 else 'картина смешанная'); tot[stim] = dict(n=len(rr), well=w, anti=x, verdict=v)
        L.append('- %s: опытов %d; «как яма» %d, «наоборот» %d, не решено %d. **Итог: %s.**' % (nm, len(rr), w, x, len(rr) - dec, v))
    L.append('\n## Сводка для машинного чтения\n'); L.append('```json\n' + json.dumps(dict(rows=rows, total=tot), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(dict(rows=rows, total=tot), open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 5.6)); col = {'Orientation': '#2471a3', 'Motion': '#d68910', 'Position': 'k'}; nm = {'Orientation': 'ориентация', 'Motion': 'движение', 'Position': 'положение (Stein)'}; seen = set()
        for r in rows:
            ax.errorbar(r['lam'][0], r['m'][0], xerr=[[r['lam'][0] - r['lam'][1]], [r['lam'][2] - r['lam'][0]]], yerr=[[r['m'][0] - r['m'][1]], [r['m'][2] - r['m'][0]]], fmt='o', color=col[r['stimulus']], capsize=2, ms=5, lw=0.8, label=None if r['stimulus'] in seen else nm[r['stimulus']]); seen.add(r['stimulus'])
        ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7); xl = ax.get_xlim(); yl = ax.get_ylim()
        for x, y, s in ((0.97, 0.97, 'как яма'), (0.03, 0.03, 'как яма'), (0.03, 0.97, 'наоборот'), (0.97, 0.03, 'наоборот')): ax.text(x, y, s, transform=ax.transAxes, ha='right' if x > 0.5 else 'left', va='top' if y > 0.5 else 'bottom', color='0.45', fontsize=9)
        ax.set_xlabel('lam: к осям  ←   куда тянет ответы   →  к диагоналям'); ax.set_ylabel('m: у диагоналей  ←   где разброс больше   →  у осей'); ax.legend(frameon=False, fontsize=8, loc='center right'); ax.set_title('Каждая точка — один опыт', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'figures.png'), dpi=130)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    print('\n' + text[:text.find('## Сводка для машинного чтения')]); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
