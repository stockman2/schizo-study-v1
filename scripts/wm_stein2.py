# -*- coding: utf-8 -*-
"""
wm_stein2.py — откуда берётся стягивание ответов к диагоналям в данных Stein, Barbosa и соавт. (2020).

ЧТО УЖЕ ИЗВЕСТНО (wm_stein.py). Около половины дисперсии ошибок — систематическая «карта сноса»: ответы уходят от
вертикали и горизонтали и стягиваются к диагоналям, тем сильнее, чем длиннее пауза. Карта одинакова у здоровых и больных.

ДВА ОБЪЯСНЕНИЯ
  Динамическое:   во время паузы след сам сползает к диагонали с постоянной относительной скоростью k.
                  Тогда доля стягивания lam(t) растёт как 1 - lam(t) = (1 - lam(0)) * exp(-k t) и от шума не зависит.
  «Через вывод»:  след только случайно блуждает, а стягивание возникает при ответе — неточное воспоминание подправляется
                  к середине квадранта тем сильнее, чем оно неточнее: lam = V / (V + Vp), V — случайная дисперсия,
                  Vp — постоянная «ширина представления о квадранте».

ИЗМЕРЯЕМЫЕ ВЕЛИЧИНЫ (у каждого человека при каждой паузе)
  lam — сила стягивания: ошибка = -lam * (расстояние точки до ближайшей диагонали); lam = 0,16 значит, что ответ
        проходит 16 % пути от точки до диагонали. Низкие гармоники (1 и 2 периода на круг) учитываются отдельно.
  V   — случайная дисперсия ошибки (после вычета карты сноса и влияния прошлой пробы).

ПРАВИЛА ЧТЕНИЯ (записаны до расчёта)
  П1. Карта считается «кардинальной», если при паузе 3 с на гармоники 4, 8 и 12 приходится не меньше 60 %
      систематической дисперсии (оценка по двум независимым половинам проб).
  П2. Рост стягивания. По lam(0), lam(1) и дисперсиям строятся два предсказания для lam(3):
        динамическое:   1 - lam(3) = (1 - lam(0)) * ((1 - lam(1)) / (1 - lam(0)))^3;
        «через вывод»:  lam(3) * (1 - lam(3)) = (V(3) / V(1)) * lam(1) * (1 - lam(1))
                        (следует из lam = Vпамяти/(Vпамяти + Vp) и того, что наблюдаемая дисперсия V = (1 - lam)^2 * Vпамяти).
      Объяснение «совместимо», если 95 % интервал разности (наблюдаемое минус предсказанное) содержит ноль.
      Совместимо одно -> оно и выигрывает; оба или ни одного -> «не решено». Основной расчёт — по всем 52 людям вместе.
  П3. Связь между людьми при паузе 1 с: ранговая корреляция lam и V (обе величины центрированы внутри своей группы).
        нижняя граница 95 % интервала выше нуля            -> в пользу «через вывод»;
        интервал содержит ноль и верхняя граница ниже 0,3  -> против «через вывод»;
        иначе -> «не решено». Рядом печатается надёжность обеих величин: при низкой надёжности связь не увидеть.
  Описательно (без правила): случайный разброс у осей и у диагоналей; lam по группам.

ЗАПУСК
  python wm_stein2.py --selftest     # два искусственных мира (динамический и «через вывод»): различает ли их метод
  python wm_stein2.py                # настоящие данные (data/behavior.pkl, тот же файл, что раньше)
  Результат: папка wm_stein2_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform
import numpy as np

URL = 'https://github.com/comptelab/serialNMDA/raw/master/data/behavior.pkl'
GROUPS = ['C', 'S', 'E']; GNAME = {'C': 'здоровые', 'S': 'шизофрения', 'E': 'энцефалит', 'ALL': 'все вместе'}
DELAYS = [0.0, 1.0, 3.0]; SIGMA_DOG = 0.8; NBINS = 24; NSPLIT = 60; NBOOT = 2000; KMAX = 12
T0 = time.time()


def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)

def dog(x, sigma=SIGMA_DOG): return x * np.exp(-x * x / (2 * sigma * sigma)) / (sigma * np.exp(-0.5))

def dist_to_diag(theta):
    """Расстояние (град) от точки до ближайшей диагонали: от -45 до +45; ноль на диагонали, +-45 на осях."""
    return np.mod(np.degrees(theta), 90.0) - 45.0


def load_real(path):
    import pandas as pd
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True); stage('файла %s нет — скачиваю из репозитория авторов (13 МБ)' % path); urllib.request.urlretrieve(URL, path)
    d = pd.read_pickle(path); n0 = len(d)
    d = d[((d.RT < 3) & (d.ITI < 5) & (d.raderror < 5) & (d.error.abs() < 1)).values]; d = d[(d.trial % 48 != 1).values]
    return dict(subject=d.subject.values.astype(str), group=d.group.values.astype(str), delay=d.delay.astype(int).values / 60.0, target=d.target.values.astype(float),
                prevcurr=d.prevcurr.values.astype(float), err=np.degrees(d.error.values.astype(float))), dict(n_raw=n0, n_final=len(d))


def synth(world, seed):
    """Искусственный мир: 'dyn' — след сползает к диагонали сам; 'inf' — стягивание возникает при ответе и зависит от шума."""
    rng = np.random.default_rng(seed); S = dict(subject=[], group=[], delay=[], target=[], prevcurr=[], err=[])
    for g, ns in (('C', 19), ('S', 17), ('E', 16)):
        for k in range(ns):
            n = 1050; th = rng.uniform(-np.pi, np.pi, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); pc = np.angle(np.exp(1j * (np.roll(th, 1) - th)))
            noise = np.exp(rng.normal(0, 0.3)) * (1.3 if g != 'C' else 1.0)                 # личный уровень шума; у больных выше
            V = noise * (10.0 + 10.0 * (dl > 0) + 4.0 * dl); dlt = dist_to_diag(th)
            if world == 'inf':
                lam = V / (V + 147.0 * np.exp(rng.normal(0, 0.3))); e = -lam * dlt + rng.normal(0, np.sqrt(V)) * (1 - lam)   # подправка к середине квадранта; ширина Vp у каждого своя
            else:
                kk = 0.088 * np.exp(rng.normal(0, 0.3)); lam = 1 - 0.95 * np.exp(-kk * dl); e = -lam * dlt + rng.normal(0, np.sqrt(V))
            e = e + 0.8 * np.sin(th + 1.0) + 0.5 * np.array([{0.0: -0.5, 1.0: 0.5, 3.0: 1.5}[t] for t in dl]) * dog(pc)
            for key, v in zip(('subject', 'group', 'delay', 'target', 'prevcurr', 'err'), ([g + '%02d' % k] * n, [g] * n, dl, th, pc, e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}


# ---------- оценки у одного человека ----------
def split_half_sys_var(theta, e, rng):
    b = np.minimum(((theta + np.pi) / (2 * np.pi) * NBINS).astype(int), NBINS - 1); acc = []
    for _ in range(NSPLIT):
        h = rng.random(len(e)) < 0.5; s1 = np.bincount(b[h], e[h], NBINS); n1 = np.bincount(b[h], minlength=NBINS); s2 = np.bincount(b[~h], e[~h], NBINS); n2 = np.bincount(b[~h], minlength=NBINS)
        ok = (n1 > 0) & (n2 > 0)
        if ok.sum() >= NBINS // 2: w = (n1 + n2)[ok].astype(float); acc.append(np.sum(w * (s1[ok] / n1[ok]) * (s2[ok] / n2[ok])) / w.sum())
    return float(np.mean(acc)) if acc else np.nan

def harm_power(theta, e, rng):
    """Дисперсия карты по гармоникам 1..KMAX: произведение коэффициентов из двух независимых половин (несмещённо)."""
    ks = np.arange(1, KMAX + 1)[:, None]; acc = np.zeros(KMAX); m = 0
    for _ in range(NSPLIT):
        h = rng.random(len(e)) < 0.5
        if h.sum() < 20 or (~h).sum() < 20: continue
        c = []
        for hh in (h, ~h): c.append((2 * np.mean(e[hh] * np.sin(ks * theta[hh]), 1), 2 * np.mean(e[hh] * np.cos(ks * theta[hh]), 1)))
        acc += (c[0][0] * c[1][0] + c[0][1] * c[1][1]) / 2; m += 1
    return acc / max(m, 1)

def fit_lam(theta, e):
    """Сила стягивания к диагонали; низкие гармоники (1 и 2) подгоняются одновременно, чтобы не мешали."""
    X = np.stack([dist_to_diag(theta), np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta), np.ones_like(theta)], 1)
    c = np.linalg.lstsq(X, e, rcond=None)[0]; return -float(c[0]), e - X @ c

def per_subject(d):
    rng = np.random.default_rng(2027); R = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; r = dict(group=d['group'][ms][0], lam={}, lamA={}, lamB={}, V={}, VA={}, VB={}, Vsys={}, harm={}, vloc={})
        for t in DELAYS:
            m = ms & (d['delay'] == t); e = d['err'][m]; th = d['target'][m]; n = int(m.sum())
            e = e - e.mean(); x = dog(d['prevcurr'][m]); e = e - np.sum(e * x) / np.sum(x * x) * x; vt = float(e.var())
            vs = split_half_sys_var(th, e, rng) + vt / n; r['Vsys'][t] = vs; r['V'][t] = vt - vs; r['harm'][t] = harm_power(th, e, rng)
            r['lam'][t], res = fit_lam(th, e); odd = np.arange(n) % 2 == 1
            for key_l, key_v, hh in (('lamA', 'VA', odd), ('lamB', 'VB', ~odd)):                # две половины проб — для оценки надёжности
                r[key_l][t], rr = fit_lam(th[hh], e[hh]); r[key_v][t] = float(rr.var())
            ad = np.abs(dist_to_diag(th)); r['vloc'][t] = [float(res[(ad >= a) & (ad < b)].var()) for a, b in ((0, 15), (15, 30), (30, 45.01))]
        R[s] = r
    return R


def ci(vals, fn, rng):
    vals = np.asarray(vals, float); n = len(vals); est = fn(vals.mean(0)); bs = np.array([fn(vals[rng.integers(0, n, n)].mean(0)) for _ in range(NBOOT)])
    return [float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float); return float(np.corrcoef(ra, rb)[0, 1])

def pred_dyn(m): return 1 - (1 - m[0]) * ((1 - m[1]) / (1 - m[0])) ** 3                       # m = [lam0, lam1, lam3, V0, V1, V3]
def pred_inf(m):
    """«Через вывод»: ответ = (1-lam)*воспоминание + lam*середина квадранта, поэтому наблюдаемая дисперсия V = (1-lam)^2 * Vпамяти,
    а lam = Vпамяти/(Vпамяти + Vp). Отсюда lam*(1-lam) пропорционально V: lam3*(1-lam3) = (V3/V1) * lam1*(1-lam1)."""
    if not (0 < m[1] < 1): return np.nan
    c = m[5] / m[4] * m[1] * (1 - m[1]); return (1 - np.sqrt(1 - 4 * c)) / 2 if c <= 0.25 else np.nan


def analyse(d):
    stage('оценки у каждого человека: гармоники карты, сила стягивания, случайная дисперсия'); R = per_subject(d); rng = np.random.default_rng(11); out = dict(sets={})
    stage('средние, интервалы (бутстрэп по людям, %d повторов) и проверка правил' % NBOOT)
    for g in GROUPS + ['ALL']:
        subs = [s for s in R if g == 'ALL' or R[s]['group'] == g]; A = lambda key: np.array([[R[s][key][t] for t in DELAYS] for s in subs]); G = dict(n=len(subs))
        lam, V = A('lam'), A('V'); M = np.column_stack([lam, V])
        G['lam'] = {str(t): ci(lam[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)}; G['V'] = {str(t): ci(V[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)}
        G['pred_dyn'] = ci(M, pred_dyn, rng); G['pred_inf'] = ci(M, pred_inf, rng)
        G['obs_minus_dyn'] = ci(M, lambda m: m[2] - pred_dyn(m), rng); G['obs_minus_inf'] = ci(M, lambda m: m[2] - pred_inf(m), rng)
        H = np.array([[R[s]['harm'][t] for t in DELAYS] for s in subs]); G['harm'] = {str(t): H[:, i].mean(0).tolist() for i, t in enumerate(DELAYS)}
        card = lambda h: h[[3, 7, 11]].sum() / h.sum()
        G['card_share_3s'] = ci(H[:, 2], card, rng); G['low_share_3s'] = ci(H[:, 2], lambda h: h[[0, 1, 2]].sum() / h.sum(), rng); G['Vsys_3s_bins'] = float(np.mean([R[s]['Vsys'][3.0] for s in subs]))
        VL = np.array([[R[s]['vloc'][t] for t in DELAYS] for s in subs]); G['vloc'] = {str(t): VL[:, i].mean(0).tolist() for i, t in enumerate(DELAYS)}
        G['vloc_ratio_axis_to_diag'] = {str(t): ci(VL[:, i][:, [0, 2]], lambda m: m[1] / m[0], rng) for i, t in enumerate(DELAYS)}
        out['sets'][g] = G
    # П3: связь между людьми при паузе 1 с (и, для справки, при 3 с), центрирование внутри группы
    subs = list(R); grp = np.array([R[s]['group'] for s in subs]); out['corr'] = {}
    for t in (1.0, 3.0):
        lam = np.array([R[s]['lam'][t] for s in subs]); V = np.array([R[s]['V'][t] for s in subs]); lamA = np.array([R[s]['lamA'][t] for s in subs]); lamB = np.array([R[s]['lamB'][t] for s in subs])
        VA = np.array([R[s]['VA'][t] for s in subs]); VB = np.array([R[s]['VB'][t] for s in subs])
        cen = lambda v: v - np.array([v[grp == g].mean() for g in grp]); lc, vc = cen(lam), cen(V); rho = spearman(lc, vc); n = len(subs)
        bs = []
        for _ in range(NBOOT):
            i = rng.integers(0, n, n); bs.append(spearman(lc[i], vc[i]))
        sb = lambda a, b: (lambda r: 2 * r / (1 + r))(np.corrcoef(cen(a), cen(b))[0, 1])
        out['corr'][str(t)] = dict(rho=[rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], rel_lam=float(sb(lamA, lamB)), rel_V=float(sb(VA, VB)), lam=lam.tolist(), V=V.tolist(), group=grp.tolist())
    # вердикты
    G = out['sets']['ALL']; inz = lambda c: c[1] <= 0 <= c[2]; okd, oki = inz(G['obs_minus_dyn']), inz(G['obs_minus_inf'])
    out['verdict'] = dict(P1='кардинальная' if G['card_share_3s'][0] >= 0.6 else 'не кардинальная',
                          P2='динамическое' if (okd and not oki) else ('через вывод' if (oki and not okd) else 'не решено (%s)' % ('оба совместимы' if okd else 'ни одно не совместимо')))
    c = out['corr']['1.0']['rho']; out['verdict']['P3'] = 'в пользу «через вывод»' if c[1] > 0 else ('против «через вывод»' if (c[1] <= 0 <= c[2] and c[2] < 0.3) else 'не решено')
    return out


def f3(x, nd=3): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def report(out, meta, outdir, tag=''):
    S = out['sets']; L = ['# Данные Stein 2020: откуда стягивание к диагоналям%s\n' % tag]
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    if meta: L.append('- проб после отбора: %d (в статье 52 394: %s)' % (meta['n_final'], 'совпало' if meta['n_final'] == 52394 else '**НЕ совпало**'))
    L.append('- в скобках 95 % интервал (бутстрэп по людям)\n')
    L.append('## П1. Из каких гармоник состоит карта (пауза 3 с)\n'); L.append('| набор | доля гармоник 4, 8, 12 | доля гармоник 1–3 | дисперсия карты: сумма гармоник / по ячейкам, град² |\n|---|---|---|---|')
    for g in GROUPS + ['ALL']: L.append('| %s | %s | %s | %.1f / %.1f |' % (GNAME[g], f3(S[g]['card_share_3s'], 2), f3(S[g]['low_share_3s'], 2), sum(S[g]['harm']['3.0']), S[g]['Vsys_3s_bins']))
    h = S['ALL']['harm']; L.append('\nДисперсия по гармоникам, все люди, град² (гармоники 1…12):'); L.append('')
    L.append('| пауза | ' + ' | '.join(str(k) for k in range(1, KMAX + 1)) + ' |\n|' + '---|' * (KMAX + 1))
    for t in ('0.0', '1.0', '3.0'): L.append('| %s с | ' % t[0] + ' | '.join('%.2f' % v for v in h[t]) + ' |')
    L.append('\n**Вывод по правилу: карта %s.**' % out['verdict']['P1'])
    L.append('\n## Сила стягивания к диагонали и случайная дисперсия\n'); L.append('| набор | людей | lam: 0 с | lam: 1 с | lam: 3 с | V: 0 с | V: 1 с | V: 3 с |\n|---|---|---|---|---|---|---|---|')
    for g in GROUPS + ['ALL']: L.append('| %s | %d | %s | %s | %s | %s | %s | %s |' % (GNAME[g], S[g]['n'], f3(S[g]['lam']['0.0']), f3(S[g]['lam']['1.0']), f3(S[g]['lam']['3.0']), f3(S[g]['V']['0.0'], 1), f3(S[g]['V']['1.0'], 1), f3(S[g]['V']['3.0'], 1)))
    L.append('\n## П2. Рост стягивания: наблюдаемое lam при 3 с против двух предсказаний\n'); L.append('| набор | наблюдается | динамическое предсказывает | разность | «через вывод» предсказывает | разность |\n|---|---|---|---|---|---|')
    for g in ['ALL'] + GROUPS: L.append('| %s | %s | %s | %s | %s | %s |' % (GNAME[g], f3(S[g]['lam']['3.0']), f3(S[g]['pred_dyn']), f3(S[g]['obs_minus_dyn']), f3(S[g]['pred_inf']), f3(S[g]['obs_minus_inf'])))
    L.append('\n**Вывод по правилу (по всем людям): %s.**' % out['verdict']['P2'])
    L.append('\n## П3. Связь силы стягивания со случайной дисперсией между людьми\n'); L.append('| пауза | ранговая корреляция | надёжность lam | надёжность V |\n|---|---|---|---|')
    for t in ('1.0', '3.0'): c = out['corr'][t]; L.append('| %s с%s | %s | %.2f | %.2f |' % (t[0], ' (основная)' if t == '1.0' else ' (справочно)', f3(c['rho'], 2), c['rel_lam'], c['rel_V']))
    L.append('\n**Вывод по правилу: %s.**' % out['verdict']['P3'])
    L.append('\n## Описательно: случайный разброс у диагоналей и у осей (дисперсия остатка, град², все люди)\n'); L.append('| пауза | у диагонали (0–15°) | середина (15–30°) | у оси (30–45°) | отношение «у оси / у диагонали» |\n|---|---|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): v = S['ALL']['vloc'][t]; L.append('| %s с | %.1f | %.1f | %.1f | %s |' % (t[0], v[0], v[1], v[2], f3(S['ALL']['vloc_ratio_axis_to_diag'][t], 2)))
    L.append('\n## Сводка для машинного чтения\n')
    slim = {g: {k: v for k, v in S[g].items()} for g in S}; cr = {t: {k: v for k, v in out['corr'][t].items() if k in ('rho', 'rel_lam', 'rel_V')} for t in out['corr']}
    L.append('```json\n' + json.dumps(dict(meta=meta, sets=slim, corr=cr, verdict=out['verdict']), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text)
    json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        col = {'C': 'k', 'S': '#c0392b', 'E': '#2471a3'}; fig, ax = plt.subplots(1, 4, figsize=(17, 4)); w = 0.27
        for i, t in enumerate(('0.0', '1.0', '3.0')): ax[0].bar(np.arange(1, KMAX + 1) + (i - 1) * w, S['ALL']['harm'][t], w, label='пауза %s с' % t[0])
        ax[0].set_xlabel('гармоника (периодов на круг)'); ax[0].set_ylabel('дисперсия, град²'); ax[0].set_title('Состав карты сноса', fontsize=10); ax[0].legend(frameon=False, fontsize=8)
        for g in GROUPS:
            m = np.array([S[g]['lam'][str(t)] for t in DELAYS]); ax[1].errorbar(DELAYS, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o', color=col[g], capsize=3, label=GNAME[g])
        a = S['ALL']; ax[1].errorbar([3.15], [a['pred_dyn'][0]], yerr=[[a['pred_dyn'][0] - a['pred_dyn'][1]], [a['pred_dyn'][2] - a['pred_dyn'][0]]], fmt='s', color='#d68910', capsize=3, label='предсказание: динамическое')
        ax[1].errorbar([3.3], [a['pred_inf'][0]], yerr=[[a['pred_inf'][0] - a['pred_inf'][1]], [a['pred_inf'][2] - a['pred_inf'][0]]], fmt='^', color='#1e8449', capsize=3, label='предсказание: «через вывод»')
        ax[1].set_xlabel('пауза, с'); ax[1].set_ylabel('lam'); ax[1].set_title('Сила стягивания к диагонали', fontsize=10); ax[1].legend(frameon=False, fontsize=7)
        c = out['corr']['1.0']
        for g in GROUPS: mk = np.array(c['group']) == g; ax[2].plot(np.array(c['V'])[mk], np.array(c['lam'])[mk], 'o', color=col[g], ms=4, label=GNAME[g])
        ax[2].set_xlabel('случайная дисперсия V, град²'); ax[2].set_ylabel('lam'); ax[2].set_title('Люди при паузе 1 с', fontsize=10); ax[2].legend(frameon=False, fontsize=8)
        for i, t in enumerate(('0.0', '1.0', '3.0')): ax[3].plot([7.5, 22.5, 37.5], S['ALL']['vloc'][t], '-o', label='пауза %s с' % t[0])
        ax[3].set_xlabel('расстояние точки до диагонали, град'); ax[3].set_ylabel('дисперсия остатка, град²'); ax[3].set_title('Случайный разброс: от диагонали к оси', fontsize=10); ax[3].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=130)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    return text


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--out', default='wm_stein2_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА: два искусственных мира, по три независимых набора на каждый. Метод должен назвать мир правильно.'); good = 0; tot = 0
        for world, want2, want3 in (('dyn', 'динамическое', None), ('inf', 'через вывод', 'в пользу «через вывод»')):
            for seed in (1, 2, 3):
                stage('мир «%s», набор %d' % ('динамический' if world == 'dyn' else 'через вывод', seed)); out = analyse(synth(world, seed)); v = out['verdict']; G = out['sets']['ALL']
                ok2 = v['P2'] == want2; ok3 = (want3 is None and not v['P3'].startswith('в пользу')) or v['P3'] == want3; good += ok2 + ok3; tot += 2
                print('   П1: %s | П2: %s %s | П3: %s %s | lam(3 с) = %.3f, предсказания %.3f и %.3f; корреляция %.2f' % (v['P1'], v['P2'], '(верно)' if ok2 else '(НЕВЕРНО)', v['P3'], '(верно)' if ok3 else '(НЕВЕРНО)',
                      G['lam']['3.0'][0], G['pred_dyn'][0], G['pred_inf'][0], out['corr']['1.0']['rho'][0]), flush=True)
            report(out, None, a.out + '_selftest_' + world, tag=' — ИСКУССТВЕННЫЙ МИР «%s»' % world)
        print('ИТОГ ПРОВЕРКИ: верных ответов %d из %d. %s' % (good, tot, 'Метод различает два мира.' if good == tot else 'Метод различает миры НЕ во всех случаях — см. строки выше; это его разрешающая способность.')); return
    stage('читаю %s' % a.data); d, meta = load_real(a.data); stage('проб после отбора %d, людей %d' % (meta['n_final'], len(np.unique(d['subject']))))
    out = analyse(d); text = report(out, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
