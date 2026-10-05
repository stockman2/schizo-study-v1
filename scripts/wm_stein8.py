# -*- coding: utf-8 -*-
"""
wm_stein8.py — пересмотр после непройденной проверки (wm_stein7.py): гладкая форма потенциала вместо «пилы».

ЧТО БЫЛО. В wm_stein7.py стягивание к диагонали задавалось пилой: сила растёт линейно до самой оси и там меняет знак
скачком. Дифференциальная модель с такой пилой дала верный уровень разброса у осей, но ровный профиль до оси и скачок,
а в данных разброс нарастает от диагонали к оси постепенно. Проверка при паузе 3 с не пройдена.
Измеренная карта смещений при этом глаже пилы: на гармонику с 8 периодами приходится примерно в 12 раз меньше мощности,
чем на гармонику с 4 периодами (у пилы — в 4 раза).

ЧТО МЕНЯЕТСЯ. Форма стягивания g(x) задаётся гармониками 4, 8 и 12 и подстраивается так, чтобы средняя карта смещений,
которую даёт сама модель, совпала с измеренной (усреднение по людям и паузам). Форма одна на всех, но у каждой модели
своя: одна и та же карта получается в Д2 и в О2 из немного разных g. Форма определяется по СРЕДНИМ ошибкам, а проверка
идёт по РАЗБРОСУ, который в настройку не входит. Всё остальное — как в wm_stein7.py.
  Д2, дифференциальная:      dx = −(D(t)/Vs) · g(x) · dt + sqrt(2·D(t)) · dW
  О2, подправка при ответе:  ответ = m − lam · g(m),  lam = Vпамяти/(Vпамяти + Vp)   (теперь без скачка на осях)

ПРОВЕРЯЕМЫЕ ВЕЛИЧИНЫ (паузы 1 и 3 с)
  R_ось = дисперсия остатка у осей (30–45° от диагонали) / у диагоналей (0–15°)
  R_сер = дисперсия остатка в середине (15–30°)          / у диагоналей (0–15°)

ПРАВИЛО (записано до расчёта; основная проверка — исходные сессии, 52 человека)
  Модель «совместима», если для ВСЕХ ЧЕТЫРЁХ сравнений (две величины x две паузы) 99 % интервал разности
  «данные − модель» содержит ноль (бутстрэп по людям на общей выборке).
    совместима только Д2 -> в пользу дифференциальной модели;   только О2 -> в пользу подправки при ответе;
    обе -> профиль разброса их не различает;                     ни одна -> обе модели профиль не описывают.
  ОГОВОРКА, записанная заранее: это пересмотр модели после непройденной проверки, а профиль разброса в обоих наборах
  данных уже был виден. Поэтому положительный исход весит меньше, чем предсказание с первого раза.
  Справочно: сессии post; пауза 0 с; сила стягивания в моделях; островершинность остатков.

ЗАПУСК (нужны wm_stein2.py … wm_stein5.py и wm_stein7.py в той же папке)
  python wm_stein8.py --selftest     # искусственные миры Д2 и О2 (около 10 минут)
  python wm_stein8.py                # настоящие данные (около 4 минут)
  Результат: папка wm_stein8_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform
import numpy as np
import wm_stein2 as W
import wm_stein3 as W3
import wm_stein5 as W5
import wm_stein7 as S7

DELAYS = W.DELAYS; NREP = S7.NREP; DTAU = S7.DTAU; NBOOT = 2000; NREFINE = 7; NT = 1800; T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)
XT = (np.arange(NT) + 0.5) * 90.0 / NT                                  # сетка мест в пределах одного квадранта, град
def gfun(x_deg, G): return G[(np.mod(x_deg, 90.0) * (NT / 90.0)).astype(np.int64) % NT]


def table_from_coef(c6):
    """Таблица формы стягивания по коэффициентам гармоник 4, 8, 12 средней карты ошибок; нормировка на единичный наклон относительно пилы."""
    xr = np.radians(XT); b = np.zeros(NT)
    for j, k in enumerate((4, 8, 12)): b += c6[2 * j] * np.sin(k * xr) + c6[2 * j + 1] * np.cos(k * xr)
    dl = S7.ddiag(XT); lam_bar = -np.sum(b * dl) / np.sum(dl * dl); return -b / lam_bar, float(lam_bar)

def norm_coef(cells):
    """Средняя по людям карта (гармоники 4, 8, 12), нормированная на собственный наклон. cells: список (человек, число проб, 24 коэффициента)."""
    by = {}
    for s_, n_, c_ in cells: a = by.setdefault(s_, [0.0, 0.0]); a[0] = a[0] + n_ * c_; a[1] += n_
    c = np.mean([v[0] / v[1] for v in by.values()], 0); c6 = np.array([c[2 * (k - 1) + j] for k in (4, 8, 12) for j in (0, 1)]); G, lam_bar = table_from_coef(c6); return c6 / lam_bar

def data_cells(d, serial=True):
    cells = []
    for s_ in np.unique(d['subject']):
        ms = d['subject'] == s_
        for t in DELAYS:
            m = ms & (d['delay'] == t); e = d['err'][m]; e = e - e.mean()
            if serial: x = W.dog(d['prevcurr'][m]); e = e - np.sum(e * x) / np.sum(x * x) * x
            cells.append((s_, int(m.sum()), W5.harm_fit(d['target'][m], e)))
    return cells

def shape_info(G, c6):
    dl = S7.ddiag(XT); a4 = np.hypot(c6[0], c6[1]); gr = np.gradient(G, XT)
    return dict(amp8_rel=float(np.hypot(c6[2], c6[3]) / a4), amp12_rel=float(np.hypot(c6[4], c6[5]) / a4), slope_diag=float(gr[NT // 2]), slope_axis=float(gr[1]), where_max=float(abs(dl[np.argmax(np.abs(G))])))

def true_shape():
    """Форма для искусственных миров: гладкая, с теми же соотношениями гармоник, что в настоящей карте."""
    u = np.radians(S7.ddiag(XT)); g = np.sin(4 * u) - 0.29 * np.sin(8 * u) + 0.06 * np.sin(12 * u); dl = S7.ddiag(XT); return g * np.sum(dl * dl) / np.sum(g * dl)


# ---------- модели ----------
def sim_D2(th0, tau, vs, G, rng):
    nst = np.round(tau / DTAU).astype(int); order = np.argsort(-nst); x = th0[order].copy(); inv = 1.0 / vs[order]; ns = nst[order]; sd = np.sqrt(2 * DTAU)
    cnt = np.searchsorted(-ns, -np.arange(1, ns.max() + 1), side='right') if ns.max() > 0 else np.array([], int)
    for k in range(len(cnt)):
        m = cnt[k]
        if m == 0: break
        x[:m] += -gfun(x[:m], G) * inv[:m] * DTAU + sd * rng.standard_normal(m)
    out = np.empty_like(x); out[order] = x; return out

def sim_O2(th0, vmem, lam, G, rng):
    m = th0 + np.sqrt(vmem) * rng.standard_normal(len(th0)); return m - lam * gfun(m, G)

def predict(M, model, target, rng, label):
    """Настройка прогонами самой модели: (1) общая дисперсия каждого человека при каждой паузе, (2) суммарное стягивание
    каждого человека, (3) форма средней карты смещений (гармоники 4, 8, 12, общая для всех). Разброс по полосам не используется."""
    subs = list(M); par = {}; noise = {}; shp = target.copy(); G, _ = table_from_coef(shp); z = lambda c: np.array([c[0] + 1j * c[1], c[2] + 1j * c[3], c[4] + 1j * c[5]])
    for s in subs:
        n = np.array([M[s][t]['n'] for t in DELAYS], float); V = np.array([M[s][t]['V'] for t in DELAYS]); lam = np.array([M[s][t]['lam'] for t in DELAYS])
        den = np.sum(n * lam * ((2 - lam) if model == 'D' else (1 - lam))); num = np.sum(n * V); par[s] = num / den if den > 0.02 * n.sum() else np.inf
        for t, v in zip(DELAYS, V):
            if model == 'D': noise[s, t] = v / 2 if not np.isfinite(par[s]) else -(par[s] / 2) * np.log(1 - min(v / par[s], 0.95))
            else: q = 0.0 if not np.isfinite(par[s]) else min(v / par[s], 0.24); l0 = (1 - np.sqrt(1 - 4 * q)) / 2; noise[s, t] = v / (1 - l0) ** 2
    P = None
    for it in range(NREFINE + 1):
        th0 = []; a1 = []; a2 = []; idx = []
        for s in subs:
            for t in DELAYS:
                th = np.degrees(np.tile(M[s][t]['th'], NREP)); th0.append(th); idx.append((s, t, len(th))); a1.append(np.full(len(th), noise[s, t]))
                a2.append(np.full(len(th), par[s] if model == 'D' else (noise[s, t] / (noise[s, t] + par[s]) if np.isfinite(par[s]) else 0.0)))
        th0 = np.concatenate(th0); a1 = np.concatenate(a1); a2 = np.concatenate(a2)
        resp = sim_D2(th0, a1, a2, G, rng) if model == 'D' else sim_O2(th0, a1, a2, G, rng); err = S7.wrap(resp - th0); P = {s: dict(group=M[s]['group']) for s in subs}; p = 0; worst = 0.0; cells = []
        for s, t, n in idx:
            thr = np.radians(th0[p:p + n]); e = err[p:p + n]; lam, V, v, k = S7.analyse_block(thr, e); P[s][t] = dict(lam=lam, V=V, vloc=v, kurt=k); cells.append((s, n, W5.harm_fit(thr, e - e.mean()))); p += n
            f = M[s][t]['V'] / V if V > 0 else 1.0; worst = max(worst, abs(f - 1))
            if it < NREFINE: noise[s, t] *= float(np.clip(f, 0.5, 2.0))
        gd = gs = 0.0
        for s in subs:
            ad = sum(M[s][t]['n'] * M[s][t]['lam'] for t in DELAYS); am = sum(M[s][t]['n'] * P[s][t]['lam'] for t in DELAYS); gd += ad; gs += am
            if it < NREFINE and np.isfinite(par[s]) and ad > 0 and am > 0: par[s] *= float(np.clip(am / ad, 0.5, 2.0))
        cs = norm_coef(cells); r = z(target) / z(cs); miss = float(np.max(np.abs(z(cs) - z(target))) / np.abs(z(target))[0])
        if it < NREFINE:
            r = np.clip(np.abs(r), 0.5, 2.0) * np.exp(1j * np.angle(r)); zz = z(shp) * r; shp = np.array([zz[0].real, zz[0].imag, zz[1].real, zz[1].imag, zz[2].real, zz[2].imag]); G, _ = table_from_coef(shp)
        stage('   модель %s2 (%s): проход %d из %d; расхождение: общая дисперсия до %.1f %%, стягивание %.1f %%, форма карты %.1f %%' % ('Д' if model == 'D' else 'О', label, it + 1, NREFINE + 1, 100 * worst, 100 * (gs / gd - 1), 100 * miss))
    return P, {s: float(par[s]) if np.isfinite(par[s]) else None for s in subs}, G, shp

def compare(M, PD, PR, rng):
    subs = list(M); n = len(subs); out = {}; X = dict(data=M, D=PD, R=PR)
    arr = lambda Y, key: np.array([[Y[s][t][key] for t in DELAYS] for s in subs]); VL = {k: arr(Y, 'vloc') for k, Y in X.items()}; KU = {k: arr(Y, 'kurt') for k, Y in X.items()}; LA = {k: arr(Y, 'lam') for k, Y in X.items()}; VV = {k: arr(Y, 'V') for k, Y in X.items()}
    idx = rng.integers(0, n, (NBOOT, n)); full = np.arange(n); ratio = lambda A, i, ti, b: A[i][:, ti, b].mean() / A[i][:, ti, 0].mean(); ok = dict(D=True, R=True)
    for ti, t in enumerate(DELAYS):
        o = {}
        for k in X:
            for nm, b in (('Rax', 2), ('Rmid', 1)):
                bs = np.array([ratio(VL[k], i, ti, b) for i in idx]); o['%s_%s' % (nm, k)] = [float(ratio(VL[k], full, ti, b)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            o['bands_' + k] = VL[k][:, ti].mean(0).tolist(); o['kurt_' + k] = np.nanmean(KU[k][:, ti], 0).tolist(); o['lam_' + k] = float(LA[k][:, ti].mean()); o['V_' + k] = float(VV[k][:, ti].mean())
        for k in ('D', 'R'):
            for nm, b in (('Rax', 2), ('Rmid', 1)):
                bs = np.array([ratio(VL['data'], i, ti, b) - ratio(VL[k], i, ti, b) for i in idx]); dd = [float(ratio(VL['data'], full, ti, b) - ratio(VL[k], full, ti, b)), float(np.percentile(bs, 0.5)), float(np.percentile(bs, 99.5))]
                o['d%s_%s' % (nm, k)] = dd
                if t in (1.0, 3.0) and not (dd[1] <= 0 <= dd[2]): ok[k] = False
        out[str(t)] = o
    out['compatible'] = ok
    out['verdict'] = ('в пользу дифференциальной модели' if ok['D'] and not ok['R'] else 'в пользу подправки при ответе' if ok['R'] and not ok['D'] else
                      'профиль разброса эти модели не различает (обе совместимы)' if ok['D'] else 'обе модели профиль не описывают')
    return out

def run(d, label, seed, serial=True):
    stage('%s: измерения по данным (%d человек)' % (label, len(np.unique(d['subject'])))); M = S7.measure(d, serial); rng = np.random.default_rng(seed)
    stage('%s: средняя карта смещений (гармоники 4, 8, 12) — цель для формы стягивания' % label); target = norm_coef(data_cells(d, serial)); G0, _ = table_from_coef(target)
    stage('%s: модель Д2' % label); PD, parD, GD, sD = predict(M, 'D', target, rng, label); stage('%s: модель О2' % label); PR, parR, GR, sR = predict(M, 'R', target, rng, label)
    stage('%s: сравнение и интервалы (бутстрэп по людям, %d повторов)' % (label, NBOOT)); out = compare(M, PD, PR, rng); out['n'] = len(M)
    out['shape'] = dict(map=shape_info(G0, target), D=shape_info(GD, sD), R=shape_info(GR, sR)); out['G'] = dict(map=G0.tolist(), D=GD.tolist(), R=GR.tolist())
    out['Vs_median'] = float(np.median([v for v in parD.values() if v])); out['Vp_median'] = float(np.median([v for v in parR.values() if v])); return out


def f3(x, nd=2): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def section(L, out, title, with_rule):
    L.append('## %s (людей: %d)\n' % (title, out['n'])); sh = out['shape']
    L.append('Форма средней карты и подстроенные под неё формы стягивания (у пилы: гармоника 8 — 0,50 от гармоники 4, гармоника 12 — 0,33; наклон у диагонали 1):\n')
    L.append('| | гармоника 8 / гармоника 4 | гармоника 12 / гармоника 4 | наклон у диагонали | наклон у оси | наибольшее стягивание на расстоянии от диагонали |\n|---|---|---|---|---|---|')
    for key, nm_ in (('map', 'измеренная карта'), ('D', 'стягивание в модели Д2'), ('R', 'стягивание в модели О2')): x = sh[key]; L.append('| %s | %.2f | %.2f | %.2f | %.2f | %.0f° |' % (nm_, x['amp8_rel'], x['amp12_rel'], x['slope_diag'], x['slope_axis'], x['where_max']))
    L.append('')
    L.append('Медиана параметра ямы: Vs = %.0f град² (Д2), Vp = %.0f град² (О2).\n' % (out['Vs_median'], out['Vp_median']))
    for nm, ttl in (('Rax', 'R_ось: у оси / у диагонали'), ('Rmid', 'R_сер: середина / у диагонали')):
        L.append('**%s**\n' % ttl); L.append('| пауза | данные | модель Д2 | данные − Д2 | модель О2 | данные − О2 |\n|---|---|---|---|---|---|')
        for t in ('0.0', '1.0', '3.0'): o = out[t]; L.append('| %s с%s | %s | %s | %s | %s | %s |' % (t[0], ' (справочно)' if t == '0.0' else '', f3(o[nm + '_data']), f3(o[nm + '_D']), f3(o['d%s_D' % nm]), f3(o[nm + '_R']), f3(o['d%s_R' % nm])))
        L.append('')
    if with_rule: L.append('**Вывод по правилу: %s.** Совместимость: Д2 — %s, О2 — %s.' % (out['verdict'], 'да' if out['compatible']['D'] else 'нет', 'да' if out['compatible']['R'] else 'нет'))
    else: L.append('Справочно, без правила: по тому же критерию получилось бы «%s».' % out['verdict'])
    L.append('\nДисперсия остатка по полосам, град² (у диагонали / середина / у оси):\n'); L.append('| пауза | данные | модель Д2 | модель О2 |\n|---|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): o = out[t]; L.append('| %s с | %s | %s | %s |' % (t[0], ' / '.join('%.1f' % x for x in o['bands_data']), ' / '.join('%.1f' % x for x in o['bands_D']), ' / '.join('%.1f' % x for x in o['bands_R'])))
    L.append('\nПроверка настройки — сила стягивания lam и общая дисперсия V (данные / Д2 / О2):\n'); L.append('| пауза | lam | V, град² |\n|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): o = out[t]; L.append('| %s с | %.3f / %.3f / %.3f | %.1f / %.1f / %.1f |' % (t[0], o['lam_data'], o['lam_D'], o['lam_R'], o['V_data'], o['V_D'], o['V_R']))
    L.append('\nОстровершинность остатков при 3 с (у диагонали / середина / у оси): данные %s; Д2 %s; О2 %s.\n' % tuple(' / '.join('%.2f' % x for x in out['3.0']['kurt_' + k]) for k in ('data', 'D', 'R')))

def report(res, meta, outdir):
    L = ['# Гладкий потенциал: дифференциальная модель против подправки при ответе\n']
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    L.append('- в скобках 95 % интервал, а для разностей «данные − модель» — 99 % (бутстрэп по людям)')
    L.append('- форма стягивания подстроена под среднюю карту ошибок (гармоники 4, 8, 12); модели настроены по силе стягивания и общей дисперсии каждого человека; распределение дисперсии по окружности в настройку не входило')
    L.append('- это пересмотр модели после непройденной проверки; профиль разброса в обоих наборах уже был виден\n')
    section(L, res['base'], 'Основная проверка: исходные сессии', True)
    if 'post' in res: section(L, res['post'], 'Справочно: сессии post', False)
    slim = {k: {kk: vv for kk, vv in res[k].items() if kk != 'G'} for k in res}
    L.append('## Сводка для машинного чтения\n'); L.append('```json\n' + json.dumps(dict(meta=meta, **slim), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(res, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        keys = [k for k in ('base', 'post') if k in res]; fig, ax = plt.subplots(len(keys), 5, figsize=(21, 4 * len(keys)), squeeze=False); col = {'data': 'k', 'D': '#c0392b', 'R': '#2471a3'}; nm = {'data': 'данные', 'D': 'модель Д2', 'R': 'модель О2'}
        for r, key in enumerate(keys):
            o = res[key]; dl = S7.ddiag(XT); ax[r][0].plot(dl, np.array(o['G']['map']), color='k', label='стягивание прямо по измеренной карте'); ax[r][0].plot(dl, np.array(o['G']['D']), color='#c0392b', ls='--', label='стягивание в Д2'); ax[r][0].plot(dl, np.array(o['G']['R']), color='#2471a3', ls='--', label='стягивание в О2'); ax[r][0].plot(dl, dl, color='gray', ls=':', label='пила (прежняя)')
            ax[r][0].axhline(0, color='gray', lw=0.5); ax[r][0].set_xlabel('расстояние до диагонали, град (со знаком)'); ax[r][0].set_ylabel('стягивание g, усл. град'); ax[r][0].set_title('Форма стягивания', fontsize=10); ax[r][0].legend(frameon=False, fontsize=8)
            for j, t in enumerate(('1.0', '3.0')):
                for k in ('data', 'D', 'R'): ax[r][1 + j].plot([7.5, 22.5, 37.5], o[t]['bands_' + k], '-o' if k == 'data' else '--s', color=col[k], label=nm[k])
                ax[r][1 + j].set_title('%s, пауза %s с' % ('исходные сессии' if key == 'base' else 'сессии post', t[0]), fontsize=10); ax[r][1 + j].set_xlabel('расстояние до диагонали, град'); ax[r][1 + j].set_ylabel('дисперсия остатка, град²')
            ax[r][1].legend(frameon=False, fontsize=8)
            for c_, (nmk, ttl) in enumerate((('Rax', 'У оси / у диагонали'), ('Rmid', 'Середина / у диагонали'))):
                for i, k in enumerate(('data', 'D', 'R')):
                    m = np.array([o[t]['%s_%s' % (nmk, k)] for t in ('0.0', '1.0', '3.0')]); ax[r][3 + c_].errorbar(np.array([0, 1, 3]) + (i - 1) * 0.07, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o' if k == 'data' else '--s', color=col[k], capsize=3, label=nm[k])
                ax[r][3 + c_].axhline(1, color='gray', lw=0.6); ax[r][3 + c_].set_title(ttl, fontsize=10); ax[r][3 + c_].set_xlabel('пауза, с'); ax[r][3 + c_].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=120)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    return text


def synth_world(model, seed):
    rng = np.random.default_rng(seed); G = true_shape(); S = {k: [] for k in ('subject', 'group', 'delay', 'target', 'prevcurr', 'err')}
    for g, ns in (('C', 19), ('S', 17), ('E', 16)):
        for k in range(ns):
            n = 1050; th = rng.uniform(-180, 180, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); mult = np.exp(rng.normal(0, 0.3)); vs = 140.0 * np.exp(rng.normal(0, 0.3))
            tau = mult * np.array([{0.0: 6.0, 1.0: 16.0, 3.0: 21.0}[t] for t in dl])
            if model == 'D': resp = sim_D2(th, tau, np.full(n, vs), G, rng)
            else: vm = 2 * tau; resp = sim_O2(th, vm, vm / (vm + 2 * vs), G, rng)
            e = S7.wrap(resp - th) + 1.2 * np.sin(rng.uniform(-np.pi, np.pi) - np.radians(th))
            for key, v in zip(S, ([g + '%02d' % k] * n, [g] * n, dl, np.radians(th), np.zeros(n), e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}

def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--retest', default=os.path.join('data', 'behavior_retest.pkl'))
    ap.add_argument('--out', default='wm_stein8_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА: искусственные люди порождены моделью Д2 или О2 с гладкой формой стягивания; форма восстанавливается')
        print('из их же средних ошибок, как это делается для настоящих данных. Метод должен назвать породившую модель.')
        for model, name in (('D', 'Д2'), ('R', 'О2')):
            for seed in (1, 2):
                o = run(synth_world(model, seed), 'мир %s, набор %d' % (name, seed), 100 + seed, serial=False); s = o['shape']['D' if model == 'D' else 'R']
                print('   ИСТИНА: модель %s -> вердикт: %s' % (name, o['verdict'])); print('      восстановленная форма стягивания: гармоника 8 / гармоника 4 = %.2f (заложено 0.29); наклон у диагонали %.2f (заложено 1.69), у оси %.2f (заложено -4.97)' % (s['amp8_rel'], s['slope_diag'], s['slope_axis']))
                for t in ('1.0', '3.0'): print('      пауза %s с: у оси — «данные» %s | Д2 %.2f | О2 %.2f;  середина — «данные» %s | Д2 %.2f | О2 %.2f' % (t[0], f3(o[t]['Rax_data']), o[t]['Rax_D'][0], o[t]['Rax_R'][0], f3(o[t]['Rmid_data']), o[t]['Rmid_D'][0], o[t]['Rmid_R'][0]), flush=True)
        return
    res = {}; stage('читаю %s' % a.data); d, meta = W.load_real(a.data); res['base'] = run(d, 'исходные сессии', 81)
    if os.path.exists(a.retest):
        stage('читаю %s' % a.retest); d2, meta2 = W3.load(a.retest); m = d2['session'] == 'post'; res['post'] = run({k: v[m] for k, v in d2.items() if k != 'session'}, 'сессии post', 82); meta = dict(meta, n_post=meta2['n_post'])
    else: stage('файла %s нет — сессии post пропущены' % a.retest)
    text = report(res, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text[:text.find('## Сводка для машинного чтения')] + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
