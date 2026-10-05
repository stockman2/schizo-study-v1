# -*- coding: utf-8 -*-
"""
wm_stein9.py — проверка гипотезы о двух родах смещений в памяти на положение (данные Stein и соавт., 2020).

ГИПОТЕЗА. Смещения бывают двух родов. Первые идут вслед за неопределённостью (стягивание к диагоналям, общий сдвиг) —
это проверено раньше. Вторые — медленный дрейф, которого нет в первую секунду паузы и который нарастает потом.
К дрейфу относятся (а) стягивание к оси 0–180 град (проверено раньше) и, предположительно, (б) притяжение к положению
точки в ПРЕДЫДУЩЕЙ пробе. При шизофрении ослаблен именно дрейф.

ВЕЛИЧИНЫ (у каждого человека при каждой паузе t)
  beta(t) — притяжение к прошлой пробе, град (плюс — притяжение, минус — отталкивание), как в wm_stein.py;
  s2(t)   — стягивание к оси 0–180, град, как в wm_stein6.py;
  V(t)    — случайная дисперсия (после вычета влияния прошлой пробы и карты смещений).

ПРАВИЛА (записаны до расчёта)
  П1. Закон роста притяжения к прошлому. Сессии post (22 человека; серийная зависимость на них ещё не считалась).
      Два предсказания для beta(3) по паузам 0 и 1 с:
        «вслед за неопределённостью»: beta(0) + [beta(1) − beta(0)] * [V(3) − V(0)] / [V(1) − V(0)];
        «постоянная скорость»:        beta(0) + 3 * [beta(1) − beta(0)].
      ПОДТВЕРЖДЕНО, если 95 % интервал разности (наблюдаемое минус «вслед за неопределённостью») не содержит ноль
      и разность положительна (притяжение растёт быстрее, чем неопределённость).
  П2. Поздний дрейф к оси у больных шизофренией. Исходные сессии. L = [s2(3) − s2(1)] / 2 у каждого человека.
      ПОДТВЕРЖДЕНО, если 95 % интервал разности средних (здоровые минус шизофрения) целиком выше нуля.
      Оговорка: эти данные уже просматривались, независимой выборки больных нет.
  П3. Связь двух дрейфов у одного человека. Исходные сессии, все люди, величины центрированы внутри группы.
      Ранговая корреляция позднего прироста притяжения [beta(3) − beta(1)] и позднего дрейфа к оси L.
      ПОДТВЕРЖДЕНО, если нижняя граница 95 % интервала выше нуля. Рядом печатается надёжность обеих величин.
  ИСХОД: П1 и П2 подтверждены -> гипотеза о двух родах смещений и её связи с шизофренией поддержана;
         только П1 -> притяжение к прошлому есть дрейф, о больных утверждать нельзя;  П1 не подтверждено -> путь закрыт.

ЗАПУСК (нужны wm_stein2.py, wm_stein3.py, wm_stein5.py и оба файла данных в папке data/)
  python wm_stein9.py --selftest    # два искусственных мира (несколько секунд)
  python wm_stein9.py               # настоящие данные
  Результат: папка wm_stein9_out/ — report.md, results.json
"""
import sys, os, json, time, argparse, platform
import numpy as np
import wm_stein2 as W
import wm_stein3 as W3
import wm_stein5 as W5

DELAYS = W.DELAYS; NBOOT = 2000; GN = {'C': 'здоровые', 'S': 'шизофрения', 'E': 'энцефалит'}; T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)


def cell(th, pc, e):
    e = e - e.mean(); x = W.dog(pc); beta = float(np.sum(e * x) / np.sum(x * x)); e1 = e - beta * x; c = W5.harm_fit(th, e1)
    res = e1 - W5.recon(c[None, :], range(1, W5.KMAX + 1), th)[0]; p = 2 * W5.KMAX + 3          # число подогнанных величин: гармоники, среднее, прошлая проба
    return beta, float(-c[2]), float(np.sum(res ** 2) / max(len(e) - p, 1))                       # несмещённая дисперсия: при малом числе проб подгонка иначе занижает её

def per_subject(d):
    """По каждому человеку: beta, s2, V при трёх паузах, а также beta и s2 по чётным и нечётным пробам (для надёжности)."""
    R = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; r = dict(group=d['group'][ms][0]); ok = True
        for t in DELAYS:
            m = ms & (d['delay'] == t)
            if m.sum() < 60: ok = False; break
            th, pc, e = d['target'][m], d['prevcurr'][m], d['err'][m]; odd = np.arange(m.sum()) % 2 == 1
            r[t] = cell(th, pc, e) + cell(th[odd], pc[odd], e[odd])[:2] + cell(th[~odd], pc[~odd], e[~odd])[:2]
        if ok: R[s] = r
    return R

def ci(M, fn, rng):
    M = np.asarray(M, float); n = len(M); bs = np.array([fn(M[rng.integers(0, n, n)].mean(0)) for _ in range(NBOOT)]); return [float(fn(M.mean(0))), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]

def spearman(a, b):
    r = lambda v: np.argsort(np.argsort(v)).astype(float); return float(np.corrcoef(r(a), r(b))[0, 1])

def growth(R, subs, rng):
    """П1: наблюдаемое beta(3) и два предсказания. Столбцы: beta0, beta1, beta3, V0, V1, V3."""
    M = np.array([[R[s][t][0] for t in DELAYS] + [R[s][t][2] for t in DELAYS] for s in subs])
    unc = lambda m: m[0] + (m[1] - m[0]) * (m[5] - m[3]) / (m[4] - m[3]); spd = lambda m: m[0] + 3 * (m[1] - m[0])
    return dict(n=len(subs), beta=[ci(M[:, [i]], lambda m: m[0], rng) for i in range(3)], V=[float(M[:, 3 + i].mean()) for i in range(3)], pred_unc=ci(M, unc, rng), pred_speed=ci(M, spd, rng),
                obs_minus_unc=ci(M, lambda m: m[2] - unc(m), rng), obs_minus_speed=ci(M, lambda m: m[2] - spd(m), rng))

def analyse(base, post):
    rng = np.random.default_rng(95); out = {}
    stage('оценки у каждого человека: исходные сессии'); RB = per_subject(base); stage('оценки у каждого человека: сессии post'); RP = per_subject(post)
    stage('проверка правил (бутстрэп по людям, %d повторов)' % NBOOT)
    out['P1'] = dict(post=growth(RP, list(RP), rng), base_C=growth(RB, [s for s in RB if RB[s]['group'] == 'C'], rng), base_all=growth(RB, list(RB), rng))
    late = lambda r, j: (r[3.0][j] - r[1.0][j]) / (2.0 if j == 1 else 1.0)        # j = 1: дрейф к оси, град/с; j = 0: прирост притяжения за две секунды
    G = {}
    for g in ('C', 'S', 'E'):
        subs = [s for s in RB if RB[s]['group'] == g]
        if subs: G[g] = dict(n=len(subs), L=ci(np.array([[late(RB[s], 1)] for s in subs]), lambda m: m[0], rng), dbeta=ci(np.array([[late(RB[s], 0)] for s in subs]), lambda m: m[0], rng),
                             s2=[float(np.mean([RB[s][t][1] for s in subs])) for t in DELAYS], _L=[late(RB[s], 1) for s in subs])
    def diff(a, b):
        a = np.array(G[a]['_L']); b = np.array(G[b]['_L']); bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(NBOOT)]
        return [float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out['P2'] = dict(groups={g: {k: v for k, v in G[g].items() if k != '_L'} for g in G}, C_minus_S=diff('C', 'S') if 'S' in G and 'C' in G else None)
    subs = list(RB); grp = np.array([RB[s]['group'] for s in subs]); cen = lambda v: v - np.array([v[grp == g].mean() for g in grp])
    A = cen(np.array([late(RB[s], 0) for s in subs])); B = cen(np.array([late(RB[s], 1) for s in subs])); n = len(subs); bs = []
    for _ in range(NBOOT): i = rng.integers(0, n, n); bs.append(spearman(A[i], B[i]))
    half = lambda j: (cen(np.array([RB[s][3.0][3 + j] - RB[s][1.0][3 + j] for s in subs])), cen(np.array([RB[s][3.0][5 + j] - RB[s][1.0][5 + j] for s in subs])))
    rel = lambda j: (lambda r: float(2 * r / (1 + r)))(np.corrcoef(*half(j))[0, 1])
    out['P3'] = dict(n=n, rho=[spearman(A, B), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], rel_beta=rel(0), rel_axis=rel(1))
    p = out['P1']['post']; v1 = p['obs_minus_unc'][1] > 0; d = out['P2']['C_minus_S']; v2 = bool(d and d[1] > 0); v3 = out['P3']['rho'][1] > 0
    out['verdict'] = dict(P1='подтверждено' if v1 else 'не подтверждено', P2='подтверждено' if v2 else 'не подтверждено', P3='подтверждено' if v3 else 'не подтверждено',
                          speed_compatible=bool(p['obs_minus_speed'][1] <= 0 <= p['obs_minus_speed'][2]),
                          outcome='гипотеза о двух родах смещений и её связи с шизофренией поддержана' if (v1 and v2) else ('притяжение к прошлому есть дрейф; о больных утверждать нельзя' if v1 else 'путь закрыт'))
    return out


def f3(x, nd=2): return '%+.*f [%+.*f; %+.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def report(out, outdir, tag=''):
    v = out['verdict']; L = ['# Два рода смещений: проверка%s\n' % tag]
    L.append('- среда: Python %s, numpy %s, %s; время %.0f с; в скобках 95 %% интервал (бутстрэп по людям)\n' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    L.append('## П1. Закон роста притяжения к прошлой пробе\n'); L.append('| набор | людей | beta: 0 с | beta: 1 с | beta: 3 с (наблюдается) | «вслед за неопределённостью» | наблюдаемое минус оно | «постоянная скорость» | наблюдаемое минус она |\n|---|---|---|---|---|---|---|---|---|')
    for key, nm in (('post', 'сессии post (основная проверка)'), ('base_C', 'исходные, здоровые (уже видено)'), ('base_all', 'исходные, все (уже видено)')):
        p = out['P1'][key]; L.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s |' % (nm, p['n'], f3(p['beta'][0]), f3(p['beta'][1]), f3(p['beta'][2]), f3(p['pred_unc']), f3(p['obs_minus_unc']), f3(p['pred_speed']), f3(p['obs_minus_speed'])))
    L.append('\n**Вывод по правилу: %s.** «Постоянная скорость» с данными сессий post %s.' % (v['P1'], 'совместима' if v['speed_compatible'] else 'не совместима'))
    L.append('\n## П2. Поздний дрейф по группам (исходные сессии)\n'); L.append('| группа | людей | s2: 0 / 1 / 3 с, град | поздний дрейф к оси L, град/с | поздний прирост притяжения к прошлому, град |\n|---|---|---|---|---|')
    for g, x in out['P2']['groups'].items(): L.append('| %s | %d | %+.2f / %+.2f / %+.2f | %s | %s |' % (GN.get(g, g), x['n'], *x['s2'], f3(x['L']), f3(x['dbeta'])))
    if out['P2']['C_minus_S']: L.append('\nРазность позднего дрейфа к оси, здоровые минус шизофрения: %s град/с.' % f3(out['P2']['C_minus_S']))
    L.append('\n**Вывод по правилу: %s.**' % v['P2'])
    p3 = out['P3']; L.append('\n## П3. Связь двух дрейфов у одного человека\n'); L.append('- ранговая корреляция (людей %d): %s; надёжность позднего прироста притяжения %.2f, позднего дрейфа к оси %.2f' % (p3['n'], f3(p3['rho']), p3['rel_beta'], p3['rel_axis']))
    L.append('\n**Вывод по правилу: %s.**' % v['P3']); L.append('\n## Исход\n'); L.append('**%s.**' % v['outcome'].capitalize())
    L.append('\n## Сводка для машинного чтения\n'); L.append('```json\n' + json.dumps(out, ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False); return text


def synth(world, seed, groups):
    """'two': притяжение к прошлому и дрейф к оси нарастают после первой секунды, у больных (S) их нет, у человека они связаны.
    'one': притяжение идёт вслед за неопределённостью, групп и связи нет, дрейфа к оси нет."""
    rng = np.random.default_rng(seed); S = {k: [] for k in ('subject', 'group', 'delay', 'target', 'prevcurr', 'err')}
    for g, ns in groups:
        for k in range(ns):
            n = 1100; th = rng.uniform(-np.pi, np.pi, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); pc = np.angle(np.exp(1j * (np.roll(th, 1) - th))); V = np.exp(rng.normal(0, 0.25)) * (10 + 12 * (dl > 0) + 4 * dl)
            if world == 'two': f = (0.0 if g == 'S' else np.exp(rng.normal(0, 0.5))); late = np.clip(dl - 1, 0, None); b = -0.5 + f * (0.3 * (dl > 0) + 1.0 * late); a2 = f * 0.5 * late
            else: b = -0.5 + 0.09 * (V - 10); a2 = 0.0
            e = b * W.dog(pc) + a2 * np.sin(2 * (0.0 - th)) - (V / (V + 300)) * W.dist_to_diag(th) + rng.normal(0, np.sqrt(V))
            for key, v in zip(S, ([g + '%02d' % k] * n, [g] * n, dl, th, pc, e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}

def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--retest', default=os.path.join('data', 'behavior_retest.pkl')); ap.add_argument('--out', default='wm_stein9_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА. Мир «два рода»: ожидается подтверждение всех трёх правил. Мир «один род»: ни одного.')
        for world, nm in (('two', 'два рода'), ('one', 'один род')):
            for seed in (1, 2):
                o = analyse(synth(world, seed, (('C', 19), ('S', 17), ('E', 16))), synth(world, 50 + seed, (('C', 8), ('E', 14)))); v = o['verdict']
                print('  мир «%s», набор %d: П1 %s | П2 %s | П3 %s (корреляция %.2f) -> %s' % (nm, seed, v['P1'], v['P2'], v['P3'], o['P3']['rho'][0], v['outcome']), flush=True)
        return
    stage('читаю %s' % a.data); base, _ = W.load_real(a.data); stage('читаю %s' % a.retest); d2, _ = W3.load(a.retest); m = d2['session'] == 'post'; post = {k: v[m] for k, v in d2.items() if k != 'session'}
    out = analyse(base, post); text = report(out, a.out); print('\n' + text[:text.find('## Сводка для машинного чтения')]); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
