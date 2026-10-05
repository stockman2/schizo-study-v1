# -*- coding: utf-8 -*-
"""
wm_stein.py — первый взгляд на открытые данные Stein, Barbosa и соавт. (Nat Commun 2020) о рабочей памяти.

ОПЫТ. Человеку на 0,25 с показывают точку на окружности, затем пауза 0, 1 или 3 с, затем он щёлкает мышью туда,
где была точка. Ошибка = угол между ответом и точкой. Группы: C — здоровые (19), S — шизофрения (17),
E — анти-NMDA-рецепторный энцефалит (16). Около 1000 проб на человека.

ТРИ ВОПРОСА (и правила чтения ответов, записанные до расчёта)
  В1. Что растёт с паузой — случайный разброс или систематический, зависящий от места точки, снос?
      Ошибка каждой пробы раскладывается на: влияние прошлой пробы; «карту сноса» (среднюю ошибку как функцию места
      точки; её дисперсия оценивается несмещённо по двум независимым половинам проб); остаток — случайный разброс.
      Правило: доля систематической части в ПРИРОСТЕ дисперсии между 1 и 3 с:
        <= 0,2  -> рост определяется случайным разбросом (шум);
        >= 0,5  -> рост определяется систематическим сносом (неоднородность);
        иначе   -> вклад дают оба.
  В2. По какому закону растёт случайная часть? R = [V(3) - V(0)] / [V(1) - V(0)], V — дисперсия случайной части.
        чистая диффузия от момента исчезновения точки: R = 3;
        R < 3: есть быстрая начальная потеря (переход от восприятия к памяти) либо насыщение;
        R > 3: рост ускоряется (след уезжает с постоянной скоростью: R = 9).
      Оговорка: при паузе 0 с точка остаётся на экране до начала ответа, то есть памяти там нет вовсе.
      Поэтому R < 3 нельзя однозначно отнести к диффузии или не-диффузии; двух ненулевых пауз для этого мало.
  В3. Противоречит ли отсутствие различий в точности у Stein результату Gold 2020 (наклон разброса у больных
      больше на 0,16 град/с)? Считается разность наклонов (S минус C) между 1 и 3 с и её 95 % интервал.
        интервал содержит 0,16 -> противоречия нет (у Stein просто мало данных для такого эффекта);
        интервал лежит ниже 0,16 -> противоречие настоящее.
  Сверка с опубликованным: число проб после отбора должно быть 52 394; у здоровых прошлое притягивает и тем сильнее,
      чем длиннее пауза; у больных шизофренией притяжения нет.

ЗАПУСК
  pip install numpy pandas matplotlib
  python wm_stein.py --selftest        # проверка метода на искусственных данных с известным ответом (несколько секунд)
  python wm_stein.py                   # анализ настоящих данных; файл data/behavior.pkl (13 МБ) скачается сам
  Результат: папка wm_stein_out/ — report.md, results.json, figures.png
  Файл данных — это pickle из репозитория авторов github.com/comptelab/serialNMDA; pickle исполняет код при чтении,
  поэтому берите его только оттуда.
"""
import sys, os, json, time, argparse, platform
import numpy as np

URL = 'https://github.com/comptelab/serialNMDA/raw/master/data/behavior.pkl'
GROUPS = ['C', 'S', 'E']; GNAME = {'C': 'здоровые', 'S': 'шизофрения', 'E': 'энцефалит'}
DELAYS = [0.0, 1.0, 3.0]; SIGMA_DOG = 0.8; NBINS = 24; NSPLIT = 60; NBOOT = 2000
GOLD_DIFF = 0.16; GOLD_C, GOLD_S = 0.24, 0.40; STARC_C, STARC_S = 0.15, 0.26
T0 = time.time()


def stage(msg):
    print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)


def dog(x, sigma=SIGMA_DOG):
    """Форма влияния прошлой пробы: производная гауссианы, нормированная на максимум 1; положительна при x > 0."""
    return x * np.exp(-x * x / (2 * sigma * sigma)) / (sigma * np.exp(-0.5))


# ---------- данные ----------
def load_real(path):
    import pandas as pd
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        stage('файла %s нет — скачиваю из репозитория авторов (13 МБ)' % path)
        urllib.request.urlretrieve(URL, path)
    d = pd.read_pickle(path); n0 = len(d)
    keep = (d.RT < 3) & (d.ITI < 5) & (d.raderror < 5) & (d.error.abs() < 1)          # отбор, как у авторов
    d = d[keep.values]; n1 = len(d); d = d[(d.trial % 48 != 1).values]                  # первая проба каждого блока отбрасывается
    out = dict(subject=d.subject.values.astype(str), group=d.group.values.astype(str), delay=d.delay.astype(int).values / 60.0,
               target=d.target.values.astype(float), prevcurr=d.prevcurr.values.astype(float), err=np.degrees(d.error.values.astype(float)))
    return out, dict(n_raw=n0, n_after_filter=n1, n_final=len(d))


def synth(seed=1):
    """Искусственные данные с известным ответом — для проверки метода."""
    rng = np.random.default_rng(seed); S = dict(subject=[], group=[], delay=[], target=[], prevcurr=[], err=[])
    truth = dict(D=dict(C=1.0, S=2.0, E=1.5), V0=9.0, Venc=4.0, drift=0.6, b0=1.0,
                 beta=dict(C={0.0: -0.5, 1.0: 0.8, 3.0: 2.0}, S={0.0: -0.5, 1.0: -0.5, 3.0: -0.5}, E={0.0: -0.5, 1.0: 0.3, 3.0: 1.0}))
    for g, ns in (('C', 19), ('S', 17), ('E', 16)):
        for k in range(ns):
            n = 1050; th = rng.uniform(-np.pi, np.pi, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); ph = rng.uniform(0, 2 * np.pi)
            pc = np.angle(np.exp(1j * (np.roll(th, 1) - th)))
            sysb = truth['b0'] * np.sin(4 * th) + truth['drift'] * dl * np.sin(2 * th + ph)        # карта при 0 с и снос, растущий с паузой
            ser = np.array([truth['beta'][g][t] for t in dl]) * dog(pc)
            var = truth['V0'] + truth['Venc'] * (dl > 0) + 2 * truth['D'][g] * dl
            e = sysb + ser + rng.normal(0, np.sqrt(var)) + rng.normal(0, 0.5)                       # последний член — личная постоянная ошибка
            for key, v in zip(('subject', 'group', 'delay', 'target', 'prevcurr', 'err'), ([g + '%02d' % k] * n, [g] * n, dl, th, pc, e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}, truth


# ---------- разложение ошибки ----------
def split_half_sys_var(theta, e, rng):
    """Несмещённая оценка дисперсии «карты сноса»: ковариация средних по ячейкам места в двух независимых половинах проб."""
    b = np.minimum(((theta + np.pi) / (2 * np.pi) * NBINS).astype(int), NBINS - 1); acc = []
    for _ in range(NSPLIT):
        h = rng.random(len(e)) < 0.5
        s1 = np.bincount(b[h], e[h], NBINS); n1 = np.bincount(b[h], minlength=NBINS)
        s2 = np.bincount(b[~h], e[~h], NBINS); n2 = np.bincount(b[~h], minlength=NBINS)
        ok = (n1 > 0) & (n2 > 0)
        if ok.sum() < NBINS // 2: continue
        w = (n1 + n2)[ok].astype(float); acc.append(np.sum(w * (s1[ok] / n1[ok]) * (s2[ok] / n2[ok])) / w.sum())
    return float(np.mean(acc)) if acc else np.nan


def bias_map(theta, e):
    b = np.minimum(((theta + np.pi) / (2 * np.pi) * NBINS).astype(int), NBINS - 1)
    return np.bincount(b, e, NBINS) / np.maximum(np.bincount(b, minlength=NBINS), 1)


def per_subject(d):
    """Для каждого человека и каждой паузы: сила влияния прошлой пробы, дисперсии (вся, систематическая, случайная), карта сноса."""
    rng = np.random.default_rng(2026); rows = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; g = d['group'][ms][0]; r = dict(group=g, n={}, beta={}, Vtot={}, Vsys={}, Vrand={}, sd={}, map={})
        for t in DELAYS:
            m = ms & (d['delay'] == t); e = d['err'][m]; n = int(m.sum()); r['n'][t] = n
            if n < 60:
                for k in ('beta', 'Vtot', 'Vsys', 'Vrand', 'sd'): r[k][t] = np.nan
                r['map'][t] = np.full(NBINS, np.nan); continue
            e = e - e.mean(); x = dog(d['prevcurr'][m]); beta = float(np.sum(e * x) / np.sum(x * x)); e1 = e - beta * x   # убрано влияние прошлой пробы
            vt = float(e1.var()); vs = split_half_sys_var(d['target'][m], e1, rng) + vt / n      # + vt/n: поправка на вычитание среднего
            r['beta'][t] = beta; r['Vtot'][t] = vt; r['Vsys'][t] = vs; r['Vrand'][t] = vt - vs; r['sd'][t] = float(np.sqrt(vt)); r['map'][t] = bias_map(d['target'][m], e1)
        rows[s] = r
    return rows


def boot(vals, fn, rng, nb=NBOOT):
    """Среднее по людям и 95 % интервал бутстрэпом по людям. vals — массив (люди x величины), fn — функция от средних."""
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals).any(1)]; n = len(vals)
    est = fn(vals.mean(0)); bs = np.array([fn(vals[rng.integers(0, n, n)].mean(0)) for _ in range(nb)])
    return float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n


def analyse(d, log):
    stage('разложение ошибки по каждому человеку и каждой паузе (%d половинных разбиений на каждую оценку)' % NSPLIT)
    R = per_subject(d); rng = np.random.default_rng(7); out = dict(groups={})
    stage('средние по группам и интервалы (бутстрэп по людям, %d повторов)' % NBOOT)
    for g in GROUPS:
        subs = [s for s in R if R[s]['group'] == g]; G = dict(n_subjects=len(subs), n_trials={str(t): int(sum(R[s]['n'][t] for s in subs)) for t in DELAYS})
        arr = lambda key: np.array([[R[s][key][t] for t in DELAYS] for s in subs])
        for key in ('sd', 'Vtot', 'Vsys', 'Vrand', 'beta'):
            A = arr(key); G[key] = {str(t): boot(A[:, [i]], lambda m: m[0], rng)[:3] for i, t in enumerate(DELAYS)}
        Vt, Vs, Vr, SD = arr('Vtot'), arr('Vsys'), arr('Vrand'), arr('sd')
        G['share_sys_growth'] = boot(np.column_stack([Vs[:, 2] - Vs[:, 1], Vt[:, 2] - Vt[:, 1]]), lambda m: m[0] / m[1] if m[1] != 0 else np.nan, rng)[:3]
        G['R'] = boot(np.column_stack([Vr[:, 2] - Vr[:, 0], Vr[:, 1] - Vr[:, 0]]), lambda m: m[0] / m[1] if m[1] > 0 else np.nan, rng)[:3]
        G['D'] = boot((Vr[:, [2]] - Vr[:, [1]]) / 4.0, lambda m: m[0], rng)[:3]                   # град^2/с: прирост дисперсии за 2 с, делённый на 2*2
        G['Venc'] = boot(np.column_stack([Vr[:, 1] - Vr[:, 0], (Vr[:, 2] - Vr[:, 1]) / 2.0]), lambda m: m[0] - m[1], rng)[:3]
        G['slope_sd'] = boot((SD[:, [2]] - SD[:, [1]]) / 2.0, lambda m: m[0], rng)[:3]             # град/с между 1 и 3 с
        G['_slopes'] = ((SD[:, 2] - SD[:, 1]) / 2.0).tolist(); G['_sd0'] = SD[:, 0].tolist()
        maps = np.array([R[s]['map'][1.0] for s in subs]); maps = maps[~np.isnan(maps).any(1)]
        c = np.corrcoef(maps); G['map_similarity_between_subjects'] = float(c[np.triu_indices(len(maps), 1)].mean()) if len(maps) > 2 else np.nan
        cc = [np.corrcoef(R[s]['map'][1.0], R[s]['map'][3.0])[0, 1] for s in subs if not np.isnan(R[s]['map'][3.0]).any()]
        G['map_corr_1s_3s_within_subject'] = float(np.nanmean(cc)) if cc else np.nan
        G['mean_map'] = {str(t): np.nanmean([R[s]['map'][t] for s in subs], 0).tolist() for t in DELAYS}
        out['groups'][g] = G
    # разности между группами (S минус C)
    def diff(key, nb=NBOOT):
        a = np.array(out['groups']['S'][key]); b = np.array(out['groups']['C'][key]); a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(nb)]
        return float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    out['slope_diff_S_minus_C'] = diff('_slopes'); out['sd0_diff_S_minus_C'] = diff('_sd0')
    return out


# ---------- отчёт ----------
def f3(x, nd=2): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def report(out, meta, outdir, synthetic=False):
    L = ['# Данные Stein 2020: что растёт с паузой%s\n' % (' — ИСКУССТВЕННЫЕ ДАННЫЕ, проверка метода' if synthetic else '')]
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    if meta: L.append('- проб в файле %d, после отбора авторов %d, после удаления первых проб блоков %d (в статье 52 394: %s)' % (
        meta['n_raw'], meta['n_after_filter'], meta['n_final'], 'совпало' if meta['n_final'] == 52394 else '**НЕ совпало**'))
    L.append('- все величины в градусах; в скобках 95 % интервал (бутстрэп по людям)\n')
    G = out['groups']
    L.append('## Сверка: влияние прошлой пробы (градусы в максимуме; плюс — притяжение, минус — отталкивание)\n')
    L.append('| группа | людей | пауза 0 с | пауза 1 с | пауза 3 с |\n|---|---|---|---|---|')
    for g in GROUPS: L.append('| %s | %d | %s | %s | %s |' % (GNAME[g], G[g]['n_subjects'], f3(G[g]['beta']['0.0']), f3(G[g]['beta']['1.0']), f3(G[g]['beta']['3.0'])))
    L.append('\n## Разброс ошибки (стандартное отклонение после вычета влияния прошлой пробы)\n')
    L.append('| группа | 0 с | 1 с | 3 с | наклон между 1 и 3 с, град/с |\n|---|---|---|---|---|')
    for g in GROUPS: L.append('| %s | %s | %s | %s | %s |' % (GNAME[g], f3(G[g]['sd']['0.0']), f3(G[g]['sd']['1.0']), f3(G[g]['sd']['3.0']), f3(G[g]['slope_sd'], 3)))
    L.append('\nДля сравнения, наклоны из литературы: Gold 2020 — здоровые %.2f, больные %.2f; Starc 2017 — %.2f и %.2f град/с.' % (GOLD_C, GOLD_S, STARC_C, STARC_S))
    L.append('\n## В1. Из чего состоит дисперсия ошибки (град²)\n')
    L.append('| группа | пауза | вся | систематическая (карта сноса) | случайная |\n|---|---|---|---|---|')
    for g in GROUPS:
        for t in ('0.0', '1.0', '3.0'): L.append('| %s | %s с | %s | %s | %s |' % (GNAME[g], t[0], f3(G[g]['Vtot'][t], 1), f3(G[g]['Vsys'][t], 2), f3(G[g]['Vrand'][t], 1)))
    L.append('\n| группа | доля систематической части в приросте между 1 и 3 с | вывод по правилу | сходство карт между людьми (1 с) | сходство карт 1 с и 3 с у одного человека |\n|---|---|---|---|---|')
    verdict = {}
    for g in GROUPS:
        sh = G[g]['share_sys_growth']; v = 'шум' if sh[0] <= 0.2 else ('неоднородность' if sh[0] >= 0.5 else 'оба'); verdict[g] = v
        L.append('| %s | %s | **%s** | %.2f | %.2f |' % (GNAME[g], f3(sh), v, G[g]['map_similarity_between_subjects'], G[g]['map_corr_1s_3s_within_subject']))
    L.append('\n## В2. Закон роста случайной части\n')
    L.append('| группа | R (3 — диффузия, 9 — постоянная скорость) | вывод по правилу | D между 1 и 3 с, град²/с | начальная потеря, град² |\n|---|---|---|---|---|')
    for g in GROUPS:
        r = G[g]['R']; v = 'совместимо с диффузией без начальной потери' if r[1] <= 3 <= r[2] else ('есть начальная потеря или насыщение' if r[2] < 3 else 'рост ускоряется')
        L.append('| %s | %s | %s | %s | %s |' % (GNAME[g], f3(r), v, f3(G[g]['D']), f3(G[g]['Venc'], 1)))
    L.append('\nНапоминание: при паузе 0 с памяти нет (точка на экране), поэтому «начальная потеря» может быть просто ценой перехода от восприятия к памяти.')
    sd = out['slope_diff_S_minus_C']; s0 = out['sd0_diff_S_minus_C']
    v3 = 'противоречия с Gold нет' if sd[1] <= GOLD_DIFF <= sd[2] else ('противоречие настоящее: эффект Gold выше интервала' if sd[2] < GOLD_DIFF else 'эффект больше, чем у Gold')
    L.append('\n## В3. Совместимость с Gold 2020\n')
    L.append('- разность наклонов разброса, шизофрения минус здоровые: %s град/с; у Gold %.2f. **%s.**' % (f3(sd, 3), GOLD_DIFF, v3.capitalize()))
    L.append('- разность разброса при паузе 0 с, шизофрения минус здоровые: %s град (у Starc и Gold больные хуже уже здесь).' % f3(s0))
    L.append('\n## Сводка для машинного чтения\n')
    slim = {g: {k: v for k, v in G[g].items() if not k.startswith('_') and k != 'mean_map'} for g in GROUPS}
    L.append('```json\n' + json.dumps(dict(synthetic=synthetic, meta=meta, groups=slim, slope_diff_S_minus_C=sd, sd0_diff_S_minus_C=s0, verdict_Q1=verdict, verdict_Q3=v3), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text)
    json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        col = {'C': 'k', 'S': '#c0392b', 'E': '#2471a3'}; fig, ax = plt.subplots(1, 4, figsize=(17, 4))
        for g in GROUPS:
            for a, key, ttl, yl in ((ax[0], 'sd', 'Разброс ошибки', 'град'), (ax[1], 'Vrand', 'Случайная часть дисперсии', 'град²'), (ax[2], 'beta', 'Влияние прошлой пробы (+ притяжение)', 'град')):
                m = np.array([G[g][key][str(t)] for t in DELAYS]); a.errorbar(DELAYS, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o', color=col[g], capsize=3, label=GNAME[g])
                a.set_title(ttl, fontsize=10); a.set_xlabel('пауза, с'); a.set_ylabel(yl)
            x = (np.arange(NBINS) + 0.5) * 360 / NBINS - 180; ax[3].plot(x, G[g]['mean_map']['3.0'], color=col[g], label=GNAME[g] + ', 3 с'); ax[3].plot(x, G[g]['mean_map']['0.0'], color=col[g], ls=':', lw=0.8)
        ax[2].axhline(0, color='gray', lw=0.6); ax[3].axhline(0, color='gray', lw=0.6); ax[3].set_title('Средняя карта сноса (пунктир — пауза 0 с)', fontsize=10); ax[3].set_xlabel('место точки, град'); ax[3].set_ylabel('средняя ошибка, град')
        ax[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=130)
    except ImportError:
        stage('matplotlib не установлен — рисунок пропущен')
    return text


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--out', default='wm_stein_out')
    ap.add_argument('--selftest', action='store_true', help='проверка метода на искусственных данных с известным ответом')
    ap.add_argument('--check-data', action='store_true', help='только загрузить данные и показать число проб, без анализа')
    a = ap.parse_args()
    if a.selftest:
        stage('создаю три независимых искусственных набора с известным ответом и объединяю их')
        parts = []
        for sd_ in (1, 2, 3):
            d_, truth = synth(sd_); d_['subject'] = np.array(['%d_%s' % (sd_, x) for x in d_['subject']]); parts.append(d_)
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}; out = analyse(d, None)
        text = report(out, None, a.out + '_selftest', synthetic=True); print('\n' + text)
        G = out['groups']; ok = True; chk = []; att = lambda k: (np.sin(k * np.pi / NBINS) / (k * np.pi / NBINS)) ** 2    # ослабление от усреднения по ячейкам
        tv = lambda t: truth['b0'] ** 2 / 2 * att(4) + (truth['drift'] * t) ** 2 / 2 * att(2)
        for g in GROUPS:
            chk.append(('D, ' + GNAME[g], G[g]['D'], truth['D'][g]))
            chk.append(('начальная потеря, ' + GNAME[g], G[g]['Venc'], truth['Venc']))
            for t in DELAYS: chk.append(('влияние прошлой пробы при %g с, %s' % (t, GNAME[g]), G[g]['beta'][str(t)], truth['beta'][g][t]))
            for t in DELAYS: chk.append(('дисперсия карты сноса при %g с, %s' % (t, GNAME[g]), G[g]['Vsys'][str(t)], tv(t)))
        print('ПРОВЕРКА МЕТОДА: получено [интервал] / заложено / расхождение в погрешностях')
        zs = []
        for name, got, want in chk:
            se = max((got[2] - got[1]) / 3.92, 1e-9); z = abs(got[0] - want) / se; zs.append(z)
            print('  %-50s %6.2f [%6.2f; %6.2f] / %5.2f / %.1f%s' % (name, got[0], got[1], got[2], want, z, '' if z <= 3 else '  <-- больше трёх'))
        zs = np.array(zs); ok = (zs <= 4).all() and (zs > 3).sum() <= 1     # из 24 проверок одна может случайно выйти за три погрешности
        print('Правило: ни одного расхождения больше 4 погрешностей и не больше одного — больше 3.')
        print('ИТОГ ПРОВЕРКИ: %s' % ('метод восстанавливает заложенные величины' if ok else 'ЕСТЬ РАСХОЖДЕНИЯ — анализу настоящих данных доверять нельзя')); return
    stage('читаю %s' % a.data); d, meta = load_real(a.data)
    stage('проб: в файле %d, после отбора %d, итог %d; людей %d' % (meta['n_raw'], meta['n_after_filter'], meta['n_final'], len(np.unique(d['subject']))))
    if a.check_data: return
    out = analyse(d, None); text = report(out, meta, a.out)
    print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
