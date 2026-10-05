# -*- coding: utf-8 -*-
"""
wm_stein3.py — повтор находок на независимых данных: повторные сессии из того же репозитория (Stein и соавт., 2020).

ДАННЫЕ. Файл data/behavior_retest.pkl: 22 человека (8 здоровых, 14 с энцефалитом), у каждого две сессии —
'pre' (исходная, она уже входила в прежние расчёты) и 'post' (через 3-12 месяцев; эти пробы мы ещё не трогали).
Больных шизофренией в файле нет, поэтому находку про их разброс у осей здесь проверить нельзя.
Люди те же, что раньше, а пробы новые: это проверка устойчивости закономерностей, а не перенос на новых людей.

ЧТО ПРОВЕРЯЕТСЯ (правила записаны до расчёта; lam — сила стягивания к диагонали, V — случайная дисперсия)
  Р1. Главный результат прошлого скрипта. На сессиях 'post', все 22 человека: наблюдаемое lam(3 с) против двух
      предсказаний (динамическое с постоянной скоростью и «через вывод»), формулы те же, что в wm_stein2.py.
      Повторилось, если снова совместимо только «через вывод».
  Р2. Постоянство отношения Q = lam*(1-lam)/V при трёх паузах (в прошлый раз замечено после расчёта).
      Повторилось, если оба 95 % интервала — для Q(1 с)/Q(0 с) и для Q(3 с)/Q(1 с) — содержат единицу.
  Р3. Случайный разброс у осей растёт с паузой быстрее, чем у диагоналей. Отношение «дисперсия у оси / у диагонали»:
      повторилось, если разность этого отношения при 3 с и при 0 с положительна и её интервал не содержит ноль.
  Р4. Направленное предсказание из гипотезы «больные слабее опираются на привычное представление о месте».
      При энцефалите влияние прошлой пробы нормализуется с выздоровлением (это показано в статье). Если стягивание
      к диагонали той же природы, то у больных Q при паузе 1 с должно вырасти от 'pre' к 'post'.
        интервал разности Q(post) - Q(pre) у больных целиком выше нуля -> предсказание подтверждено;
        интервал содержит ноль -> не подтверждено; целиком ниже нуля -> опровергнуто.
      У здоровых изменения не ожидается (печатается для сравнения).

ЗАПУСК (нужен wm_stein2.py в той же папке — оттуда берутся оценки)
  python wm_stein3.py --selftest     # два искусственных мира при том же числе людей: что метод способен различить
  python wm_stein3.py                # настоящие данные; файл (9 МБ) скачается сам
  Результат: папка wm_stein3_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform
import numpy as np
import wm_stein2 as W

URL = 'https://github.com/comptelab/serialNMDA/raw/master/data/behavior_retest.pkl'
DELAYS = W.DELAYS; GN = {'C': 'здоровые', 'E': 'энцефалит', 'ALL': 'все вместе'}; T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)


def load(path):
    import pandas as pd
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True); stage('файла %s нет — скачиваю из репозитория авторов (9 МБ)' % path); urllib.request.urlretrieve(URL, path)
    d = pd.read_pickle(path); n0 = len(d)
    d = d[((d.RT < 3) & (d.ITI < 5) & (d.raderror < 5) & (d.error.abs() < 1)).values]; d = d[(d.trial % 48 != 1).values]
    out = dict(subject=d.subject.values.astype(str), group=d.group.values.astype(str), session=d.session.values.astype(str), delay=d.delay.astype(int).values / 60.0,
               target=d.target.values.astype(float), prevcurr=d.prevcurr.values.astype(float), err=np.degrees(d.error.values.astype(float)))
    return out, dict(n_raw=n0, n_final=len(d), n_post=int((out['session'] == 'post').sum()), n_pre=int((out['session'] == 'pre').sum()))


def synth(world, seed):
    """Искусственный мир с двумя сессиями. 'inf': стягивание зависит от шума, разброс у осей растёт с паузой, у больных
    «ширина квадранта» уменьшается от pre к post. 'dyn': постоянная скорость сноса, разброс не зависит от места, изменений нет."""
    rng = np.random.default_rng(seed); S = {k: [] for k in ('subject', 'group', 'session', 'delay', 'target', 'prevcurr', 'err')}
    for g, ns in (('C', 8), ('E', 14)):
        for k in range(ns):
            noise = np.exp(rng.normal(0, 0.3)) * (1.4 if g == 'E' else 1.0); vp0 = 240.0 * np.exp(rng.normal(0, 0.3)); kk = 0.088 * np.exp(rng.normal(0, 0.3))
            for ses in ('pre', 'post'):
                n = 1100; th = rng.uniform(-np.pi, np.pi, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); pc = np.angle(np.exp(1j * (np.roll(th, 1) - th))); dlt = W.dist_to_diag(th)
                V = noise * (10.0 + 10.0 * (dl > 0) + 4.0 * dl)
                if world == 'inf':
                    vp = vp0 * (2.0 if (g == 'E' and ses == 'pre') else (1.3 if g == 'E' else 1.0)); lam = V / (V + vp)
                    a = np.array([{0.0: 0.15, 1.0: 0.45, 3.0: 0.8}[t] for t in dl]); sd = np.sqrt(V * (1 + a * (np.abs(dlt) / 45.0) ** 2) / (1 + a / 3))
                    e = -lam * dlt + rng.normal(0, 1, n) * sd * (1 - lam)
                else:
                    lam = 1 - 0.95 * np.exp(-kk * dl); e = -lam * dlt + rng.normal(0, np.sqrt(V))
                e = e + 0.8 * np.sin(th + 1.0)
                for key, v in zip(S, ([g + '%02d' % k] * n, [g] * n, [ses] * n, dl, th, pc, e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}


def analyse(d):
    R = {}
    for ses in ('pre', 'post'):
        stage('оценки у каждого человека, сессия %s' % ses); m = d['session'] == ses; R[ses] = W.per_subject({k: v[m] for k, v in d.items() if k != 'session'})
    stage('интервалы (бутстрэп по людям, %d повторов) и проверка правил' % W.NBOOT); rng = np.random.default_rng(31); out = dict(sets={}, change={}); P = R['post']
    Q = lambda lam, v: lam * (1 - lam) / v
    for g in ('C', 'E', 'ALL'):
        subs = [s for s in P if g == 'ALL' or P[s]['group'] == g]; A = lambda key: np.array([[P[s][key][t] for t in DELAYS] for s in subs]); lam, V = A('lam'), A('V'); M = np.column_stack([lam, V]); G = dict(n=len(subs))
        G['lam'] = {str(t): W.ci(lam[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)}; G['V'] = {str(t): W.ci(V[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)}
        G['pred_dyn'] = W.ci(M, W.pred_dyn, rng); G['pred_inf'] = W.ci(M, W.pred_inf, rng); G['obs_minus_dyn'] = W.ci(M, lambda m: m[2] - W.pred_dyn(m), rng); G['obs_minus_inf'] = W.ci(M, lambda m: m[2] - W.pred_inf(m), rng)
        G['Q'] = {str(t): W.ci(M, lambda m, i=i: Q(m[i], m[3 + i]), rng) for i, t in enumerate(DELAYS)}; G['Vp'] = {str(t): W.ci(M, lambda m, i=i: 1 / Q(m[i], m[3 + i]) if m[i] > 0 else np.nan, rng) for i, t in enumerate(DELAYS)}
        G['Q1_over_Q0'] = W.ci(M, lambda m: Q(m[1], m[4]) / Q(m[0], m[3]) if m[0] > 0 else np.nan, rng); G['Q3_over_Q1'] = W.ci(M, lambda m: Q(m[2], m[5]) / Q(m[1], m[4]) if m[1] > 0 else np.nan, rng)
        VL = np.array([[P[s]['vloc'][t] for t in DELAYS] for s in subs]); G['vloc'] = {str(t): VL[:, i].mean(0).tolist() for i, t in enumerate(DELAYS)}
        G['axis_ratio'] = {str(t): W.ci(VL[:, i][:, [0, 2]], lambda m: m[1] / m[0], rng) for i, t in enumerate(DELAYS)}
        G['axis_ratio_growth'] = W.ci(np.column_stack([VL[:, 0][:, [0, 2]], VL[:, 2][:, [0, 2]]]), lambda m: m[3] / m[2] - m[1] / m[0], rng)
        out['sets'][g] = G
    for g in ('C', 'E'):                                    # Р4: изменение от pre к post у тех же людей (пауза 1 с)
        subs = [s for s in P if P[s]['group'] == g and s in R['pre']]; M = np.array([[R['pre'][s]['lam'][1.0], R['pre'][s]['V'][1.0], P[s]['lam'][1.0], P[s]['V'][1.0]] for s in subs])
        out['change'][g] = dict(n=len(subs), Q_pre=W.ci(M, lambda m: Q(m[0], m[1]), rng), Q_post=W.ci(M, lambda m: Q(m[2], m[3]), rng), dQ=W.ci(M, lambda m: Q(m[2], m[3]) - Q(m[0], m[1]), rng),
                                lam_pre=W.ci(M[:, [0]], lambda m: m[0], rng), lam_post=W.ci(M[:, [2]], lambda m: m[0], rng), V_pre=W.ci(M[:, [1]], lambda m: m[0], rng), V_post=W.ci(M[:, [3]], lambda m: m[0], rng))
    G = out['sets']['ALL']; inz = lambda c, x=0.0: c[1] <= x <= c[2]; okd, oki = inz(G['obs_minus_dyn']), inz(G['obs_minus_inf']); dq = out['change']['E']['dQ']; gr = G['axis_ratio_growth']
    out['verdict'] = dict(
        R1='повторилось: совместимо только «через вывод»' if (oki and not okd) else ('НЕ повторилось: совместимо только динамическое' if (okd and not oki) else 'не решено (%s)' % ('оба совместимы' if okd else 'ни одно не совместимо')),
        R2='повторилось: отношение постоянно' if (inz(G['Q1_over_Q0'], 1.0) and inz(G['Q3_over_Q1'], 1.0)) else 'НЕ повторилось: отношение меняется с паузой',
        R3='повторилось: разброс у осей растёт быстрее' if gr[1] > 0 else ('не повторилось' if inz(gr) else 'обратный знак'),
        R4='подтверждено' if dq[1] > 0 else ('не подтверждено' if inz(dq) else 'опровергнуто'))
    return out


def f3(x, nd=3): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def report(out, meta, outdir, tag=''):
    S = out['sets']; v = out['verdict']; L = ['# Повтор на независимых данных: повторные сессии%s\n' % tag]
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    if meta: L.append('- проб после отбора: %d, из них в сессиях post (новые) %d, в сессиях pre %d' % (meta['n_final'], meta['n_post'], meta['n_pre']))
    L.append('- людей: здоровых %d, с энцефалитом %d; в скобках 95 %% интервал (бутстрэп по людям)\n' % (S['C']['n'], S['E']['n']))
    L.append('## Сессии post: сила стягивания и случайная дисперсия\n'); L.append('| набор | lam: 0 с | lam: 1 с | lam: 3 с | V: 0 с | V: 1 с | V: 3 с |\n|---|---|---|---|---|---|---|')
    for g in ('C', 'E', 'ALL'): L.append('| %s | %s | %s | %s | %s | %s | %s |' % (GN[g], f3(S[g]['lam']['0.0']), f3(S[g]['lam']['1.0']), f3(S[g]['lam']['3.0']), f3(S[g]['V']['0.0'], 1), f3(S[g]['V']['1.0'], 1), f3(S[g]['V']['3.0'], 1)))
    L.append('\n## Р1. lam при 3 с против двух предсказаний\n'); L.append('| набор | наблюдается | динамическое | разность | «через вывод» | разность |\n|---|---|---|---|---|---|')
    for g in ('ALL', 'C', 'E'): L.append('| %s | %s | %s | %s | %s | %s |' % (GN[g], f3(S[g]['lam']['3.0']), f3(S[g]['pred_dyn']), f3(S[g]['obs_minus_dyn']), f3(S[g]['pred_inf']), f3(S[g]['obs_minus_inf'])))
    L.append('\n**Вывод по правилу: %s.**' % v['R1'])
    L.append('\n## Р2. Отношение Q = lam·(1 − lam)/V при трёх паузах\n'); L.append('| набор | Q: 0 с | Q: 1 с | Q: 3 с | Q(1)/Q(0) | Q(3)/Q(1) | «ширина квадранта» 1/Q при 1 с, град² |\n|---|---|---|---|---|---|---|')
    for g in ('ALL', 'C', 'E'): L.append('| %s | %s | %s | %s | %s | %s | %s |' % (GN[g], f3(S[g]['Q']['0.0'], 4), f3(S[g]['Q']['1.0'], 4), f3(S[g]['Q']['3.0'], 4), f3(S[g]['Q1_over_Q0'], 2), f3(S[g]['Q3_over_Q1'], 2), f3(S[g]['Vp']['1.0'], 0)))
    L.append('\n**Вывод по правилу: %s.**' % v['R2'])
    L.append('\n## Р3. Случайный разброс у диагоналей и у осей (дисперсия остатка, град²)\n'); L.append('| набор | пауза | у диагонали | середина | у оси | отношение «у оси / у диагонали» |\n|---|---|---|---|---|---|')
    for g in ('ALL', 'C', 'E'):
        for t in ('0.0', '1.0', '3.0'): x = S[g]['vloc'][t]; L.append('| %s | %s с | %.1f | %.1f | %.1f | %s |' % (GN[g], t[0], x[0], x[1], x[2], f3(S[g]['axis_ratio'][t], 2)))
    L.append('\nРост отношения от 0 с к 3 с: все вместе %s; здоровые %s; энцефалит %s.' % (f3(S['ALL']['axis_ratio_growth'], 2), f3(S['C']['axis_ratio_growth'], 2), f3(S['E']['axis_ratio_growth'], 2)))
    L.append('\n**Вывод по правилу: %s.**' % v['R3'])
    L.append('\n## Р4. Изменение от pre к post у тех же людей (пауза 1 с)\n'); L.append('| группа | людей | lam: pre | lam: post | V: pre | V: post | Q: pre | Q: post | разность Q |\n|---|---|---|---|---|---|---|---|---|')
    for g in ('E', 'C'): c = out['change'][g]; L.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s |' % (GN[g], c['n'], f3(c['lam_pre']), f3(c['lam_post']), f3(c['V_pre'], 1), f3(c['V_post'], 1), f3(c['Q_pre'], 4), f3(c['Q_post'], 4), f3(c['dQ'], 4)))
    L.append('\n**Вывод по правилу (больные): предсказание %s.**' % v['R4'])
    L.append('\n## Сводка для машинного чтения\n'); L.append('```json\n' + json.dumps(dict(meta=meta, sets=S, change=out['change'], verdict=v), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        col = {'C': 'k', 'E': '#2471a3', 'ALL': '#7d3c98'}; fig, ax = plt.subplots(1, 4, figsize=(17, 4))
        for g in ('C', 'E'):
            m = np.array([S[g]['lam'][str(t)] for t in DELAYS]); ax[0].errorbar(DELAYS, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o', color=col[g], capsize=3, label=GN[g])
            q = np.array([S[g]['Q'][str(t)] for t in DELAYS]); ax[1].errorbar(np.array(DELAYS) + (0.05 if g == 'E' else -0.05), q[:, 0], yerr=[q[:, 0] - q[:, 1], q[:, 2] - q[:, 0]], fmt='-o', color=col[g], capsize=3, label=GN[g])
        a = S['ALL']
        for key, x, mk, c, lab in (('pred_dyn', 3.15, 's', '#d68910', 'предсказание: динамическое'), ('pred_inf', 3.3, '^', '#1e8449', 'предсказание: «через вывод»')):
            ax[0].errorbar([x], [a[key][0]], yerr=[[a[key][0] - a[key][1]], [a[key][2] - a[key][0]]], fmt=mk, color=c, capsize=3, label=lab)
        ax[0].set_title('Сила стягивания, сессии post', fontsize=10); ax[0].set_xlabel('пауза, с'); ax[0].set_ylabel('lam'); ax[0].legend(frameon=False, fontsize=7)
        ax[1].set_title('Отношение Q = lam·(1 − lam)/V', fontsize=10); ax[1].set_xlabel('пауза, с'); ax[1].set_ylim(0, None); ax[1].legend(frameon=False, fontsize=8)
        for t in ('0.0', '1.0', '3.0'): ax[2].plot([7.5, 22.5, 37.5], S['ALL']['vloc'][t], '-o', label='пауза %s с' % t[0])
        ax[2].set_title('Случайный разброс: от диагонали к оси', fontsize=10); ax[2].set_xlabel('расстояние точки до диагонали, град'); ax[2].set_ylabel('дисперсия остатка, град²'); ax[2].legend(frameon=False, fontsize=8)
        for j, g in enumerate(('E', 'C')):
            c = out['change'][g]; y = [c['Q_pre'], c['Q_post']]; ax[3].errorbar([j - 0.12, j + 0.12], [y[0][0], y[1][0]], yerr=[[y[0][0] - y[0][1], y[1][0] - y[1][1]], [y[0][2] - y[0][0], y[1][2] - y[1][0]]], fmt='-o', color=col[g], capsize=3)
        ax[3].set_xticks([0, 1]); ax[3].set_xticklabels(['энцефалит\npre → post', 'здоровые\npre → post']); ax[3].set_ylim(0, None); ax[3].set_title('Q при паузе 1 с: до и после', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=130)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    return text


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior_retest.pkl')); ap.add_argument('--out', default='wm_stein3_out')
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--check-data', action='store_true'); a = ap.parse_args()
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА при том же числе людей (8 и 14), по три набора на мир.')
        print('Мир «через вывод»: заложены постоянное Q, рост разброса у осей и рост Q у больных от pre к post -> ожидается «повторилось» по Р1-Р3 и «подтверждено» по Р4.')
        print('Мир «динамический»: ничего из этого не заложено -> по всем четырём правилам ожидается обратное.')
        for world in ('inf', 'dyn'):
            for seed in (1, 2, 3):
                stage('мир «%s», набор %d' % ('через вывод' if world == 'inf' else 'динамический', seed)); out = analyse(synth(world, seed)); v = out['verdict']
                print('   Р1: %s | Р2: %s | Р3: %s | Р4: %s' % (v['R1'], v['R2'], v['R3'], v['R4']), flush=True)
        return
    stage('читаю %s' % a.data); d, meta = load(a.data)
    stage('проб после отбора %d (post %d, pre %d); людей %d: %s' % (meta['n_final'], meta['n_post'], meta['n_pre'], len(np.unique(d['subject'])), {str(g): int(len(np.unique(d['subject'][d['group'] == g]))) for g in np.unique(d['group'])}))
    if a.check_data: return
    out = analyse(d); text = report(out, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
