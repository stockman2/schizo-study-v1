# -*- coding: utf-8 -*-
"""
wm_stein6.py — проверка находок разведки (wm_stein4.py, wm_stein5.py) на отложенных данных: сессиях 'post'.

ДАННЫЕ. data/behavior_retest.pkl: 22 человека (8 здоровых, 14 с энцефалитом). Сессии 'post' в разведке не участвовали.
Больных шизофренией в файле нет, поэтому замеченные у них особенности здесь проверить нельзя.

ОБОЗНАЧЕНИЯ
  s2(t) — у человека при паузе t: размах двухпериодной части в сторону оси 0–180 град (плюс — стягивание к этой оси,
          минус — к оси ±90 град), градусы.
  p1(t) — размах однопериодного сдвига в сторону направления −85 град (оно взято из разведки и здесь не подбирается).
  V(t)  — случайная дисперсия.

ПРАВИЛА (записаны до расчёта; всё считается по сессиям post, если не сказано иное)
  П1. Поздний общий дрейф к оси 0–180. Поздний прирост за секунду L = [s2(3) − s2(1)] / 2, ранний F = s2(1) − s2(0),
      усреднение по людям.
        подтверждено полностью: интервал L целиком выше нуля И интервал разности F − L целиком ниже нуля
                                 (то есть в первую секунду прирост меньше, чем потом за секунду);
        подтверждено частично:  интервал L выше нуля, но разность F − L не отличима от нуля;
        не подтверждено:        интервал L содержит ноль.
  П2. Привязка личных осей к осям экрана. При паузе 3 с считается число людей, у которых личная ось двухпериодной
      части лежит ближе 22,5 град к 0 или к 90 град. Подтверждено, если таких не меньше 16 из 22
      (при случайных осях вероятность этого 0,026).
  П3. Личные поля — устойчивая черта человека. Корреляция между сессиями pre и post личных векторов (после вычета
      среднего по людям), пауза 1 с; отдельно для одного и для двух периодов.
      Подтверждено, если нижняя граница 95 % интервала выше нуля.
  П4а. Направление общего однопериодного сдвига при паузе 3 с. Подтверждено, если интервал направления целиком лежит
       между −135 и −45 град.
  П4б. Закон роста однопериодного сдвига. Ранний прирост F1 = p1(1) − p1(0), поздний L1 = p1(3) − p1(1) (за две секунды).
       Два предсказания для L1: «постоянная скорость» — 2 * F1; «вслед за неопределённостью» — F1 * [V(3) − V(1)] / [V(1) − V(0)].
       Совместимо то, для которого интервал разности (наблюдаемое минус предсказанное) содержит ноль.
       Выигрывает то, что совместимо в одиночку; иначе «не решено».

ЗАПУСК (нужны wm_stein2.py … wm_stein5.py в той же папке)
  python wm_stein6.py --selftest     # два искусственных мира при том же числе людей: что метод способен различить
  python wm_stein6.py                # настоящие данные (data/behavior_retest.pkl)
  Результат: папка wm_stein6_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform
from math import comb
import numpy as np
import wm_stein2 as W
import wm_stein3 as W3
import wm_stein4 as W4
import wm_stein5 as W5

DELAYS = W.DELAYS; NBOOT = 2000; PHI0 = np.radians(-85.0); GN = {'C': 'здоровые', 'E': 'энцефалит', 'ALL': 'все вместе'}; T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)
s2_of = lambda c: -c[2]                                                      # c = [a1, b1, a2, b2, …]; A*sin(2*(0 − th)) = −A*sin(2*th)
p1_of = lambda c: -c[0] * np.cos(PHI0) + c[1] * np.sin(PHI0)                 # A*cos(phi − phi0)


def synth(world, seed):
    """'real': заложено всё, что замечено в разведке. 'null': позднего дрейфа нет, личные оси случайны и не сохраняются
    между сессиями, однопериодный сдвиг растёт с постоянной скоростью."""
    rng = np.random.default_rng(seed); S = {k: [] for k in ('subject', 'group', 'session', 'delay', 'target', 'prevcurr', 'err')}
    for g, ns in (('C', 8), ('E', 14)):
        for k in range(ns):
            noise = np.exp(rng.normal(0, 0.3)) * (1.3 if g == 'E' else 1.0); own1 = rng.uniform(-np.pi, np.pi); u = rng.random()
            ax2 = 0.0 if u < 0.6 else (np.pi / 2 if u < 0.85 else rng.uniform(0, np.pi)); amp2 = 2.0 * np.exp(rng.normal(0, 0.5))
            for ses in ('pre', 'post'):
                if world == 'null': own1 = rng.uniform(-np.pi, np.pi); ax2 = rng.uniform(0, np.pi)
                n = 1100; th = rng.uniform(-np.pi, np.pi, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); pc = np.angle(np.exp(1j * (np.roll(th, 1) - th)))
                V = noise * (10.0 + 10.0 * (dl > 0) + 4.0 * dl); lam = V / (V + 300.0); gr = np.array([{0.0: 0.5, 1.0: 0.75, 3.0: 1.0}[t] for t in dl])
                e = -lam * W.dist_to_diag(th) + rng.normal(0, np.sqrt(V)) * (1 - lam) + 1.5 * np.sin(own1 - th) + amp2 * gr * np.sin(2 * (ax2 - th))
                if world == 'real': e = e + (0.9 + 0.035 * V) * np.sin(PHI0 - th) + 0.52 * np.clip(dl - 1.0, 0, None) * np.sin(2 * (0.0 - th))
                else: e = e + (0.9 + 0.7 * dl) * np.sin(PHI0 - th)
                for key, v in zip(S, ([g + '%02d' % k] * n, [g] * n, [ses] * n, dl, th, pc, e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}


def per_subject(d):
    R = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; r = dict(group=d['group'][ms][0])
        for ses in ('pre', 'post'):
            c = {}; V = {}
            for t in DELAYS:
                m = ms & (d['session'] == ses) & (d['delay'] == t); e = d['err'][m]; th = d['target'][m]; e = e - e.mean(); x = W.dog(d['prevcurr'][m]); e = e - np.sum(e * x) / np.sum(x * x) * x
                c[t] = W5.harm_fit(th, e); V[t] = float((e - W5.recon(c[t][None, :], range(1, W5.KMAX + 1), th)[0]).var())
            r[ses] = dict(c=c, V=V)
        R[s] = r
    return R

def ci(M, fn, rng):
    M = np.asarray(M, float); n = len(M); est = fn(M.mean(0)); bs = np.array([fn(M[rng.integers(0, n, n)].mean(0)) for _ in range(NBOOT)])
    return [float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]

def vec_corr(A, B, rng):
    """Корреляция личных векторов между сессиями (обе составляющие вместе, после вычета среднего по людям) с интервалом."""
    def r(i): a = A[i] - A[i].mean(0); b = B[i] - B[i].mean(0); return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    n = len(A); bs = [r(rng.integers(0, n, n)) for _ in range(NBOOT)]; return [r(np.arange(n)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def analyse(d):
    stage('гармоники у каждого человека в сессиях pre и post'); R = per_subject(d); rng = np.random.default_rng(61); subs = list(R); out = dict(n=len(subs), groups={g: int(sum(R[s]['group'] == g for s in subs)) for g in ('C', 'E')})
    stage('проверка правил, интервалы (бутстрэп по людям, %d повторов)' % NBOOT)
    def table(ses, sel):
        c = lambda t: np.array([R[s][ses]['c'][t] for s in sel]); V = lambda t: np.array([R[s][ses]['V'][t] for s in sel])
        s2 = np.column_stack([[s2_of(x) for x in c(t)] for t in DELAYS]); p1 = np.column_stack([[p1_of(x) for x in c(t)] for t in DELAYS]); Vv = np.column_stack([V(t) for t in DELAYS]); o2 = np.column_stack([[x[3] for x in c(t)] for t in DELAYS])
        G = dict(s2={str(t): ci(s2[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)}, p1={str(t): ci(p1[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)},
                 V={str(t): ci(Vv[:, [i]], lambda m: m[0], rng) for i, t in enumerate(DELAYS)}, orth2_late=ci((o2[:, [2]] - o2[:, [1]]) / 2, lambda m: m[0], rng))
        G['L'] = ci((s2[:, [2]] - s2[:, [1]]) / 2, lambda m: m[0], rng); G['F'] = ci(s2[:, [1]] - s2[:, [0]], lambda m: m[0], rng); G['F_minus_L'] = ci(np.column_stack([s2[:, 1] - s2[:, 0], (s2[:, 2] - s2[:, 1]) / 2]), lambda m: m[0] - m[1], rng)
        M = np.column_stack([p1, Vv]); f1 = lambda m: m[1] - m[0]; l1 = lambda m: m[2] - m[1]; pv = lambda m: f1(m) * (m[5] - m[4]) / (m[4] - m[3])
        G['F1'] = ci(M, f1, rng); G['L1'] = ci(M, l1, rng); G['pred_speed'] = ci(M, lambda m: 2 * f1(m), rng); G['pred_unc'] = ci(M, pv, rng)
        G['L1_minus_speed'] = ci(M, lambda m: l1(m) - 2 * f1(m), rng); G['L1_minus_unc'] = ci(M, lambda m: l1(m) - pv(m), rng)
        C3 = c(3.0)[:, 0:2]; A, phi = W4.to_dir(C3.mean(0), 1); n = len(sel); bs = []
        for _ in range(NBOOT):
            i = rng.integers(0, n, n); a_, p_ = W4.to_dir(C3[i].mean(0), 1); bs.append(((p_ - phi + 180) % 360 - 180, a_))
        bs = np.array(bs); G['dir1'] = [phi, phi + float(np.percentile(bs[:, 0], 2.5)), phi + float(np.percentile(bs[:, 0], 97.5))]; G['amp1'] = [A, float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))]
        ax = np.array([W4.to_dir(x[2:4], 2)[1] for x in c(3.0)]); am = np.array([W4.to_dir(x[2:4], 2)[0] for x in c(3.0)]); near0 = np.minimum(ax, 180 - ax) < 22.5; near90 = np.abs(ax - 90) < 22.5
        G['axes'] = dict(n=n, near0=int(near0.sum()), near90=int(near90.sum()), between=int((~near0 & ~near90).sum()), p=float(sum(comb(n, k) for k in range(int((near0 | near90).sum()), n + 1)) / 2 ** n), axis=ax.tolist(), amp=am.tolist())
        return G
    out['post'] = {g: table('post', [s for s in subs if g == 'ALL' or R[s]['group'] == g]) for g in ('ALL', 'C', 'E')}; out['pre'] = {'ALL': table('pre', subs)}
    out['stability'] = {}
    for t in (1.0, 3.0):
        A = np.array([R[s]['pre']['c'][t] for s in subs]); B = np.array([R[s]['post']['c'][t] for s in subs])
        out['stability'][str(t)] = dict(k1=vec_corr(A[:, 0:2], B[:, 0:2], rng), k2=vec_corr(A[:, 2:4], B[:, 2:4], rng), pre1=A[:, 0:2].tolist(), post1=B[:, 0:2].tolist(), pre2=A[:, 2:4].tolist(), post2=B[:, 2:4].tolist())
    G = out['post']['ALL']; inz = lambda c: c[1] <= 0 <= c[2]; sp, un = inz(G['L1_minus_speed']), inz(G['L1_minus_unc']); a = G['axes']; st = out['stability']['1.0']
    out['verdict'] = dict(
        P1='подтверждено полностью' if (G['L'][1] > 0 and G['F_minus_L'][2] < 0) else ('подтверждено частично: поздний дрейф есть, отличие от первой секунды не разрешено' if G['L'][1] > 0 else 'не подтверждено'),
        P2='подтверждено' if (a['near0'] + a['near90']) >= 16 and a['n'] == 22 else ('подтверждено' if a['p'] < 0.05 else 'не подтверждено'),
        P3_k1='подтверждено' if st['k1'][1] > 0 else 'не подтверждено', P3_k2='подтверждено' if st['k2'][1] > 0 else 'не подтверждено',
        P4a='подтверждено' if (-135 <= G['dir1'][1] and G['dir1'][2] <= -45) else 'не подтверждено',
        P4b='вслед за неопределённостью' if (un and not sp) else ('постоянная скорость' if (sp and not un) else 'не решено (%s)' % ('оба совместимы' if sp else 'ни одно не совместимо')))
    return out


def f3(x, nd=2): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def report(out, meta, outdir, tag=''):
    P = out['post']; v = out['verdict']; L = ['# Проверка находок о гармониках на отложенных сессиях post%s\n' % tag]
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    if meta: L.append('- проб после отбора: в сессиях post %d, в сессиях pre %d' % (meta['n_post'], meta['n_pre']))
    L.append('- людей %d: здоровых %d, с энцефалитом %d; в скобках 95 %% интервал (бутстрэп по людям)\n' % (out['n'], out['groups']['C'], out['groups']['E']))
    L.append('## П1. Поздний общий дрейф к оси 0–180 (два периода)\n'); L.append('| набор | s2: 0 с | s2: 1 с | s2: 3 с | ранний прирост F (за 1-ю секунду) | поздний прирост L (за секунду) | F − L | поздний прирост поперёк оси |\n|---|---|---|---|---|---|---|---|')
    for name, G in (('post, все', P['ALL']), ('post, здоровые', P['C']), ('post, энцефалит', P['E']), ('pre, все (для сравнения, уже видено)', out['pre']['ALL'])):
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (name, f3(G['s2']['0.0']), f3(G['s2']['1.0']), f3(G['s2']['3.0']), f3(G['F']), f3(G['L']), f3(G['F_minus_L']), f3(G['orth2_late'])))
    L.append('\n**Вывод по правилу: %s.**' % v['P1'])
    L.append('\n## П2. Привязка личных осей к осям экрана (пауза 3 с)\n'); L.append('| набор | людей | ось у 0–180 | ось у ±90 | между осями | вероятность при случайных осях |\n|---|---|---|---|---|---|')
    for name, G in (('post, все', P['ALL']), ('post, здоровые', P['C']), ('post, энцефалит', P['E']), ('pre, все (уже видено)', out['pre']['ALL'])):
        a = G['axes']; L.append('| %s | %d | %d | %d | %d | %.4f |' % (name, a['n'], a['near0'], a['near90'], a['between'], a['p']))
    L.append('\n**Вывод по правилу: %s.**' % v['P2'])
    L.append('\n## П3. Личные поля в двух сессиях: корреляция личных векторов pre и post\n'); L.append('| пауза | один период | два периода |\n|---|---|---|')
    for t in ('1.0', '3.0'): L.append('| %s с%s | %s | %s |' % (t[0], ' (основная)' if t == '1.0' else ' (справочно)', f3(out['stability'][t]['k1']), f3(out['stability'][t]['k2'])))
    L.append('\n**Вывод по правилу: один период — %s; два периода — %s.**' % (v['P3_k1'], v['P3_k2']))
    L.append('\n## П4. Общий однопериодный сдвиг\n'); L.append('| набор | направление при 3 с, град | размах при 3 с | p1: 0 с | p1: 1 с | p1: 3 с |\n|---|---|---|---|---|---|')
    for name, G in (('post, все', P['ALL']), ('post, здоровые', P['C']), ('post, энцефалит', P['E']), ('pre, все (уже видено)', out['pre']['ALL'])):
        L.append('| %s | %s | %s | %s | %s | %s |' % (name, f3(G['dir1'], 0), f3(G['amp1']), f3(G['p1']['0.0']), f3(G['p1']['1.0']), f3(G['p1']['3.0'])))
    G = P['ALL']; L.append('\n**Вывод по правилу П4а (направление): %s.**\n' % v['P4a'])
    L.append('| величина (post, все) | значение, град |\n|---|---|'); L.append('| ранний прирост F1 (0→1 с) | %s |' % f3(G['F1'])); L.append('| поздний прирост L1 (1→3 с, за две секунды) | %s |' % f3(G['L1']))
    L.append('| предсказание «постоянная скорость» | %s; наблюдаемое минус предсказанное %s |' % (f3(G['pred_speed']), f3(G['L1_minus_speed']))); L.append('| предсказание «вслед за неопределённостью» | %s; наблюдаемое минус предсказанное %s |' % (f3(G['pred_unc']), f3(G['L1_minus_unc'])))
    L.append('\n**Вывод по правилу П4б (закон роста): %s.**' % v['P4b'])
    L.append('\n## Сводка для машинного чтения\n')
    slim = dict(post={g: {k: x for k, x in P[g].items()} for g in P}, pre=out['pre'], stability={t: {k: out['stability'][t][k] for k in ('k1', 'k2')} for t in out['stability']}, verdict=v, n=out['n'], groups=out['groups'])
    L.append('```json\n' + json.dumps(dict(meta=meta, **slim), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(17, 4)); d = [0, 1, 3]
        for name, G, c in (('post', P['ALL'], 'k'), ('pre (уже видено)', out['pre']['ALL'], '0.6')):
            m = np.array([G['s2'][str(float(t))] for t in d]); ax[0].errorbar(np.array(d) + (0.04 if c == 'k' else -0.04), m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o', color=c, capsize=3, label=name)
            m = np.array([G['p1'][str(float(t))] for t in d]); ax[3].errorbar(np.array(d) + (0.04 if c == 'k' else -0.04), m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o', color=c, capsize=3, label=name)
        ax[0].axhline(0, color='gray', lw=0.6); ax[0].set_title('Два периода: общее стягивание к оси 0–180', fontsize=10); ax[0].set_xlabel('пауза, с'); ax[0].set_ylabel('град'); ax[0].legend(frameon=False, fontsize=8)
        a = P['ALL']['axes']; ax[1].hist(a['axis'], bins=np.arange(0, 181, 15), color='0.5'); ax[1].axvspan(0, 22.5, color='g', alpha=0.12); ax[1].axvspan(157.5, 180, color='g', alpha=0.12); ax[1].axvspan(67.5, 112.5, color='b', alpha=0.10)
        ax[1].set_title('Личные оси двух периодов, post, 3 с\n(зелёное — у оси 0–180, синее — у ±90)', fontsize=9); ax[1].set_xlabel('ось, град'); ax[1].set_ylabel('людей')
        s = out['stability']['1.0']
        for key, c, lab in (('1', '#c0392b', 'один период'), ('2', '#2471a3', 'два периода')):
            A = np.array(s['pre' + key]); B = np.array(s['post' + key]); ax[2].plot((A - A.mean(0)).ravel(), (B - B.mean(0)).ravel(), 'o', ms=4, color=c, alpha=0.7, label='%s: r = %.2f' % (lab, s['k' + key][0]))
        ax[2].axhline(0, color='gray', lw=0.5); ax[2].axvline(0, color='gray', lw=0.5); ax[2].set_title('Личные составляющие: pre против post, 1 с', fontsize=10); ax[2].set_xlabel('pre, град'); ax[2].set_ylabel('post, град'); ax[2].legend(frameon=False, fontsize=8)
        G = P['ALL']; ax[3].errorbar([3.2], [G['p1']['1.0'][0] + G['pred_speed'][0]], yerr=[[G['pred_speed'][0] - G['pred_speed'][1]], [G['pred_speed'][2] - G['pred_speed'][0]]], fmt='s', color='#d68910', capsize=3, label='предсказание: постоянная скорость')
        ax[3].errorbar([3.35], [G['p1']['1.0'][0] + G['pred_unc'][0]], yerr=[[G['pred_unc'][0] - G['pred_unc'][1]], [G['pred_unc'][2] - G['pred_unc'][0]]], fmt='^', color='#1e8449', capsize=3, label='предсказание: вслед за неопределённостью')
        ax[3].set_title('Один период: сдвиг в сторону −85°', fontsize=10); ax[3].set_xlabel('пауза, с'); ax[3].set_ylabel('град'); ax[3].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=130)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    return text


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior_retest.pkl')); ap.add_argument('--out', default='wm_stein6_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА при том же числе людей (8 и 14), по три набора на мир.')
        print('Мир «как в разведке»: заложены поздний дрейф к оси 0–180, привязка личных осей, устойчивые личные поля, сдвиг к −85° вслед за неопределённостью.')
        print('Мир «пустой»: позднего дрейфа нет, личные оси случайны и меняются между сессиями, сдвиг к −85° растёт с постоянной скоростью.')
        for world in ('real', 'null'):
            for seed in (1, 2, 3):
                stage('мир «%s», набор %d' % ('как в разведке' if world == 'real' else 'пустой', seed)); o = analyse(synth(world, seed)); v = o['verdict']; ax_ = o['post']['ALL']['axes']
                print('   П1: %s | П2: %s (%d из 22) | П3: один период — %s, два — %s | П4а: %s | П4б: %s' % (v['P1'], v['P2'], ax_['near0'] + ax_['near90'], v['P3_k1'], v['P3_k2'], v['P4a'], v['P4b']), flush=True)
        return
    stage('читаю %s' % a.data); d, meta = W3.load(a.data); stage('проб после отбора: post %d, pre %d; людей %d' % (meta['n_post'], meta['n_pre'], len(np.unique(d['subject']))))
    out = analyse(d); text = report(out, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
