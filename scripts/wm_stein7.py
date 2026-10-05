# -*- coding: utf-8 -*-
"""
wm_stein7.py — проверка дифференциальной модели по тому, чего в её настройке не было: по профилю случайного разброса.

ДВЕ МОДЕЛИ (обе воспроизводят закон «стягивание пропорционально неопределённости»)
  Д, дифференциальная. След блуждает по окружности в потенциале с четырьмя ямами (дно на диагоналях, излом на осях):
        dx = −(D(t)/Vs) · delta(x) · dt + sqrt(2·D(t)) · dW,     delta(x) — расстояние до ближайшей диагонали.
     Снос пропорционален силе шума D(t). Ответ — положение следа в конце паузы.
  О, подправка при ответе. След только блуждает (без сноса), а при ответе сдвигается к диагонали того квадранта,
     в котором он оказался: ответ = m − lam · delta(m), lam = Vпамяти/(Vпамяти + Vp).
  Для сравнения: если бы разброс от места не зависел, отношение «у оси / у диагонали» равнялось бы 1.

НАСТРОЙКА (без подгонки к проверяемой величине)
  У каждого человека по его же данным берутся сила стягивания lam(t) и общая случайная дисперсия V(t) при трёх паузах.
  Из них: один параметр ямы на человека (Vs для Д, Vp для О — такой, чтобы суммарное по трём паузам стягивание в модели
  равнялось измеренному) и количество шума при каждой паузе (такое, чтобы модель дала ту же ОБЩУЮ дисперсию V(t)).
  Настройка идёт прогонами самой модели через тот же расчёт, что применяется к данным. Как дисперсия распределена
  по окружности, в настройку не входит — это и есть предсказание.
  Модельные пробы создаются в тех же местах окружности, что и настоящие, и обрабатываются тем же расчётом.

ПРОВЕРЯЕМАЯ ВЕЛИЧИНА
  R(t) = (средняя дисперсия остатка у осей, 30–45° от диагонали) / (у диагоналей, 0–15°), паузы 1 и 3 с.

ПРАВИЛО (записано до расчёта; основная проверка — исходные сессии, 52 человека)
  Модель «совместима», если при ОБЕИХ паузах, 1 и 3 с, 99 % интервал разности R(данные) − R(модель) содержит ноль
  (бутстрэп по людям, данные и модель пересчитываются на одной и той же выборке людей; 99 %, а не 95 %, потому что
  сравнений сразу четыре — иначе верная модель отвергалась бы слишком часто).
    совместима только Д -> в пользу дифференциальной модели;
    совместима только О -> в пользу подправки при ответе;
    обе -> профиль разброса эти модели не различает;
    ни одна -> обе модели профиль не описывают.
  Справочно (без правила): середина (15–30°), пауза 0 с, сессии post, островершинность распределения остатков,
  и проверка настройки — какое lam(t) получилось в моделях.

ЗАПУСК (нужны wm_stein2.py и wm_stein3.py в той же папке)
  python wm_stein7.py --selftest     # искусственные миры Д и О: способен ли метод их различить (около 5 минут)
  python wm_stein7.py                # настоящие данные (около 2–3 минут)
  Результат: папка wm_stein7_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform
import numpy as np
import wm_stein2 as W
import wm_stein3 as W3

DELAYS = W.DELAYS; NREP = 20; DTAU = 0.05; NBOOT = 2000; NREFINE = 5; BANDS = ((0, 15), (15, 30), (30, 45.01)); T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)
def ddiag(th_deg): return np.mod(th_deg, 90.0) - 45.0
def wrap(x): return (x + 180.0) % 360.0 - 180.0


def analyse_block(theta_rad, e):
    """Тот же расчёт, что в прежних скриптах: стягивание + низкие гармоники, затем остаток по трём полосам расстояния до диагонали."""
    lam, res = W.fit_lam(theta_rad, e - e.mean()); ad = np.abs(W.dist_to_diag(theta_rad)); v = []; k = []
    for a, b in BANDS:
        r = res[(ad >= a) & (ad < b)]; r = r - r.mean(); s2 = r.var(); v.append(float(s2)); k.append(float(np.mean(r ** 4) / s2 ** 2 - 3.0) if s2 > 0 else np.nan)
    return lam, float(res.var()), v, k

def measure(d, serial=True):
    """По данным: у каждого человека при каждой паузе lam, V, дисперсия по полосам, островершинность; места точек запоминаются для моделей."""
    M = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; r = dict(group=d['group'][ms][0])
        for t in DELAYS:
            m = ms & (d['delay'] == t); e = d['err'][m]; th = d['target'][m]; e = e - e.mean()
            if serial: x = W.dog(d['prevcurr'][m]); e = e - np.sum(e * x) / np.sum(x * x) * x
            lam, V, v, k = analyse_block(th, e); r[t] = dict(lam=lam, V=V, vloc=v, kurt=k, th=th, n=len(th))
        M[s] = r
    return M


# ---------- модели ----------
def sim_D(th0, tau, vs, rng):
    """Блуждание в потенциале с четырьмя ямами; время измеряется накопленным шумом tau (град²). Все частицы сразу."""
    nst = np.round(tau / DTAU).astype(int); order = np.argsort(-nst); x = th0[order].copy(); inv = 1.0 / vs[order]; ns = nst[order]; sd = np.sqrt(2 * DTAU)
    cnt = np.searchsorted(-ns, -np.arange(1, ns.max() + 1), side='right') if ns.max() > 0 else np.array([], int)      # сколько частиц ещё идут на шаге k
    for k in range(len(cnt)):
        m = cnt[k]
        if m == 0: break
        x[:m] += -ddiag(x[:m]) * inv[:m] * DTAU + sd * rng.standard_normal(m)
    out = np.empty_like(x); out[order] = x; return out

def sim_R(th0, vmem, lam, rng):
    m = th0 + np.sqrt(vmem) * rng.standard_normal(len(th0)); return m - lam * ddiag(m)

def predict(M, model, rng, label):
    """Настройка по lam и общей V каждого человека, затем модельные пробы и тот же расчёт. Возвращает то же, что measure()."""
    subs = list(M); par = {}; noise = {}
    for s in subs:
        n = np.array([M[s][t]['n'] for t in DELAYS], float); V = np.array([M[s][t]['V'] for t in DELAYS]); lam = np.array([M[s][t]['lam'] for t in DELAYS])
        den = np.sum(n * lam * ((2 - lam) if model == 'D' else (1 - lam))); num = np.sum(n * V); par[s] = num / den if den > 0.02 * n.sum() else np.inf
        for t, v in zip(DELAYS, V):
            if model == 'D': noise[s, t] = v / 2 if not np.isfinite(par[s]) else -(par[s] / 2) * np.log(1 - min(v / par[s], 0.95))
            else:
                q = 0.0 if not np.isfinite(par[s]) else min(v / par[s], 0.24); l0 = (1 - np.sqrt(1 - 4 * q)) / 2; noise[s, t] = v / (1 - l0) ** 2      # шум памяти до подправки
    P = None
    for it in range(NREFINE + 1):
        th0 = []; a1 = []; a2 = []; idx = []
        for s in subs:
            for t in DELAYS:
                th = np.degrees(np.tile(M[s][t]['th'], NREP)); th0.append(th); idx.append((s, t, len(th)))
                if model == 'D': a1.append(np.full(len(th), noise[s, t])); a2.append(np.full(len(th), par[s]))
                else:
                    lam = noise[s, t] / (noise[s, t] + par[s]) if np.isfinite(par[s]) else 0.0
                    a1.append(np.full(len(th), noise[s, t])); a2.append(np.full(len(th), lam))
        th0 = np.concatenate(th0); a1 = np.concatenate(a1); a2 = np.concatenate(a2)
        resp = sim_D(th0, a1, a2, rng) if model == 'D' else sim_R(th0, a1, a2, rng); err = wrap(resp - th0); P = {s: dict(group=M[s]['group']) for s in subs}; p = 0; worst = 0.0
        for s, t, n in idx:
            lam, V, v, k = analyse_block(np.radians(th0[p:p + n]), err[p:p + n]); P[s][t] = dict(lam=lam, V=V, vloc=v, kurt=k); p += n
            f = M[s][t]['V'] / V if V > 0 else 1.0; worst = max(worst, abs(f - 1))
            if it < NREFINE: noise[s, t] *= float(np.clip(f, 0.5, 2.0))
        # подстройка параметра ямы: суммарное по паузам стягивание в модели должно равняться измеренному у этого человека
        gd = gs = 0.0
        for s in subs:
            ad = sum(M[s][t]['n'] * M[s][t]['lam'] for t in DELAYS); am = sum(M[s][t]['n'] * P[s][t]['lam'] for t in DELAYS); gd += ad; gs += am
            if it < NREFINE and np.isfinite(par[s]) and ad > 0 and am > 0: par[s] *= float(np.clip(am / ad, 0.5, 2.0))
        stage('   модель %s (%s): проход %d из %d; расхождение с человеком: общая дисперсия до %.1f %%, стягивание в среднем %.1f %%' % (model, label, it + 1, NREFINE + 1, 100 * worst, 100 * (gs / gd - 1)))
    return P, {s: float(par[s]) if np.isfinite(par[s]) else None for s in subs}


# ---------- сравнение ----------
def compare(M, PD, PR, rng):
    subs = list(M); n = len(subs); out = {}
    def arr(X, key): return np.array([[X[s][t][key] for t in DELAYS] for s in subs])
    VL = {k: arr(X, 'vloc') for k, X in (('data', M), ('D', PD), ('R', PR))}; KU = {k: arr(X, 'kurt') for k, X in (('data', M), ('D', PD), ('R', PR))}
    LA = {k: arr(X, 'lam') for k, X in (('data', M), ('D', PD), ('R', PR))}; VV = {k: arr(X, 'V') for k, X in (('data', M), ('D', PD), ('R', PR))}
    idx = rng.integers(0, n, (NBOOT, n)); ratio = lambda A, i, ti, b=2: A[i][:, ti, b].mean() / A[i][:, ti, 0].mean(); full = np.arange(n)
    for ti, t in enumerate(DELAYS):
        o = {}
        for k in ('data', 'D', 'R'):
            bs = np.array([ratio(VL[k], i, ti) for i in idx]); o['R_' + k] = [float(ratio(VL[k], full, ti)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            bm = np.array([ratio(VL[k], i, ti, 1) for i in idx]); o['Rmid_' + k] = [float(ratio(VL[k], full, ti, 1)), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
            o['bands_' + k] = VL[k][:, ti].mean(0).tolist(); o['kurt_' + k] = np.nanmean(KU[k][:, ti], 0).tolist(); o['lam_' + k] = float(LA[k][:, ti].mean()); o['V_' + k] = float(VV[k][:, ti].mean())
        for k in ('D', 'R'):
            bs = np.array([ratio(VL['data'], i, ti) - ratio(VL[k], i, ti) for i in idx]); o['diff_' + k] = [float(ratio(VL['data'], full, ti) - ratio(VL[k], full, ti)), float(np.percentile(bs, 0.5)), float(np.percentile(bs, 99.5))]
        out[str(t)] = o
    ok = {k: all(out[t]['diff_' + k][1] <= 0 <= out[t]['diff_' + k][2] for t in ('1.0', '3.0')) for k in ('D', 'R')}
    out['compatible'] = ok
    out['verdict'] = ('в пользу дифференциальной модели' if ok['D'] and not ok['R'] else 'в пользу подправки при ответе' if ok['R'] and not ok['D'] else
                      'профиль разброса эти модели не различает (обе совместимы)' if ok['D'] else 'обе модели профиль не описывают')
    return out

def run(d, label, seed, serial=True):
    stage('%s: измерения по данным (%d человек)' % (label, len(np.unique(d['subject'])))); M = measure(d, serial); rng = np.random.default_rng(seed)
    stage('%s: модель Д — блуждание в потенциале, %d модельных проб на каждую настоящую, шаг %.2f град²' % (label, NREP, DTAU)); PD, parD = predict(M, 'D', rng, label)
    stage('%s: модель О — подправка при ответе' % label); PR, parR = predict(M, 'R', rng, label)
    stage('%s: сравнение и интервалы (бутстрэп по людям, %d повторов)' % (label, NBOOT)); out = compare(M, PD, PR, rng)
    out['Vs_median'] = float(np.median([v for v in parD.values() if v])); out['Vp_median'] = float(np.median([v for v in parR.values() if v])); out['n'] = len(M); return out


def f3(x, nd=2): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def section(L, out, title, with_rule):
    L.append('## %s (людей: %d)\n' % (title, out['n'])); L.append('Медиана параметра ямы: Vs = %.0f град² (модель Д), Vp = %.0f град² (модель О).\n' % (out['Vs_median'], out['Vp_median']))
    L.append('| пауза | R: данные | R: модель Д | данные − Д | R: модель О | данные − О |\n|---|---|---|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): o = out[t]; L.append('| %s с%s | %s | %s | %s | %s | %s |' % (t[0], ' (справочно)' if t == '0.0' else '', f3(o['R_data']), f3(o['R_D']), f3(o['diff_D']), f3(o['R_R']), f3(o['diff_R'])))
    if with_rule: L.append('\n**Вывод по правилу: %s.** Совместимость: Д — %s, О — %s.' % (out['verdict'], 'да' if out['compatible']['D'] else 'нет', 'да' if out['compatible']['R'] else 'нет'))
    else: L.append('\nСправочно, без правила: по тому же критерию получилось бы «%s».' % out['verdict'])
    L.append('\nДисперсия остатка по полосам, град² (у диагонали / середина / у оси):\n'); L.append('| пауза | данные | модель Д | модель О |\n|---|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): o = out[t]; L.append('| %s с | %s | %s | %s |' % (t[0], ' / '.join('%.1f' % x for x in o['bands_data']), ' / '.join('%.1f' % x for x in o['bands_D']), ' / '.join('%.1f' % x for x in o['bands_R'])))
    L.append('\nОтношение «середина / у диагонали»:\n'); L.append('| пауза | данные | модель Д | модель О |\n|---|---|---|---|')
    for t in ('1.0', '3.0'): o = out[t]; L.append('| %s с | %s | %s | %s |' % (t[0], f3(o['Rmid_data']), f3(o['Rmid_D']), f3(o['Rmid_R'])))
    L.append('\nПроверка настройки — сила стягивания lam и общая дисперсия V (данные / Д / О):\n'); L.append('| пауза | lam | V, град² |\n|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): o = out[t]; L.append('| %s с | %.3f / %.3f / %.3f | %.1f / %.1f / %.1f |' % (t[0], o['lam_data'], o['lam_D'], o['lam_R'], o['V_data'], o['V_D'], o['V_R']))
    L.append('\nОстровершинность остатков при 3 с (0 — нормальное распределение; у диагонали / середина / у оси): данные %s; Д %s; О %s.\n' % tuple(' / '.join('%.2f' % x for x in out['3.0']['kurt_' + k]) for k in ('data', 'D', 'R')))

def report(res, meta, outdir):
    L = ['# Дифференциальная модель против подправки при ответе: профиль случайного разброса\n']
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    L.append('- R — отношение дисперсии остатка у осей (30–45° от диагонали) к дисперсии у диагоналей (0–15°); в скобках 95 % интервал, а для разностей «данные − модель» — 99 % (бутстрэп по людям)')
    L.append('- модели настроены по силе стягивания и ОБЩЕЙ дисперсии каждого человека; распределение дисперсии по окружности в настройку не входило\n')
    section(L, res['base'], 'Основная проверка: исходные сессии', True)
    if 'post' in res: section(L, res['post'], 'Справочно: сессии post', False)
    L.append('## Сводка для машинного чтения\n'); L.append('```json\n' + json.dumps(dict(meta=meta, **res), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(res, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        keys = [k for k in ('base', 'post') if k in res]; fig, ax = plt.subplots(len(keys), 4, figsize=(17, 4 * len(keys)), squeeze=False); col = {'data': 'k', 'D': '#c0392b', 'R': '#2471a3'}; nm = {'data': 'данные', 'D': 'модель Д', 'R': 'модель О'}
        for r, key in enumerate(keys):
            o = res[key]
            for j, t in enumerate(('0.0', '1.0', '3.0')):
                for k in ('data', 'D', 'R'): ax[r][j].plot([7.5, 22.5, 37.5], o[t]['bands_' + k], '-o' if k == 'data' else '--s', color=col[k], label=nm[k])
                ax[r][j].set_title('%s, пауза %s с' % ('исходные сессии' if key == 'base' else 'сессии post', t[0]), fontsize=10); ax[r][j].set_xlabel('расстояние до диагонали, град'); ax[r][j].set_ylabel('дисперсия остатка, град²')
            ax[r][0].legend(frameon=False, fontsize=8)
            for i, k in enumerate(('data', 'D', 'R')):
                m = np.array([o[t]['R_' + k] for t in ('0.0', '1.0', '3.0')]); ax[r][3].errorbar(np.array([0, 1, 3]) + (i - 1) * 0.07, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='-o' if k == 'data' else '--s', color=col[k], capsize=3, label=nm[k])
            ax[r][3].axhline(1, color='gray', lw=0.6); ax[r][3].set_title('Отношение «у оси / у диагонали»', fontsize=10); ax[r][3].set_xlabel('пауза, с'); ax[r][3].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=130)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    return text


def synth_world(model, seed):
    """Искусственные люди, порождённые одной из двух моделей с известными параметрами (плюс личный постоянный сдвиг)."""
    rng = np.random.default_rng(seed); S = {k: [] for k in ('subject', 'group', 'delay', 'target', 'prevcurr', 'err')}
    for g, ns in (('C', 19), ('S', 17), ('E', 16)):
        for k in range(ns):
            n = 1050; th = rng.uniform(-180, 180, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); mult = np.exp(rng.normal(0, 0.3)); vs = 155.0 * np.exp(rng.normal(0, 0.3))
            tau = mult * np.array([{0.0: 6.0, 1.0: 16.0, 3.0: 21.0}[t] for t in dl])
            if model == 'D': resp = sim_D(th, tau, np.full(n, vs), rng)
            else: vm = 2 * tau; lam = vm / (vm + 2 * vs); resp = sim_R(th, vm, lam, rng)
            e = wrap(resp - th) + 1.2 * np.sin(rng.uniform(-np.pi, np.pi) - np.radians(th))
            for key, v in zip(S, ([g + '%02d' % k] * n, [g] * n, dl, np.radians(th), np.zeros(n), e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}

def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--retest', default=os.path.join('data', 'behavior_retest.pkl'))
    ap.add_argument('--out', default='wm_stein7_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        print('ПРОВЕРКА МЕТОДА: искусственные люди порождены моделью Д или моделью О; метод должен назвать породившую модель')
        print('(или честно сказать, что профиль разброса их не различает).')
        for model, name in (('D', 'Д'), ('R', 'О')):
            for seed in (1, 2):
                o = run(synth_world(model, seed), 'мир %s, набор %d' % (name, seed), 100 + seed, serial=False)
                print('   ИСТИНА: модель %s -> вердикт: %s' % (name, o['verdict']))
                for t in ('1.0', '3.0'): print('      пауза %s с: R «данные» %s | Д %s | О %s' % (t[0], f3(o[t]['R_data']), f3(o[t]['R_D']), f3(o[t]['R_R'])), flush=True)
        return
    res = {}; stage('читаю %s' % a.data); d, meta = W.load_real(a.data); res['base'] = run(d, 'исходные сессии', 71)
    if os.path.exists(a.retest):
        stage('читаю %s' % a.retest); d2, meta2 = W3.load(a.retest); m = d2['session'] == 'post'; res['post'] = run({k: v[m] for k, v in d2.items() if k != 'session'}, 'сессии post', 72); meta = dict(meta, n_post=meta2['n_post'])
    else: stage('файла %s нет — сессии post пропущены' % a.retest)
    text = report(res, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text[:text.find('## Сводка для машинного чтения')] + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
