# -*- coding: utf-8 -*-
"""
wm_stein4.py — РАЗВЕДКА: что такое гармоники карты сноса с одним и двумя периодами на круг (данные Stein и соавт., 2020).

ЭТО НЕ ПРОВЕРКА ГИПОТЕЗЫ, А ОПИСАНИЕ. Считается только по исходным сессиям (data/behavior.pkl, 52 человека).
Повторные сессии 'post' из behavior_retest.pkl нарочно не трогаются: на них потом можно будет честно проверить то,
что здесь будет замечено.

ГЕОМЕТРИЧЕСКИЙ СМЫСЛ
  Один период:  ошибка = A1 * sin(phi1 - место). Так выглядит сдвиг всех запомненных положений в одну сторону экрана:
                phi1 — направление, КУДА они смещены, A1 — размах в градусах дуги.
  Два периода:  ошибка = A2 * sin(2 * (phi2 - место)). Так выглядит стягивание к одной оси (и уход от перпендикулярной):
                phi2 — направление оси притяжения (от 0 до 180 град), A2 — размах.
  Важно: в репозитории нет кода самого опыта, поэтому неизвестно, где на экране угол +90 град — вверху или внизу.
  Все направления ниже даны в координатах файла данных.

ЧТО ПЕЧАТАЕТСЯ (для каждой гармоники и каждой паузы)
  1. Общая для всех людей составляющая: направление и размах среднего вектора, с интервалами.
  2. Какая доля мощности гармоники общая для людей, а какая у каждого своя (оценка по двум половинам проб, несмещённо).
  3. Надёжность личной составляющей: совпадают ли у человека векторы, оценённые по чётным и нечётным пробам.
  4. Рост с паузой: размах при 0, 1, 3 с и отношение размаха к случайной дисперсии V (для стягивания к диагоналям
     такое отношение оказалось постоянным; здесь смотрим, так ли это).
  5. «Память без восприятия»: вектор при 3 с минус вектор при 0 с — его направление и размах.
  6. По группам: общая составляющая у здоровых, при шизофрении, при энцефалите.

ЗАПУСК (нужен wm_stein2.py в той же папке)
  python wm_stein4.py --selftest     # искусственные данные с заложенными гармониками: восстанавливает ли их метод
  python wm_stein4.py                # настоящие данные
  Результат: папка wm_stein4_out/ — report.md, results.json, figures.png
"""
import sys, os, json, time, argparse, platform
import numpy as np
import wm_stein2 as W

DELAYS = W.DELAYS; GROUPS = ['C', 'S', 'E']; GN = {'C': 'здоровые', 'S': 'шизофрения', 'E': 'энцефалит', 'ALL': 'все вместе'}; NBOOT = 2000; T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)


def fit_vectors(theta, e):
    """Совместная подгонка: стягивание к диагонали + гармоники 1 и 2. Возвращает lam и векторы (a, b) гармоник: a*sin(k*th) + b*cos(k*th)."""
    X = np.stack([W.dist_to_diag(theta), np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta), np.ones_like(theta)], 1)
    c = np.linalg.lstsq(X, e, rcond=None)[0]; return -float(c[0]), np.array([c[1], c[2]]), np.array([c[3], c[4]]), float((e - X @ c).var())

def to_dir(vec, k):
    """Вектор (a, b) -> (размах, направление притяжения в градусах). A*sin(k*(phi - th)) = -A*cos(k*phi)*sin(k*th) + A*sin(k*phi)*cos(k*th)."""
    a, b = vec; A = float(np.hypot(a, b)); phi = np.degrees(np.arctan2(b, -a)) / k
    return A, float(((phi + 180) % 360 - 180) if k == 1 else (phi % 180))


def synth(seed):
    """Искусственные данные: стягивание к диагоналям «через вывод»; общий сдвиг к направлению -90 град, растущий вместе с V;
    у каждого свой постоянный сдвиг (не растёт); общее стягивание к оси 0-180 град, растущее вместе с lam."""
    rng = np.random.default_rng(seed); S = {k: [] for k in ('subject', 'group', 'delay', 'target', 'prevcurr', 'err')}
    for g, ns in (('C', 19), ('S', 17), ('E', 16)):
        for k in range(ns):
            n = 1050; th = rng.uniform(-np.pi, np.pi, n); dl = rng.choice(DELAYS, n, p=[1 / 6, 2 / 3, 1 / 6]); pc = np.angle(np.exp(1j * (np.roll(th, 1) - th)))
            noise = np.exp(rng.normal(0, 0.3)); V = noise * (10.0 + 10.0 * (dl > 0) + 4.0 * dl); lam = V / (V + 250.0); own = rng.uniform(-np.pi, np.pi)
            e = -lam * W.dist_to_diag(th) + rng.normal(0, np.sqrt(V)) * (1 - lam)
            e = e + 0.08 * V * np.sin(np.radians(-90.0) - th) + 1.5 * np.sin(own - th) + 12.0 * lam * np.sin(2 * (0.0 - th))
            for key, v in zip(S, ([g + '%02d' % k] * n, [g] * n, dl, th, pc, e)): S[key].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in S.items()}


def per_subject(d):
    R = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; r = dict(group=d['group'][ms][0])
        for t in DELAYS:
            m = ms & (d['delay'] == t); e = d['err'][m]; th = d['target'][m]; e = e - e.mean(); x = W.dog(d['prevcurr'][m]); e = e - np.sum(e * x) / np.sum(x * x) * x
            odd = np.arange(len(e)) % 2 == 1; full = fit_vectors(th, e); a = fit_vectors(th[odd], e[odd]); b = fit_vectors(th[~odd], e[~odd])
            r[t] = dict(lam=full[0], v1=full[1], v2=full[2], V=full[3], v1a=a[1], v1b=b[1], v2a=a[2], v2b=b[2])
        R[s] = r
    return R


def summarize(R, subs, rng):
    out = {}
    for k, key in ((1, 'v1'), (2, 'v2')):
        H = {}
        for t in DELAYS:
            Vf = np.array([R[s][t][key] for s in subs]); Va = np.array([R[s][t][key + 'a'] for s in subs]); Vb = np.array([R[s][t][key + 'b'] for s in subs]); Vr = np.array([R[s][t]['V'] for s in subs]); n = len(subs)
            A, phi = to_dir(Vf.mean(0), k); bs = []
            for _ in range(NBOOT):
                i = rng.integers(0, n, n); a_, p_ = to_dir(Vf[i].mean(0), k); per = 360.0 / k; dphi = (p_ - phi + per / 2) % per - per / 2
                tot = np.mean(np.sum(Va[i] * Vb[i], 1)); com = float(np.dot(Va[i].mean(0), Vb[i].mean(0))); bs.append((a_, dphi, com / tot if tot > 0 else np.nan, a_ / Vr[i].mean()))
            bs = np.array(bs); q = lambda j: [float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]
            tot = float(np.mean(np.sum(Va * Vb, 1))); com = float(np.dot(Va.mean(0), Vb.mean(0)))
            rel = float(np.corrcoef(np.concatenate([Va[:, 0] - Va[:, 0].mean(), Va[:, 1] - Va[:, 1].mean()]), np.concatenate([Vb[:, 0] - Vb[:, 0].mean(), Vb[:, 1] - Vb[:, 1].mean()]))[0, 1])
            H[str(t)] = dict(A=[A] + q(0), phi=[phi, phi + q(1)[0], phi + q(1)[1]], power_total=tot, power_common=com, share_common=[com / tot if tot > 0 else np.nan] + q(2),
                             rel_personal=2 * rel / (1 + rel), A_over_V=[A / Vr.mean()] + q(3), amp_personal=[to_dir(v, k) for v in Vf.tolist()])
        V3 = np.array([R[s][3.0][key] for s in subs]); V0 = np.array([R[s][0.0][key] for s in subs]); D = V3 - V0; A, phi = to_dir(D.mean(0), k); n = len(subs); bs = []
        for _ in range(NBOOT):
            i = rng.integers(0, n, n); a_, p_ = to_dir(D[i].mean(0), k); per = 360.0 / k; bs.append((a_, (p_ - phi + per / 2) % per - per / 2))
        bs = np.array(bs); H['memory'] = dict(A=[A, float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))], phi=[phi, phi + float(np.percentile(bs[:, 1], 2.5)), phi + float(np.percentile(bs[:, 1], 97.5))])
        out[str(k)] = H
    return out


def f3(x, nd=2): return '%.*f [%.*f; %.*f]' % (nd, x[0], nd, x[1], nd, x[2])

def report(out, meta, outdir, tag=''):
    L = ['# Разведка: гармоники карты сноса с одним и двумя периодами%s\n' % tag]
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    if meta: L.append('- проб после отбора: %d (исходные сессии; повторные сессии post не использовались)' % meta['n_final'])
    L.append('- направления в координатах файла данных; в скобках 95 % интервал (бутстрэп по людям)')
    L.append('- это описание, а не проверка: всё замеченное здесь требует подтверждения на сессиях post\n')
    names = {'1': 'Один период: сдвиг всех положений в одну сторону', '2': 'Два периода: стягивание к одной оси'}
    for k in ('1', '2'):
        H = out['ALL'][k]; L.append('## %s\n' % names[k])
        L.append('| пауза | направление общей части, град | размах общей части, град | доля общей части в мощности | мощность всего, град² | надёжность личной части | размах / V |\n|---|---|---|---|---|---|---|')
        for t in ('0.0', '1.0', '3.0'):
            h = H[t]; L.append('| %s с | %s | %s | %s | %.2f | %.2f | %s |' % (t[0], f3(h['phi'], 0), f3(h['A']), f3(h['share_common']), h['power_total'], h['rel_personal'], f3(h['A_over_V'], 3)))
        m = H['memory']; L.append('\nПрирост от 0 с к 3 с (вектор при 3 с минус вектор при 0 с): направление %s град, размах %s град.' % (f3(m['phi'], 0), f3(m['A'])))
        L.append('\nПо группам, пауза 3 с:\n'); L.append('| группа | направление, град | размах, град | доля общей части | прирост 0→3 с: направление | прирост: размах |\n|---|---|---|---|---|---|')
        for g in GROUPS:
            h = out[g][k]['3.0']; m = out[g][k]['memory']; L.append('| %s | %s | %s | %s | %s | %s |' % (GN[g], f3(h['phi'], 0), f3(h['A']), f3(h['share_common']), f3(m['phi'], 0), f3(m['A'])))
        L.append('')
    L.append('## Для сравнения: стягивание к диагоналям у тех же людей\n'); L.append('| пауза | lam | V, град² |\n|---|---|---|')
    for t in ('0.0', '1.0', '3.0'): L.append('| %s с | %.3f | %.1f |' % (t[0], out['ref']['lam'][t], out['ref']['V'][t]))
    L.append('\n## Сводка для машинного чтения\n')
    slim = {g: {k: {t: {kk: vv for kk, vv in out[g][k][t].items() if kk != 'amp_personal'} for t in out[g][k]} for k in ('1', '2')} for g in ['ALL'] + GROUPS}
    L.append('```json\n' + json.dumps(dict(meta=meta, sets=slim, ref=out['ref']), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(17, 4.2)); col = {'0.0': '#1f77b4', '1.0': '#ff7f0e', '3.0': '#2ca02c'}
        for j, k in enumerate(('1', '2')):
            ax = fig.add_subplot(1, 4, 1 + 2 * j, projection='polar'); mult = int(k)
            for t in ('0.0', '3.0'):
                pts = np.array(out['ALL'][k][t]['amp_personal']); ax.plot(np.radians(pts[:, 1] * (1 if k == '1' else 1)), pts[:, 0], 'o', ms=3, color=col[t], alpha=0.6, label='люди, пауза %s с' % t[0])
                h = out['ALL'][k][t]; ax.plot([0, np.radians(h['phi'][0])], [0, h['A'][0]], '-', color=col[t], lw=2.5)
            ax.set_title(('Один период: куда сдвиг' if k == '1' else 'Два периода: ось притяжения\n(направления от 0 до 180)') + '\nточки — люди, линии — общая часть', fontsize=9); ax.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(-0.25, -0.2))
            ax2 = fig.add_subplot(1, 4, 2 + 2 * j); d = [float(x) for x in ('0', '1', '3')]
            A = np.array([out['ALL'][k][t]['A'] for t in ('0.0', '1.0', '3.0')]); ax2.errorbar(d, A[:, 0], yerr=[A[:, 0] - A[:, 1], A[:, 2] - A[:, 0]], fmt='-o', color='k', capsize=3, label='общая часть: размах, град')
            ax2.plot(d, [np.sqrt(max(out['ALL'][k][t]['power_total'], 0) * 2) for t in ('0.0', '1.0', '3.0')], '--s', color='gray', label='все вместе с личной: размах, град')
            ax2.set_xlabel('пауза, с'); ax2.set_ylim(0, None); ax2.set_title('Рост с паузой, %s' % ('один период' if k == '1' else 'два периода'), fontsize=10); ax2.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figures.png'), dpi=130)
    except ImportError: stage('matplotlib не установлен — рисунок пропущен')
    return text


def analyse(d):
    stage('оценки у каждого человека: векторы гармоник 1 и 2 (совместно со стягиванием к диагоналям)'); R = per_subject(d); rng = np.random.default_rng(41); out = {}
    stage('общая и личная части, рост с паузой, интервалы (бутстрэп по людям, %d повторов)' % NBOOT)
    for g in ['ALL'] + GROUPS:
        subs = [s for s in R if g == 'ALL' or R[s]['group'] == g]
        if len(subs) >= 5: out[g] = summarize(R, subs, rng)
    out['ref'] = dict(lam={str(t): float(np.mean([R[s][t]['lam'] for s in R])) for t in DELAYS}, V={str(t): float(np.mean([R[s][t]['V'] for s in R])) for t in DELAYS})
    return out


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--out', default='wm_stein4_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        stage('искусственные данные: общий сдвиг к -90 град (растёт вместе с V), личный постоянный сдвиг 1,5 град, общее стягивание к оси 0 град (растёт вместе с lam)')
        out = analyse(synth(1)); text = report(out, None, a.out + '_selftest', tag=' — ИСКУССТВЕННЫЕ ДАННЫЕ'); H1, H2 = out['ALL']['1'], out['ALL']['2']
        print('ПРОВЕРКА МЕТОДА: получено [интервал] / заложено')
        for t, V, lam in (('0.0', 10.0, 10 / 260.0), ('1.0', 24.0, 24 / 274.0), ('3.0', 32.0, 32 / 282.0)):
            print('  пауза %s с: один период — направление %s / -90; размах общей части %s / около %.2f; доля общей части %s / около %.2f' % (
                t[0], f3(H1[t]['phi'], 0), f3(H1[t]['A']), 0.08 * V * 1.045, f3(H1[t]['share_common']), (0.08 * V * 1.045) ** 2 / ((0.08 * V * 1.045) ** 2 + 1.5 ** 2)))
            print('               два периода — ось %s / 0 (или 180); размах %s / около %.2f' % (f3(H2[t]['phi'], 0), f3(H2[t]['A']), 12 * lam))
        print('  прирост 0→3 с, один период: направление %s / -90; размах %s / около %.2f' % (f3(H1['memory']['phi'], 0), f3(H1['memory']['A']), 0.08 * 22 * 1.045))
        print('  (заложенные размахи приблизительны: уровень шума у людей разный, среднее V чуть выше номинала)'); return
    stage('читаю %s' % a.data); d, meta = W.load_real(a.data); stage('проб после отбора %d, людей %d' % (meta['n_final'], len(np.unique(d['subject']))))
    out = analyse(d); text = report(out, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text + '=' * 67); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
