# -*- coding: utf-8 -*-
"""
wm_stein5.py — векторные поля сдвигов на окружности (данные Stein и соавт., 2020; исходные сессии, 52 человека).

ЧТО ТАКОЕ ПОЛЕ. В каждой точке окружности стрелка по касательной показывает, куда и насколько в среднем смещён ответ,
если точка была в этом месте. Против часовой стрелки — положительная ошибка (угол ответа больше угла точки).
Поля строятся для четырёх величин:
  «0 с»      — смещение при нулевой паузе (памяти ещё нет: восприятие и движение руки);
  «0→1 с»    — прирост смещения за первую секунду паузы;
  «1→3 с»    — прирост за секунду на отрезке от 1 до 3 с (разность, делённая на 2);
  «3 с»      — полное смещение при паузе 3 с.
И для четырёх составляющих карты:
  вся карта (гармоники 1…12); один период (общий сдвиг); два периода (ось); «кардинальная» (гармоники 4, 8, 12).
Устойчивая точка поля — место, куда стрелки сходятся; неустойчивая — откуда расходятся.

ТРИ ОГОВОРКИ
  1. Поле только касательное: при ответе на экране показана окружность нужного радиуса, поэтому по радиусу помнить нечего.
  2. Пауз всего три, значит, отрезков времени два; это не непрерывное по времени поле скоростей.
  3. Поле описывает, куда смещаются ОТВЕТЫ. Из прежних расчётов следует, что смещение пропорционально неопределённости
     и могло возникать в момент ответа, а не ползти во время паузы. Поэтому читать стрелки как «траекторию следа» нельзя.
  Направления даны в координатах файла данных (где на экране +90 град, неизвестно). Сессии post не используются.

ЗАПУСК (нужны wm_stein2.py и wm_stein4.py в той же папке)
  python wm_stein5.py --selftest     # искусственные данные с известными полями
  python wm_stein5.py                # настоящие данные
  Результат: папка wm_stein5_out/ — report.md, results.json и рисунки:
     fields_all.png       поля по всем людям (4 составляющие x 4 величины)
     fields_unrolled.png  те же поля, развёрнутые в график с интервалами
     fields_groups.png    вся карта по трём группам
     people_k1.png, people_k2.png   поле одного и двух периодов у каждого человека при паузе 3 с
"""
import sys, os, json, time, argparse, platform
import numpy as np
import wm_stein2 as W
import wm_stein4 as W4

KMAX = 12; NBOOT = 1000; GRID = np.radians(np.arange(-180, 180, 5.0)); T0 = time.time()
FAMS = [('full', 'вся карта', list(range(1, KMAX + 1))), ('k1', 'один период', [1]), ('k2', 'два периода', [2]), ('card', 'кардинальная (4, 8, 12)', [4, 8, 12])]
FIELDS = [('b0', '0 с'), ('d01', 'прирост 0→1 с'), ('d13', 'прирост за 1 с на отрезке 1→3 с'), ('b3', '3 с')]
GROUPS = ['C', 'S', 'E']; GN = {'C': 'здоровые', 'S': 'шизофрения', 'E': 'энцефалит', 'ALL': 'все люди'}
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)


def harm_fit(theta, e):
    """Коэффициенты гармоник 1…KMAX: строка [a1, b1, a2, b2, …], ошибка = сумма a_k*sin(k*th) + b_k*cos(k*th)."""
    ks = np.arange(1, KMAX + 1); X = np.concatenate([np.stack([np.sin(k * theta), np.cos(k * theta)], 1) for k in ks] + [np.ones((len(theta), 1))], 1)
    return np.linalg.lstsq(X, e, rcond=None)[0][:-1]

def recon(coef, ks, grid=GRID):
    """Поле на сетке углов по выбранным гармоникам. coef: (..., 2*KMAX)."""
    out = 0.0
    for k in ks: out = out + coef[..., [2 * (k - 1)]] * np.sin(k * grid) + coef[..., [2 * (k - 1) + 1]] * np.cos(k * grid)
    return out

def per_subject(d):
    R = {}
    for s in np.unique(d['subject']):
        ms = d['subject'] == s; c = {}
        for t in W.DELAYS:
            m = ms & (d['delay'] == t); e = d['err'][m]; e = e - e.mean(); x = W.dog(d['prevcurr'][m]); e = e - np.sum(e * x) / np.sum(x * x) * x; c[t] = harm_fit(d['target'][m], e)
        R[s] = dict(group=d['group'][ms][0], b0=c[0.0], d01=c[1.0] - c[0.0], d13=(c[3.0] - c[1.0]) / 2.0, b3=c[3.0])
    return R

def crossings(f):
    """Места, где поле меняет знак: устойчивые (стрелки сходятся) и неустойчивые (расходятся), град."""
    g = np.degrees(GRID); st, un = [], []; n = len(f)
    for i in range(n):
        a, b = f[i], f[(i + 1) % n]
        if a == 0 or a * b < 0:
            x = g[i] + 5.0 * (a / (a - b)) if a != b else g[i]; x = (x + 180) % 360 - 180; (st if a > 0 else un).append(round(float(x)))
    return st, un

def analyse(d):
    stage('гармоники 1…%d у каждого человека при каждой паузе' % KMAX); R = per_subject(d); rng = np.random.default_rng(51); out = dict(grid=np.degrees(GRID).tolist(), sets={})
    stage('поля по группам и интервалы (бутстрэп по людям, %d повторов)' % NBOOT)
    for g in ['ALL'] + GROUPS:
        subs = [s for s in R if g == 'ALL' or R[s]['group'] == g]
        if len(subs) < 5: continue
        G = {}; n = len(subs); idx = rng.integers(0, n, (NBOOT, n))
        for fk, _ in FIELDS:
            C = np.array([R[s][fk] for s in subs])                     # люди x 24
            for fam, _, ks in FAMS:
                F = recon(C, ks); mean = F.mean(0); bs = F[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5], 0); st, un = crossings(mean); mx = np.abs(bs).max(1)
                G[fam + '|' + fk] = dict(mean=mean.tolist(), lo=lo.tolist(), hi=hi.tolist(), max=[float(np.abs(mean).max()), float(np.percentile(mx, 2.5)), float(np.percentile(mx, 97.5))],
                                         sig_share=float(np.mean((lo > 0) | (hi < 0))), stable=st, unstable=un)
        out['sets'][g] = G
    out['people'] = {s: dict(group=R[s]['group'], k1=recon(R[s]['b3'], [1]).tolist(), k2=recon(R[s]['b3'], [2]).tolist(), a1=W4.to_dir(R[s]['b3'][0:2], 1), a2=W4.to_dir(R[s]['b3'][2:4], 2)) for s in R}
    return out


# ---------- рисунки ----------
def draw_field(ax, grid_deg, mean, lo, hi, scale, title=None, step=2, lw=0.011):
    th = np.radians(np.asarray(grid_deg))[::step]; f = np.asarray(mean)[::step]; sig = ((np.asarray(lo) > 0) | (np.asarray(hi) < 0))[::step] if lo is not None else np.ones(len(f), bool)
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color='0.75', lw=0.8)
    L = 0.5 * f / scale if scale > 0 else f * 0; col = np.where(~sig, '0.7', np.where(f > 0, '#c0392b', '#2471a3'))
    ax.quiver(np.cos(th), np.sin(th), -np.sin(th) * L, np.cos(th) * L, angles='xy', scale_units='xy', scale=1, width=lw, color=list(col), headwidth=3.5, headlength=4)
    for a, lab in ((0, '0'), (90, '90'), (180, '180'), (-90, '−90')): ax.text(1.42 * np.cos(np.radians(a)), 1.42 * np.sin(np.radians(a)), lab, ha='center', va='center', fontsize=6, color='0.4')
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=8)

def mark_points(ax, st, un):
    for a in st: ax.plot(np.cos(np.radians(a)), np.sin(np.radians(a)), 'o', color='#1e8449', ms=5)
    for a in un: ax.plot(np.cos(np.radians(a)), np.sin(np.radians(a)), 'o', mfc='white', mec='k', ms=5)

def figures(out, outdir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    g = out['grid']; A = out['sets']['ALL']
    fig, ax = plt.subplots(4, 4, figsize=(13, 13.5))
    for i, (fam, fname, _) in enumerate(FAMS):
        sc = max(A[fam + '|' + fk]['max'][0] for fk, _ in FIELDS)
        for j, (fk, ftitle) in enumerate(FIELDS):
            h = A[fam + '|' + fk]; draw_field(ax[i][j], g, h['mean'], h['lo'], h['hi'], sc, title='%s: %s\nнаибольшая стрелка %.1f°' % (fname, ftitle, h['max'][0]))
            if h['sig_share'] > 0.3: mark_points(ax[i][j], h['stable'], h['unstable'])
    fig.suptitle('Поля сдвигов, все люди. Красные стрелки — против часовой, синие — по часовой, серые — неотличимо от нуля.\nЗелёные точки — устойчивые (стрелки сходятся), белые — неустойчивые. В каждой строке свой масштаб.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(outdir, 'fields_all.png'), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True); col = {'b0': '#1f77b4', 'd01': '#ff7f0e', 'd13': '#2ca02c', 'b3': 'k'}
    for i, (fam, fname, _) in enumerate(FAMS):
        for fk, ftitle in FIELDS:
            h = A[fam + '|' + fk]; ax[i].plot(g, h['mean'], color=col[fk], lw=1.4 if fk != 'b3' else 1.0, ls='-' if fk != 'b3' else '--', label=ftitle); ax[i].fill_between(g, h['lo'], h['hi'], color=col[fk], alpha=0.15)
        ax[i].axhline(0, color='gray', lw=0.6); ax[i].set_ylabel('сдвиг, град'); ax[i].set_title(fname, fontsize=10)
        for v in (-90, 0, 90): ax[i].axvline(v, color='0.85', lw=0.6)
    ax[0].legend(frameon=False, fontsize=8, ncol=4); ax[-1].set_xlabel('место точки, град (координаты файла)'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fields_unrolled.png'), dpi=120); plt.close(fig)
    gs = [x for x in GROUPS if x in out['sets']]; fig, ax = plt.subplots(len(gs), 4, figsize=(13, 3.4 * len(gs)), squeeze=False)
    sc = max(out['sets'][x]['full|' + fk]['max'][0] for x in gs for fk, _ in FIELDS)
    for i, x in enumerate(gs):
        for j, (fk, ftitle) in enumerate(FIELDS):
            h = out['sets'][x]['full|' + fk]; draw_field(ax[i][j], g, h['mean'], h['lo'], h['hi'], sc, title='%s: %s\nнаибольшая стрелка %.1f°' % (GN[x], ftitle, h['max'][0]))
    fig.suptitle('Вся карта по группам (масштаб общий)', fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(outdir, 'fields_groups.png'), dpi=120); plt.close(fig)
    gc = {'C': 'k', 'S': '#c0392b', 'E': '#2471a3'}; P = out['people']; subs = sorted(P, key=lambda s: (GROUPS.index(P[s]['group']) if P[s]['group'] in GROUPS else 9, s))
    for key, akey, name in (('k1', 'a1', 'один период (сдвиг в одну сторону)'), ('k2', 'a2', 'два периода (ось)')):
        cols = 9; rows = int(np.ceil(len(subs) / cols)); sc = np.percentile([np.abs(P[s][key]).max() for s in subs], 90); fig, ax = plt.subplots(rows, cols, figsize=(1.75 * cols, 1.9 * rows), squeeze=False)
        for i in range(rows * cols):
            a_ = ax[i // cols][i % cols]
            if i >= len(subs): a_.axis('off'); continue
            s = subs[i]; draw_field(a_, g, P[s][key], None, None, sc, step=6, lw=0.02); a_.set_title('%s  %.1f°' % (s, P[s][akey][0]), fontsize=7, color=gc.get(P[s]['group'], 'k'))
        fig.suptitle('Поле «%s» у каждого человека при паузе 3 с. Число — размах; цвет подписи — группа (чёрный — здоровые, красный — шизофрения, синий — энцефалит)' % name, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(outdir, 'people_%s.png' % key), dpi=110); plt.close(fig)


def report(out, meta, outdir, tag=''):
    A = out['sets']['ALL']; L = ['# Векторные поля сдвигов на окружности%s\n' % tag]
    L.append('- среда: Python %s, numpy %s, %s; время счёта %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    if meta: L.append('- проб после отбора: %d (исходные сессии)' % meta['n_final'])
    L.append('- сдвиг в градусах дуги; плюс — против часовой стрелки; углы в координатах файла; в скобках 95 % интервал наибольшего сдвига')
    L.append('- «надёжная доля» — на какой части окружности поле отличимо от нуля; точки смены знака перечислены, только если эта доля больше 0,3\n')
    for fam, fname, ks in FAMS:
        L.append('## %s\n' % fname.capitalize()); L.append('| величина | наибольший сдвиг, град | надёжная доля | устойчивые точки, град | неустойчивые точки, град |\n|---|---|---|---|---|')
        for fk, ftitle in FIELDS:
            h = A[fam + '|' + fk]; ok = h['sig_share'] > 0.3
            L.append('| %s | %.2f [%.2f; %.2f] | %.2f | %s | %s |' % (ftitle, h['max'][0], h['max'][1], h['max'][2], h['sig_share'], ', '.join(map(str, h['stable'])) if ok else '—', ', '.join(map(str, h['unstable'])) if ok else '—'))
        L.append('')
    L.append('## Вся карта по группам: наибольший сдвиг, град\n'); L.append('| группа | 0 с | прирост 0→1 с | прирост за 1 с (1→3 с) | 3 с |\n|---|---|---|---|---|')
    for g in GROUPS:
        if g in out['sets']: L.append('| %s | ' % GN[g] + ' | '.join('%.2f [%.2f; %.2f]' % tuple(out['sets'][g]['full|' + fk]['max']) for fk, _ in FIELDS) + ' |')
    P = out['people']; a2 = np.array([P[s]['a2'][1] for s in P]); a1 = np.array([P[s]['a1'][0] for s in P]); A2 = np.array([P[s]['a2'][0] for s in P])
    near = np.minimum(np.abs((a2 + 22.5) % 90 - 22.5), 90)       # расстояние личной оси до ближайшей из осей 0 или 90
    d0 = np.minimum(a2 % 90, 90 - a2 % 90); L.append('\n## Личные поля при паузе 3 с\n')
    L.append('- один период: размах у людей — медиана %.1f°, от %.1f до %.1f°' % (np.median(a1), a1.min(), a1.max()))
    L.append('- два периода: размах — медиана %.1f°, от %.1f до %.1f°; личная ось ближе 22,5° к оси 0° или 90° у %d из %d человек (при случайных осях ожидалась бы половина)' % (np.median(A2), A2.min(), A2.max(), int((d0 < 22.5).sum()), len(a2)))
    L.append('\n## Сводка для машинного чтения\n')
    slim = {g: {k: {kk: vv for kk, vv in v.items() if kk in ('max', 'sig_share', 'stable', 'unstable')} for k, v in out['sets'][g].items()} for g in out['sets']}
    L.append('```json\n' + json.dumps(dict(meta=meta, sets=slim, people={s: dict(group=P[s]['group'], k1=P[s]['a1'], k2=P[s]['a2']) for s in P}), ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(out, open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    try: stage('рисунки'); figures(out, outdir)
    except ImportError: stage('matplotlib не установлен — рисунки пропущены')
    return text


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join('data', 'behavior.pkl')); ap.add_argument('--out', default='wm_stein5_out'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest:
        stage('искусственные данные: стягивание к диагоналям; общий сдвиг к -90 град; общее стягивание к оси 0-180 град'); out = analyse(W4.synth(1)); report(out, None, a.out + '_selftest', tag=' — ИСКУССТВЕННЫЕ ДАННЫЕ'); A = out['sets']['ALL']
        print('ПРОВЕРКА МЕТОДА: точки поля при паузе 3 с, получено / заложено')
        print('  кардинальная: устойчивые %s / -135, -45, 45, 135; неустойчивые %s / -180, -90, 0, 90' % (A['card|b3']['stable'], A['card|b3']['unstable']))
        print('  один период:  устойчивая %s / -90; неустойчивая %s / 90; наибольший сдвиг %.2f / около 2,7' % (A['k1|b3']['stable'], A['k1|b3']['unstable'], A['k1|b3']['max'][0]))
        print('  два периода:  устойчивые %s / -180 (то же, что 180) и 0; неустойчивые %s / -90, 90; наибольший сдвиг %.2f / около 1,4' % (A['k2|b3']['stable'], A['k2|b3']['unstable'], A['k2|b3']['max'][0]))
        print('  рисунки для просмотра: папка %s_selftest' % a.out); return
    stage('читаю %s' % a.data); d, meta = W.load_real(a.data); stage('проб после отбора %d, людей %d' % (meta['n_final'], len(np.unique(d['subject']))))
    out = analyse(d); text = report(out, meta, a.out); print('\n' + '=' * 30 + ' ОТЧЁТ ' + '=' * 30 + '\n' + text + '=' * 67); stage('готово: папка %s' % a.out)


if __name__ == '__main__':
    main()
