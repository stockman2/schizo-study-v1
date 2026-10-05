# -*- coding: utf-8 -*-
"""
wm_open1.py — ОПИСЬ открытых данных: сводный набор Ozkirli, Chetverikov и Pascucci (Nat Hum Behav 2025),
19 исследований воспроизведения ориентации или направления движения, стандартизованные таблицы.
Репозиторий: github.com/aozkirli/Large-scale-mega-analysis-on-serial-dependence
(при использовании нужно ссылаться и на сводную работу, и на исходные исследования — список в readMe.txt репозитория).

ЗАЧЕМ. Закон «стягивание пропорционально случайной дисперсии» и дифференциальная модель проверены у нас только на
одних данных (Stein и соавт., 2020). Здесь другие люди, лаборатории и задачи. Паузы в этих таблицах не записаны, зато
во многих есть условия, меняющие разброс ошибок другим способом (неопределённость стимула и т. п.) — это независимый
способ менять неопределённость.

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ. Только опись, без проверки гипотез: сколько людей и проб, какие условия, как покрыта окружность,
каков разброс ошибок в каждом условии. Сила стягивания здесь НЕ считается — чтобы правило проверки можно было записать
до того, как она будет увидена.

ПРИЗНАКИ ПРИГОДНОСТИ (печатаются для каждого опыта)
  А — есть не меньше двух условий с заметно разным разбросом (отношение стандартных отклонений не меньше 1,2)
      и не меньше 100 проб на человека в каждом;
  Б — стимулы покрывают окружность достаточно плотно (не меньше 12 разных значений, размах не меньше 150 град);
  В — не меньше 300 проб на человека (медиана).
  Для проверки закона нужны А, Б и В вместе; для карты смещений и профиля разброса хватит Б и В.

ЗАПУСК
  pip install numpy pandas
  python wm_open1.py            # скачает 19 таблиц в папку data_open/ и напечатает опись (несколько минут из-за загрузки)
  Результат: папка wm_open1_out/ — report.md, results.json
"""
import sys, os, json, time, argparse, platform
import numpy as np

BASE = 'https://raw.githubusercontent.com/aozkirli/Large-scale-mega-analysis-on-serial-dependence/main/data/'
FILES = ['01_Abreo_et_al_2023', '02_Blonde_et_al_2023_Exp2', '03_Ceylan_Pascucci_2023_Exp2', '04_Ceylan_et_al_2021', '05_Chetverikov_Jehee_2023', '06_Cicchini_et_al_2018',
         '07_Fischer_Whitney_2014b', '08_Fischer_et_al_2020', '11_Gallagher_Benton_2022', '13_Houborg_et_al_2023', '14_Houborg_et_al_2023b', '15_Kondo_et_al_2022', '16_Lau_Mau_2019',
         '17_Moon_Kwon_2022', '18_Moon_et_al_2023', '19_Ozkirli_Pascucci_2023', '20_Pascucci_et_al_2024', '21_Sadil_et_al_2024', '22_Samaha_et_al_2019']
T0 = time.time()
def stage(msg): print('[%6.1f с] %s' % (time.time() - T0, msg), flush=True)


def fetch(name, folder):
    path = os.path.join(folder, name + '.csv')
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(folder, exist_ok=True); stage('скачиваю %s.csv' % name); urllib.request.urlretrieve(BASE + name + '.csv', path)
    return path

def describe(path):
    import pandas as pd
    d = pd.read_csv(path, sep=';'); out = []
    if 'expnum' not in d.columns: d['expnum'] = 1
    if 'cond' not in d.columns: d['cond'] = 1
    for ex, g in d.groupby('expnum'):
        g = g[np.isfinite(g['error']) & np.isfinite(g['theta'])]; per = g.groupby('obs').size(); stim = str(g['stimulus'].iloc[0]) if 'stimulus' in g else '?'
        th = np.sort(g['theta'].round(3).unique()); rt_ok = bool('rt' in g and (g['rt'] != 9).mean() > 0.5)
        conds = []
        for c, gc in g.groupby('cond'):
            pc = gc.groupby('obs')['error'].agg(['std', 'size']); conds.append(dict(cond=str(c), n_obs=int(len(pc)), trials_per_obs=float(pc['size'].median()), sd=float(pc['std'].median())))
        conds.sort(key=lambda x: x['sd']); big = [c for c in conds if c['trials_per_obs'] >= 100]
        A = len(big) >= 2 and big[-1]['sd'] / max(big[0]['sd'], 1e-9) >= 1.2; B = len(th) >= 12 and (th.max() - th.min()) >= 150; C = per.median() >= 300
        out.append(dict(study=str(g['study'].iloc[0]) if 'study' in g else os.path.basename(path), experiment=str(g['experiment'].iloc[0]) if 'experiment' in g else str(ex), stimulus=stim,
                        n_obs=int(g['obs'].nunique()), n_trials=int(len(g)), trials_per_obs=float(per.median()), n_theta=int(len(th)), theta_min=float(th.min()), theta_max=float(th.max()),
                        sd_all=float(g.groupby('obs')['error'].std().median()), rt=rt_ok, conds=conds, A=bool(A), B=bool(B), C=bool(C), columns=list(d.columns)))
    return out


def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument('--folder', default='data_open'); ap.add_argument('--out', default='wm_open1_out'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    files = FILES[:a.limit] if a.limit else FILES; rows = []
    for name in files:
        try: p = fetch(name, a.folder); r = describe(p); rows += [dict(file=name, **x) for x in r]; stage('%s: опытов %d' % (name, len(r)))
        except Exception as e: stage('%s: НЕ ПРОЧИТАН (%s)' % (name, repr(e)[:120])); rows.append(dict(file=name, error=repr(e)[:200]))
    ok = [r for r in rows if 'error' not in r]; yes = lambda b: 'да' if b else '—'
    L = ['# Опись открытых наборов данных (сводный набор Ozkirli, Chetverikov, Pascucci 2025)\n']
    L.append('- среда: Python %s, numpy %s, %s; время %.0f с' % (platform.python_version(), np.__version__, platform.system(), time.time() - T0))
    L.append('- опытов всего: %d; людей (сумма по опытам): %d; проб: %d' % (len(ok), sum(r['n_obs'] for r in ok), sum(r['n_trials'] for r in ok)))
    L.append('- разброс — стандартное отклонение ошибки, градусы, медиана по людям; сила стягивания здесь не считается')
    L.append('- А: условия с разным разбросом; Б: окружность покрыта; В: достаточно проб на человека\n')
    L.append('| файл | опыт | стимул | людей | проб на человека | разных стимулов | разброс | условий | разброс по условиям (от меньшего к большему) | А | Б | В |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in ok:
        cs = '; '.join('%s: %.1f (%d проб)' % (c['cond'], c['sd'], c['trials_per_obs']) for c in r['conds'][:6]) + (' …' if len(r['conds']) > 6 else '')
        L.append('| %s | %s | %s | %d | %d | %d | %.1f | %d | %s | %s | %s | %s |' % (r['file'][:2], r['experiment'], r['stimulus'], r['n_obs'], r['trials_per_obs'], r['n_theta'], r['sd_all'], len(r['conds']), cs, yes(r['A']), yes(r['B']), yes(r['C'])))
    full = [r for r in ok if r['A'] and r['B'] and r['C']]; part = [r for r in ok if r['B'] and r['C'] and not r['A']]
    L.append('\n## Итог описи\n'); L.append('- годятся для проверки закона (А, Б и В): %d опытов, %d человек' % (len(full), sum(r['n_obs'] for r in full)))
    for r in full: L.append('  - %s, %s: %d человек' % (r['study'], r['experiment'], r['n_obs']))
    L.append('- годятся только для карты смещений и профиля разброса (Б и В): %d опытов, %d человек' % (len(part), sum(r['n_obs'] for r in part)))
    bad = [r for r in rows if 'error' in r]
    if bad: L.append('- не прочитаны: ' + ', '.join(r['file'] for r in bad))
    L.append('\n## Сводка для машинного чтения\n'); L.append('```json\n' + json.dumps(rows, ensure_ascii=False) + '\n```')
    text = '\n'.join(L) + '\n'; os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(rows, open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print('\n' + text[:text.find('## Сводка для машинного чтения')]); stage('готово: %s' % os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
