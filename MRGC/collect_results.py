"""Collect repro results from result/*.txt into a report log and repro_results.csv.

Usage (from the MRGC folder):  python collect_results.py
Writes reports/report_<timestamp>.txt, overwrites repro_results.csv, and prints the report.
"""
import csv
import datetime
import glob
import os
import re
import subprocess

# (dataset, reduction_rate) -> (condensation ratio %, paper test accuracy), paper Table 1
PAPER = {
    ('Cora', '0.25'): (1.30, 77.43),
    ('Cora', '0.5'): (2.60, 76.72),
    ('Cora', '1.0'): (5.20, 74.40),
    ('CiteSeer', '0.25'): (0.90, 65.12),
    ('CiteSeer', '0.5'): (1.80, 63.87),
    ('CiteSeer', '1.0'): (3.60, 64.44),
    ('dblp', '0.25'): (0.11, 65.11),
    ('dblp', '0.5'): (0.23, 65.49),
    ('dblp', '1.0'): (0.45, 64.60),
    ('PubMed', '0.25'): (0.08, 74.85),
    ('PubMed', '0.5'): (0.15, 73.11),
    ('PubMed', '1.0'): (0.30, 73.04),
    ('ogbn-arxiv', '0.001'): (0.05, 59.38),
    ('ogbn-arxiv', '0.005'): (0.25, 61.79),
    ('ogbn-arxiv', '0.01'): (0.50, 63.32),
}


def parse_log(path):
    with open(path, errors='replace') as f:
        text = f.read()
    # Logs are appended to; prefer the newest attempt that finished, else the newest one
    segments = text.split('====================Ratio')[1:] or [text]
    finished = [s for s in segments if 'Final Test Accuracy' in s]
    seg = finished[-1] if finished else segments[-1]

    m = re.search(r"'runs': (\d+)", seg)
    res = {'attempts': len(segments), 'runs_expected': int(m.group(1)) if m else 5,
           'seeds': [], 'final': None, 'last_epoch': None, 'last_test': None}
    skips = None
    for line in seg.splitlines():
        m = re.match(r'Epoch\s+(\d+)', line)
        if m:
            res['last_epoch'] = int(m.group(1))
            continue
        m = re.match(r'Test Accuracy Mean ([\d.]+)', line)
        if m:
            res['last_test'] = float(m.group(1)) * 100
            continue
        m = re.match(r'Non-finite steps skipped so far: (\d+)', line)
        if m:
            skips = int(m.group(1))
            continue
        m = re.match(r'Run (\d+): best epoch (\d+), Val ([\d.]+), Test ([\d.]+)', line)
        if m:
            res['seeds'].append({'run': int(m.group(1)), 'epoch': int(m.group(2)),
                                 'val': float(m.group(3)) * 100, 'test': float(m.group(4)) * 100,
                                 'skips': skips})
            skips = None
    m = re.search(r'Final Test Accuracy over (\d+) runs: ([\d.]+) \+/- ([\d.]+)', seg)
    if m:
        res['final'] = (float(m.group(2)), float(m.group(3)))
    return res


def running_commands():
    try:
        return subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout.splitlines()
    except Exception:
        return []


def fmt(x, nd=2, sign=False):
    if x is None:
        return '-'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def main():
    procs = [p for p in running_commands() if 'main.py' in p]
    logs = {}
    for path in sorted(glob.glob('result/*.txt')):
        # <name>-<rate>-<log_index>-<fb>-<sb>-<lb>.txt; name may contain '-' (ogbn-arxiv)
        parts = os.path.basename(path)[:-4].rsplit('-', 5)
        if len(parts) == 6:
            logs[(parts[0], parts[1])] = (path, parse_log(path))

    rows = []
    for key in list(PAPER) + [k for k in logs if k not in PAPER]:
        name, rate = key
        ratio, paper = PAPER.get(key, (None, None))
        path, res = logs.get(key, (None, None))
        row = {'dataset': name, 'reduction_rate': rate, 'ratio': ratio, 'paper': paper,
               'mean': None, 'std': None, 'seeds_done': 0, 'max_skips': None, 'status': 'not started',
               'notes': [], 'res': res, 'path': path}
        if res:
            seeds = res['seeds']
            row['seeds_done'] = len(seeds)
            skip_vals = [s['skips'] for s in seeds if s['skips'] is not None]
            row['max_skips'] = max(skip_vals) if skip_vals else None
            if res['final']:
                row['mean'], row['std'] = res['final']
                row['status'] = 'done'
            else:
                if seeds:
                    tests = [s['test'] for s in seeds]
                    mean = sum(tests) / len(tests)
                    row['mean'] = mean
                    row['std'] = (sum((t - mean) ** 2 for t in tests) / len(tests)) ** 0.5
                    row['notes'].append(f'partial mean over {len(seeds)} seeds')
                alive = any(f'--name {name} ' in p and f'--reduction_rate {rate}' in p for p in procs)
                row['status'] = 'running' if alive else 'stopped/incomplete'
                if res['last_epoch']:
                    row['notes'].append(f"seed {len(seeds)} at epoch {res['last_epoch']}, "
                                        f"last test {fmt(res['last_test'])}")
            if res['attempts'] > 1:
                row['notes'].append(f"{res['attempts']} attempts in log, used newest "
                                    f"{'finished' if res['final'] else 'one'}")
        if glob.glob(os.path.join('result', 'failed_nan', f'{name}-{rate}-*.txt')):
            row['notes'].append('earlier attempt diverged to NaN (result/failed_nan)')
        if row['max_skips']:
            row['notes'].append('NaN guard skipped steps - disclose')
        row['diff'] = row['mean'] - paper if row['mean'] is not None and paper is not None else None
        rows.append(row)

    now = datetime.datetime.now()
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = 'unknown'

    out = []
    out.append('MRGC reproduction report')
    out.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}   git commit: {commit or 'unknown'}")
    out.append(f'Experiments running now: {len(procs)}')
    out.append('Test accuracy (%), mean +/- std over seeds; diff = ours - paper')
    out.append('')
    header = f"{'Dataset':<11}{'Rate':>7}{'Ratio%':>8}{'Ours':>17}{'Paper':>8}{'Diff':>8}{'Seeds':>7}{'MaxSkip':>9}  Status"
    out.append(header)
    out.append('-' * len(header))
    for r in rows:
        ours = '-' if r['mean'] is None else f"{fmt(r['mean'])} +/- {fmt(r['std'])}"
        exp = r['res']['runs_expected'] if r['res'] else 5
        out.append(f"{r['dataset']:<11}{r['reduction_rate']:>7}{fmt(r['ratio']):>8}{ours:>17}"
                   f"{fmt(r['paper']):>8}{fmt(r['diff'], sign=True):>8}{str(r['seeds_done']) + '/' + str(exp):>7}"
                   f"{fmt(r['max_skips'], 0):>9}  {r['status']}")

    done = [r for r in rows if r['status'] == 'done' and r['diff'] is not None]
    out.append('')
    out.append(f'Finished: {len(done)}/{len(PAPER)} paper experiments')
    if done:
        within = sum(abs(r['diff']) <= r['std'] for r in done)
        out.append(f'Within 1 std of paper: {within}/{len(done)}')
        out.append(f"Mean diff vs paper: {sum(r['diff'] for r in done) / len(done):+.2f}")

    out.append('')
    out.append('Per-seed detail')
    out.append('---------------')
    for r in rows:
        if not r['res']:
            continue
        out.append(f"{r['dataset']} {r['reduction_rate']}  ({r['path']})")
        for s in r['res']['seeds']:
            out.append(f"  run {s['run']}: best epoch {s['epoch']:4d}  val {s['val']:.2f}  test {s['test']:.2f}"
                       f"  skipped steps {fmt(s['skips'], 0)}")
        for n in r['notes']:
            out.append(f'  note: {n}')

    out.append('')
    out.append('Driver logs')
    out.append('-----------')
    for log in ['ratio_log.txt', 'rerun_log.txt', 'budget_log.txt']:
        if os.path.exists(log):
            with open(log, errors='replace') as f:
                text = f.read()
            out.append(f"{log}: NaN exits {text.count('tensor(True') // 4}, "
                       f"tracebacks {text.count('Traceback')}")

    report = '\n'.join(out) + '\n'
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', f"report_{now.strftime('%Y%m%d-%H%M%S')}.txt")
    with open(report_path, 'w', encoding='ascii', errors='replace') as f:
        f.write(report)

    with open('repro_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'reduction_rate', 'condensation_ratio_pct', 'ours_mean', 'ours_std',
                    'paper_mean', 'diff_vs_paper', 'seeds_done', 'max_skipped_steps', 'status', 'notes'])
        for r in rows:
            w.writerow([r['dataset'], r['reduction_rate'], fmt(r['ratio']), fmt(r['mean']).strip('-'),
                        fmt(r['std']).strip('-'), fmt(r['paper']).strip('-'),
                        '' if r['diff'] is None else f"{r['diff']:.2f}", r['seeds_done'],
                        '' if r['max_skips'] is None else r['max_skips'], r['status'], '; '.join(r['notes'])])

    print(report)
    print(f'Wrote {report_path} and repro_results.csv')


if __name__ == '__main__':
    main()
