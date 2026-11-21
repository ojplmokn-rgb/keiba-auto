# =====================================================================
#  WIN5 自動予想・記録・答え合わせ（GitHub Actions用）  買い方E・200点・お金はかけない記録
#   python win5_auto.py predict … 今日のWIN5の締切12分前まで待ち、その時点のオッズで予想して記録
#   python win5_auto.py settle  … 結果が出た回を答え合わせして成績表（report_win5.md）を更新
#  買い方E：荒れそうな2レース（予想の本命の勝率が低い2レース）は 1・2番人気＋6〜10番人気、
#           残り3レースは 1〜5番人気。当たりやすい上位600点の中から期待値の高い200点。
#  モデル：Googleドライブ keiba_auto_jra の w5_model.json / w5_jockey.csv / w5_trainer.csv / w5_horse.csv
#  マナー：netkeibaにはログインしない／待ち時間は短くしない
# =====================================================================
import os, re, sys, json, time, warnings, subprocess
from io import StringIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

MODE = sys.argv[1] if len(sys.argv) > 1 else 'predict'
JST = ZoneInfo('Asia/Tokyo')
NOW = datetime.now(JST)
TODAY = NOW.strftime('%Y%m%d')
DRV = os.environ.get('DRIVE_DIR', 'drive_jra')
OUT = 'data/win5'
REPORT = os.environ.get('REPORT_WIN5', 'report_win5.md')
LOG = f'{OUT}/win5_成績.csv'
os.makedirs(OUT, exist_ok=True)
MAXPOP, POINTS, WAIT = 10, 200, 3.0
OWNER = 'ojplmokn-rgb'
print(f'WIN5 自動予想（買い方E）　モード：{MODE}　{NOW:%Y-%m-%d %H:%M}')

S = requests.Session()
S.headers.update({'User-Agent': 'Mozilla/5.0 (personal research, low rate)'})

def get(url, wait=WAIT):
    for _ in range(3):
        try:
            r = S.get(url, timeout=30)
            time.sleep(wait)
            if r.status_code == 200:
                try:
                    return r.content.decode('utf-8')
                except UnicodeDecodeError:
                    return r.content.decode('euc_jp', errors='replace')
        except Exception:
            time.sleep(5)
    return None

def flat(c):
    if isinstance(c, tuple):
        c = ''.join(str(x) for x in c)
    return re.sub(r'\s', '', str(c))

def pick(cols, *keys):
    for k in keys:
        for c in cols:
            if k in c:
                return c
    return None

def parse_yen(s):
    s = s.replace(',', '')
    m = re.fullmatch(r'(?:(\d+)億)?(?:(\d+)万)?(\d*)', s)
    if not m or not s:
        return np.nan
    a, b, c = m.groups()
    return int(a or 0) * 10**8 + int(b or 0) * 10**4 + int(c or 0)

def gh(*a):
    return subprocess.run(['gh', *a], capture_output=True, text=True)

def notify(day, text):
    """その日の Issue にコメントして、スマホに通知する"""
    if not os.environ.get('GITHUB_ACTIONS'):
        print('（通知）\n' + text[:3000]); return
    fp = f'{OUT}/issue_{day}.txt'
    num = open(fp).read().strip() if os.path.exists(fp) else None
    if not num:
        r = gh('issue', 'create', '--title', f'{day[:4]}-{day[4:6]}-{day[6:]} WIN5の予想', '--assignee', OWNER,
               '--body', f'@{OWNER} WIN5の予想（買い方E・200点）です。締切の約12分前のオッズで計算します。'
                         '答え合わせもここに届きます。\n\n※お金はかけない記録です。')
        m = re.search(r'/issues/(\d+)', r.stdout)
        if not m:
            print('通知を作れませんでした：', r.stderr[:200]); return
        num = m.group(1)
        open(fp, 'w').write(num)
    r = gh('issue', 'comment', num, '--body', f'@{OWNER} ' + text)
    if r.returncode != 0:
        print('通知できませんでした：', r.stderr[:200])

# ---------- netkeiba ----------
def win5_page(d):
    h = get(f'https://race.netkeiba.com/top/win5.html?date={d}')
    if not h:
        return None
    soup = BeautifulSoup(h, 'html.parser')
    ids = []
    for a in soup.find_all('a', href=True):
        m = re.search(r'race_id=(\d{12})', a['href'])
        if m and m.group(1) not in ids and m.group(1)[:4] == d[:4]:
            ids.append(m.group(1))
    text = soup.get_text(' ', strip=True)
    m = re.search(r'払戻金?\s*[:：]?\s*([0-9][0-9億万,]*)円', text)
    pay = parse_yen(m.group(1)) if m else np.nan
    m = re.search(r'的中票数\s*[:：]?\s*([0-9,]+)\s*票', text)
    votes = int(m.group(1).replace(',', '')) if m else np.nan
    return {'race_ids': ids[:5], 'payout': pay, 'hit_votes': votes}

def race_winner(rid):
    h = get(f'https://race.netkeiba.com/race/result.html?race_id={rid}')
    if not h or '着順' not in h:
        return None
    try:
        for t in pd.read_html(StringIO(h)):
            cols = [flat(c) for c in t.columns]
            if pick(cols, '着順') and pick(cols, '馬番'):
                t.columns = cols
                rk = pd.to_numeric(t[pick(cols, '着順')], errors='coerce')
                nums = pd.to_numeric(t[pick(cols, '馬番')], errors='coerce')
                w = nums[rk == 1]
                return int(w.iloc[0]) if len(w) else None
    except Exception:
        return None
    return None

def shutuba(rid):
    h = get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}')
    if not h:
        return None, None
    soup = BeautifulSoup(h, 'html.parser')
    d1 = soup.select_one('.RaceData01')
    d1 = d1.get_text(' ', strip=True) if d1 else ''
    nm = soup.select_one('.RaceName')
    info = {'name': nm.get_text(strip=True) if nm else rid}
    m = re.search(r'(\d{1,2}):(\d{2})\s*発走', d1)
    info['start'] = (int(m.group(1)), int(m.group(2))) if m else None
    m = re.search(r'(芝|ダ|障)\D{0,3}(\d{3,4})', d1)
    info['surf'], info['dist'] = (m.group(1), float(m.group(2))) if m else (None, np.nan)
    t = None
    for tb in pd.read_html(StringIO(h)):
        cols = [flat(c) for c in tb.columns]
        if pick(cols, '馬番') and pick(cols, '馬名') and pick(cols, '騎手'):
            tb.columns = cols
            t = tb
            break
    if t is None:
        return None, info
    cols = list(t.columns)
    g = lambda *k: t[pick(cols, *k)] if pick(cols, *k) else pd.Series([np.nan] * len(t))
    df = pd.DataFrame({'num': pd.to_numeric(g('馬番'), errors='coerce'),
                       'name': g('馬名').astype(str).str.replace(r'\s', '', regex=True),
                       'kin': pd.to_numeric(g('斤量'), errors='coerce')})
    sa = g('性齢').astype(str)
    df['sex'], df['age'] = sa.str[0], pd.to_numeric(sa.str.extract(r'(\d+)')[0], errors='coerce')
    df['jockey'] = g('騎手').astype(str).str.replace(r'[▲△☆◇★\s]', '', regex=True)
    df['trainer'] = g('厩舎', '調教師').astype(str).str.replace(r'\s', '', regex=True)
    w = g('馬体重').astype(str).str.extract(r'(\d{3})\s*\(([+\-]?\d+)\)')
    df['wt'], df['wtdiff'] = pd.to_numeric(w[0], errors='coerce'), pd.to_numeric(w[1], errors='coerce')
    best = None
    for tb in soup.find_all('table'):
        n = len(tb.find_all('a', href=re.compile(r'/horse/\d+')))
        if n and (best is None or n > best[0]):
            best = (n, tb)
    ids = []
    if best:
        for tr in best[1].find_all('tr'):
            if tr.find('td'):
                a = tr.find('a', href=re.compile(r'/horse/\d+'))
                ids.append(re.search(r'/horse/(\d+)', a['href']).group(1) if a else None)
    df['horse_id'] = ids if len(ids) == len(df) else None
    df = df[df['num'].notna()].copy()
    df['num'] = df['num'].astype(int)
    return df, info

def odds_now(rid):
    h = get(f'https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=update', 1.0)
    out = {}
    try:
        od = json.loads(h)['data']['odds']
        od = od.get('1', od) if isinstance(od, dict) else od
        for k, v in od.items():
            val = v[0] if isinstance(v, list) else v
            try:
                out[int(k)] = float(val)
            except Exception:
                pass
    except Exception:
        pass
    return out

def horse_hist(hid, today_dt):
    h = get(f'https://db.netkeiba.com/horse/result/{hid}/')
    if not h:
        return None
    try:
        for t in pd.read_html(StringIO(h)):
            cols = [flat(c) for c in t.columns]
            if pick(cols, '日付') and pick(cols, '着順'):
                t.columns = cols
                g = lambda *k: t[pick(cols, *k)] if pick(cols, *k) else pd.Series([np.nan] * len(t))
                dist = g('距離').astype(str)
                r = pd.DataFrame({'dt': pd.to_datetime(g('日付').astype(str).str.replace('.', '/', regex=False), errors='coerce'),
                                  'rank': pd.to_numeric(g('着順'), errors='coerce'),
                                  'pop': pd.to_numeric(g('人気'), errors='coerce'),
                                  'field': pd.to_numeric(g('頭数'), errors='coerce'),
                                  'margin': pd.to_numeric(g('着差'), errors='coerce'),
                                  'agari': pd.to_numeric(g('上り', '上がり'), errors='coerce'),
                                  'surf': dist.str.extract(r'(芝|ダ|障)')[0],
                                  'dist': pd.to_numeric(dist.str.extract(r'(\d{3,4})')[0], errors='coerce')})
                r.loc[r['rank'] == 1, 'margin'] = r.loc[r['rank'] == 1, 'margin'].clip(upper=0)
                return r[(r['dt'] < today_dt) & r['rank'].notna()].sort_values('dt')
    except Exception:
        return None
    return None

# ---------- 答え合わせ ----------
def write_report(log):
    L = ['# WIN5 自動予想の成績（買い方E・200点・お金はかけない記録）', '', f'更新：{NOW:%Y-%m-%d %H:%M}', '']
    if len(log):
        cost = log['点数'].sum() * 100
        ret = log['払戻（的中時）'].sum()
        L += ['## 全期間', '', '| 回数 | 的中 | 投資 | 払戻 | 収支 | 回収率 |', '|---|---|---|---|---|---|',
              f'| {len(log)} | {int(log["的中"].sum())} | {cost:,.0f}円 | {ret:,.0f}円 | {ret - cost:+,.0f}円 | {ret / cost * 100:.0f}% |', '']
        L += ['## 回ごと（新しい順）', '', '| 日付 | 勝ち馬（人気） | 払戻金 | 買い目に入った勝ち馬 | 的中 |', '|---|---|---|---|---|']
        for _, r in log.sort_values('date', ascending=False).iterrows():
            pay = f'{r["払戻金"]:,.0f}円' if pd.notna(r['払戻金']) else 'キャリーオーバー'
            L.append(f'| {r["date"][:4]}-{r["date"][4:6]}-{r["date"][6:]} | {r["勝ち馬"]}（{r.get("人気", "")}） | {pay} | '
                     f'{r.get("入ったレース", "")} | {"★的中" if r["的中"] else "－"} |')
    else:
        L += ['まだ記録がありません。']
    L += ['', '※過去の検証（2014〜2026年）では、買い方E・200点は回収率196%、的中は約21回に1回でした（確定オッズでの計算）。']
    open(REPORT, 'w').write('\n'.join(L) + '\n')

def settle():
    log = pd.read_csv(LOG, dtype={'date': str}) if os.path.exists(LOG) else pd.DataFrame()
    done = set(log['date']) if len(log) else set()
    for f in sorted(os.listdir(OUT)):
        m = re.match(r'(\d{8})_買い目\.csv', f)
        if not m or m.group(1) in done:
            continue
        d = m.group(1)
        if d == TODAY and NOW.hour < 17:
            continue
        w5 = win5_page(d)
        if not w5 or len(w5['race_ids']) < 5 or (pd.isna(w5['payout']) and w5['hit_votes'] != 0):
            print(f'{d}：まだ結果が出ていません'); continue
        wins = [race_winner(r) for r in w5['race_ids']]
        if any(w is None for w in wins):
            print(f'{d}：まだ結果が出ていません'); continue
        bets = pd.read_csv(f'{OUT}/{f}')
        key = '-'.join(str(x) for x in wins)
        hit = key in set(bets['組み合わせ'].astype(str))
        legs_used = [set(int(c.split('-')[j]) for c in bets['組み合わせ'].astype(str)) for j in range(5)]
        inn = sum(w in u for w, u in zip(wins, legs_used))
        popmap = {}
        pf = f'{OUT}/{d}_予想.csv'
        if os.path.exists(pf):
            pr = pd.read_csv(pf)
            popmap = {(int(r['レース']), int(r['馬番'])): int(r['人気']) for _, r in pr.iterrows()}
        pops = '-'.join(str(popmap.get((j + 1, w), '?')) for j, w in enumerate(wins))
        row = {'date': d, '勝ち馬': key, '人気': pops, '払戻金': w5['payout'], '点数': len(bets), '的中': hit,
               '払戻（的中時）': w5['payout'] if hit and w5['payout'] == w5['payout'] else 0, '入ったレース': f'{inn}/5'}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        msg = (f'**{d[:4]}-{d[4:6]}-{d[6:]} WIN5の答え合わせ**\n\n勝ち馬 {key}（人気 {pops}）／払戻 '
               + (f'{w5["payout"]:,.0f}円' if pd.notna(w5['payout']) else 'キャリーオーバー')
               + f'\n\n' + ('### ★的中！' if hit else f'はずれ（買い目に入っていた勝ち馬 {inn}/5レース）'))
        print(msg)
        notify(d, msg)
    if len(log):
        log.to_csv(LOG, index=False, encoding='utf-8-sig')
    write_report(log)

# ---------- 予想 ----------
def predict():
    out_csv = f'{OUT}/{TODAY}_買い目.csv'
    if os.path.exists(out_csv):
        print('今日の予想は記録済みです。'); return
    w5 = win5_page(TODAY)
    if not w5 or len(w5['race_ids']) < 5:
        print(f'今日（{TODAY}）はWIN5がありません。'); return
    df0, info0 = shutuba(w5['race_ids'][0])
    if not info0 or not info0['start']:
        print('1レース目の発走時刻が読めませんでした。'); return
    st = info0['start']
    deadline = NOW.replace(hour=st[0], minute=st[1], second=0, microsecond=0) - timedelta(minutes=5)
    target = deadline - timedelta(minutes=12)
    wait = (target - datetime.now(JST)).total_seconds()
    print(f'締切 {deadline:%H:%M}。予想は {target:%H:%M} ごろに行います。')
    if wait > 0:
        print(f'{wait / 60:.0f}分待ちます...'); time.sleep(wait)
    if datetime.now(JST) > deadline - timedelta(minutes=2):
        print('締切に間に合わないので、今日は記録しません。'); return

    M = json.load(open(f'{DRV}/w5_model.json'))
    MF, TH = M['MF'], np.array(M['TH'])
    MED, MU, SD = pd.Series(M['MED'], dtype=float), pd.Series(M['MU']), pd.Series(M['SD'])
    BETA, SLO, SHI, K = np.array(M['BETA']), M['SLO'], M['SHI'], M['K']
    JK = pd.read_csv(f'{DRV}/w5_jockey.csv').set_index('jockey')
    TK = pd.read_csv(f'{DRV}/w5_trainer.csv').set_index('trainer')
    HK = pd.read_csv(f'{DRV}/w5_horse.csv', dtype={'horse_id': str, 'last_date': str}).set_index('horse_id')
    today_dt = pd.Timestamp(datetime.now(JST).date())

    def prior(tab, v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v not in tab.index:
            return 0, 0.08 * K / K, 0.0
        r = tab.loc[v]
        return r['n'], (r['w'] + K * 0.08) / (r['n'] + K), (r['w'] - r['ex']) / (r['n'] + K)

    FEATS = []
    for rid in w5['race_ids']:
        df, info = shutuba(rid)
        if df is None or df.empty:
            print(f'出馬表が読めませんでした（{rid}）'); return
        od = odds_now(rid)
        df['odds'] = df['num'].map(od)
        df = df[df['odds'].notna() & (df['odds'] > 0) & (df['odds'] < 999)].copy()
        if len(df) < 3:
            print(f'オッズが読めませんでした（{rid}）'); return
        n = len(df)
        df['外枠度'] = df['num'] / n
        df['牝'], df['セ'] = (df['sex'] == '牝').astype(int), (df['sex'] == 'セ').astype(int)
        df['斤量差'] = df['kin'] - df['kin'].mean()
        for tab, key, pre in [(JK, 'jockey', 'j'), (TK, 'trainer', 't'), (HK, 'horse_id', 'h')]:
            vals = [prior(tab, v) for v in df[key]]
            df[pre + '_n'] = [np.log1p(v[0]) for v in vals]
            df[pre + '_wr'] = [v[1] for v in vals]
            df[pre + '_excess'] = [v[2] for v in vals]
        hl = []
        for hid in df['horse_id']:
            if hid and hid in HK.index:
                r = HK.loc[hid]
                hl.append((1, r['last_rank'], r['last_pop'], (today_dt - pd.to_datetime(str(r['last_date'])[:8], format='%Y%m%d')).days))
            else:
                hl.append((0, 0, 0, 0))
        df['前回あり'], df['h_last_rank'], df['h_last_pop'], df['h_days'] = zip(*hl)
        rows = []
        for hid in df['horse_id']:
            pr = horse_hist(hid, today_dt) if hid else None
            o = {}
            if pr is not None:
                o['通算出走'] = np.log1p(len(pr))
                if len(pr):
                    l5, last = pr.tail(5), pr.iloc[-1]
                    same = pr[(pr['surf'] == info['surf']) & ((pr['dist'] - info['dist']).abs() <= 200)]
                    o.update({'通算勝率': (pr['rank'] == 1).mean(), '前走着順': last['rank'], '前走人気': last['pop'],
                              '前走着差': last['margin'], '前走上り': last['agari'], '間隔日': (today_dt - last['dt']).days,
                              '距離変化': info['dist'] - last['dist'], '5走平均着差': l5['margin'].mean(),
                              '5走最小着差': l5['margin'].min(), '5走着順率': (l5['rank'] / l5['field']).mean(),
                              '5走複勝率': (l5['rank'] <= 3).mean(), '5走人気以上': (l5['pop'] - l5['rank']).mean(),
                              '同条件出走': len(same),
                              '同条件複勝率': (same['rank'] <= 3).mean() if len(same) else np.nan,
                              '同条件平均着差': same['margin'].mean() if len(same) else np.nan})
            rows.append(o)
        pf = pd.DataFrame(rows, index=df.index)
        for c in pf.columns:
            df[c] = pf[c]
        for c in ['5走平均着差', '前走着差', '同条件平均着差', '前走上り']:
            df[c + '_順位'] = df[c].rank(pct=True) if c in df else np.nan
        df['過去成績あり'] = df['通算出走'].fillna(0).gt(0).astype(int) if '通算出走' in df else 0
        pm = (1 / df['odds']) / (1 / df['odds']).sum()
        xf = df.reindex(columns=MF).astype(float).fillna(MED).fillna(0)
        z = ((xf - MU[MF]) / SD[MF]).clip(-5, 5).values
        s = TH[0] * np.log(pm.values) + z @ TH[1:]
        pn = np.exp(s - s.max())
        df['オッズの勝率'], df['予想勝率'] = pm.values, pn / pn.sum()
        df = df.sort_values('odds').reset_index(drop=True)
        df['人気'] = np.arange(1, len(df) + 1)
        FEATS.append((df, info))
        print(f'  {info["name"]}：{len(df)}頭')

    # 買い方E
    def combo(arrs):
        shp = [len(a) for a in arrs]
        out = np.zeros(shp)
        for j in range(5):
            sh = [1] * 5; sh[j] = shp[j]
            out = out + arrs[j].reshape(sh)
        return out.ravel(), shp
    L = [df[df['人気'] <= MAXPOP] for df, _ in FEATS]
    OPEN = set(np.argsort([l['予想勝率'].max() for l in L])[:2].tolist())
    LP, shp = combo([np.log(l['予想勝率'].values) for l in L])
    LO, _ = combo([np.log(l['odds'].values) for l in L])
    OK, _ = combo([np.where(((l['人気'].values <= 2) | (l['人気'].values >= 6)) if j in OPEN else (l['人気'].values <= 5), 0.0, -1e9)
                   for j, l in enumerate(L)])
    prob = np.where(OK > -1, np.exp(LP), 0.0)
    ph = np.exp(np.polyval(BETA, np.clip(LO, SLO, SHI)))
    ev = prob * ph
    ok = np.where(prob > 0)[0]
    k = min(3 * POINTS, len(ok))
    cand = ok[np.argpartition(-prob[ok], k - 1)[:k]]
    sel = cand[np.argsort(-ev[cand])][:POINTS]
    idx = np.array(np.unravel_index(sel, shp)).T
    nums = [[int(L[j]['num'].values[i]) for j, i in enumerate(row)] for row in idx]
    bets = pd.DataFrame({'組み合わせ': ['-'.join(map(str, n)) for n in nums],
                         '人気': ['-'.join(str(int(L[j]['人気'].values[i])) for j, i in enumerate(row)) for row in idx],
                         '当たる確率%': (prob[sel] * 100).round(3), '予想払戻(円)': ph[sel].round(-3).astype(int)})
    bets = bets.sort_values('当たる確率%', ascending=False).reset_index(drop=True)
    bets['作成時刻'] = datetime.now(JST).strftime('%Y-%m-%d %H:%M')
    bets['買い方'] = 'E'
    bets.to_csv(out_csv, index=False, encoding='utf-8-sig')
    pr = pd.concat([df.assign(レース=j + 1)[['レース', 'num', 'name', '人気', 'odds', 'オッズの勝率', '予想勝率']]
                    for j, (df, _) in enumerate(FEATS)]).rename(columns={'num': '馬番', 'name': '馬名', 'odds': '単勝'})
    pr.to_csv(f'{OUT}/{TODAY}_予想.csv', index=False, encoding='utf-8-sig')

    # 通知・表示
    T = [f'**{TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:]} WIN5の予想**（{datetime.now(JST):%H:%M}のオッズ・締切{deadline:%H:%M}）', '',
         f'5レースとも買い目に入る確率（予想）{prob[sel].sum() * 100:.1f}%／予想回収率 {ev[sel].sum() / (len(sel) * 100) * 100:.0f}%', '']
    for j, (df, info) in enumerate(FEATS):
        d = df.sort_values('予想勝率', ascending=False).head(5)
        vc = pd.Series([n[j] for n in nums]).value_counts()
        popmap = dict(zip(df['num'], df['人気']))
        T.append(f'**{j + 1}レース目 {info["name"]}**' + ('（荒れそう：人気＋穴）' if j in OPEN else ''))
        T.append('予想：' + '　'.join(f'{m}{int(r.num)}番{r["name"]}（{int(r["人気"])}人気・{r["予想勝率"] * 100:.0f}%）'
                                    for m, (_, r) in zip(['◎', '○', '▲', '△', '△'], d.iterrows())))
        T.append('買い目で使う馬：' + '　'.join(f'{a}番({popmap.get(a)}人気){b}点' for a, b in vc.items()))
        T.append('')
    T.append('当たりやすい上位10点：' + '、'.join(bets['組み合わせ'].head(10)))
    T.append('\n※全200点は data/win5 に保存。お金はかけない記録です。')
    msg = '\n'.join(T)
    print(msg)
    notify(TODAY, msg)

if MODE == 'predict':
    predict()
    settle()
elif MODE == 'settle':
    settle()
else:
    print('使い方：python win5_auto.py predict｜settle')
