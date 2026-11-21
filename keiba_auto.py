# =====================================================================
#  競馬予想システム：①〜④まとめて 自動予想・記録・答え合わせ（GitHub Actions用）  版：au5a（2026-11-22）
#   python keiba_auto.py prepare … 前の日までの答え合わせ → 今日の出馬表と材料を作る（朝）
#   python keiba_auto.py live HH:MM … 止める時刻まで動き続け、各レースの発走10分前に最新の単勝オッズで予想して記録
#   python keiba_auto.py predict … 発走の2〜25分前のレースを予想して記録（1回だけ）
#   python keiba_auto.py settle  … 終わったレースの結果を集めて、答え合わせと成績表の更新（夕方）
#  マナー：netkeibaにはログインしない／待ち時間3秒は短くしない／集めたデータは公開・共有しない（リポジトリは非公開）
# =====================================================================
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else 'predict'
# %% セル1：設定
REGION = 'JRA'
import os, re, time, math, json, itertools, unicodedata
import datetime as dt
import pandas as pd
PLACES = {f'{i:02d}' for i in range(1, 11)}
PLACE_NAMES = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', '06': '中山',
               '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive'
except ImportError:                       # Colab以外（動作確認用）
    BASE = os.environ.get('KEIBA_BASE', '.')
MAIN_DIR = f'{BASE}/keiba_system'
HIST_DIR = f'{BASE}/keiba_history'
S5_DIR = f'{HIST_DIR}/s5_tool'
os.makedirs(S5_DIR, exist_ok=True)
LOG_CSV = f'{S5_DIR}/all_log.csv'
BIG_JSON = f'{S5_DIR}/all_params.json'
BIG_MODEL = f'{S5_DIR}/all_are50.txt'
LO, NA, NB, QCUT = 3000, 20, 15, 0.30   # 予想倍率の下限・Aの点数・Bの点数・上位30%
SPLIT = pd.Timestamp('2024-07-01')
EV_TH = 0.9      # ワイドの基準
EV3_TH = 1.0     # 3連複の基準
STAKE_W, STAKE_T = 1000, 100   # 1組の金額（ワイド・3連複）
JST = dt.timezone(dt.timedelta(hours=9))
TODAY = dt.datetime.now(JST).date()
WAIT = 3.0
HEADERS = {'User-Agent': 'Mozilla/5.0 (personal research, low rate)'}

def eligible_mask(d):
    return (d['surface_code'] != 3) & (d['n_runners'] >= 10) & (d['class_code'] != 1)
EXCLUDE_NOTE = '新馬戦・障害・10頭未満を除外'
print(f'①〜④まとめて 自動予想（中央競馬）　版：au5a　モード：{MODE}')
NEW_DIR = f'{BASE}/new'; STATE_DIR = f'{BASE}/state'
_DRV = os.environ.get('DRIVE_DIR')          # GitHub Actions：Googleドライブから取ってきたデータを、決まった場所に置く
if _DRV and os.path.isdir(_DRV):
    import shutil
    _MAP = {'ks_': f'{BASE}/keiba_system/', 'kh_': f'{BASE}/keiba_history/', 's5_': f'{BASE}/keiba_history/s5_tool/'}
    for _f in os.listdir(_DRV):
        for _p, _d in _MAP.items():
            if _f.startswith(_p):
                os.makedirs(_d, exist_ok=True); _dst = _d + _f[len(_p):]
                if not (_p == 's5_' and os.path.exists(_dst)):       # GitHubに保存済みのモデルを優先
                    shutil.copy(f'{_DRV}/{_f}', _dst)
    print('ドライブのデータを置きました')
os.makedirs(NEW_DIR, exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)
def _need(*fs):
    miss = [f for f in fs if not os.path.exists(f)]
    if miss and MODE == 'predict':
        raise SystemExit('準備がまだできていません（朝の prepare が終わっていない）：' + '、'.join(os.path.basename(m) for m in miss))



# %% セル2：通信と文字の読み取り（関数だけ）
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

class BlockedError(Exception):
    pass

def get(url):
    time.sleep(WAIT)
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code in (400, 403, 429):
        raise BlockedError(f'アクセス制限の可能性：{r.status_code}（24時間空けて再開してください）')
    if r.status_code != 200:
        return None
    r.encoding = r.apparent_encoding
    return r.text

def to_num(x):
    try:
        return float(str(x).replace(',', ''))
    except (ValueError, TypeError):
        return np.nan

def time_to_sec(s):
    m = re.match(r'(\d+):(\d+(?:\.\d+)?)', str(s))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return to_num(s)

def nfkc(s):
    return unicodedata.normalize('NFKC', str(s))

SURFACE = {'芝': 1, 'ダ': 2, 'ダート': 2, '障': 3}
GOING = {'良': 1, '稍重': 2, '稍': 2, '重': 3, '不良': 4, '不': 4}

def class_code(text):
    """クラスを数字に（大きいほど上のクラス）。新馬は1"""
    t = nfkc(text)
    if REGION == 'JRA':
        m = re.search(r'G(III|II|I|3|2|1)(?![A-Za-z0-9])', t)
        if m:
            return {'I': 9, '1': 9, 'II': 8, '2': 8, 'III': 7, '3': 7}[m.group(1)]
        if 'オープン' in t or '(L)' in t or 'OP' in t: return 6
        if '3勝' in t or '1600万' in t: return 5
        if '2勝' in t or '1000万' in t: return 4
        if '1勝' in t or '500万' in t: return 3
        if '未勝利' in t: return 2
        if '新馬' in t: return 1
        return np.nan
    if re.search(r'Jpn|重賞|\bS[1-3I]|G(III|II|I|[1-3])(?![A-Za-z0-9])', t): return 9
    m = re.search(r'([ABC])([1-3])', t)
    if m:
        return {'A1': 8, 'A2': 7, 'B1': 6, 'B2': 5, 'B3': 4.5, 'C1': 4, 'C2': 3, 'C3': 2.5}[m.group(1) + m.group(2)]
    if 'オープン' in t or 'OP' in t: return 8
    if '新馬' in t: return 1
    return np.nan


# %% セル3：レース結果ページの読み取り（関数だけ）
def race_ids_on(d):
    page = get(f'https://db.netkeiba.com/race/list/{d:%Y%m%d}/')
    if not page:
        return None                         # 読めなかった日は「済み」にしない
    ids = sorted(set(re.findall(r'/race/(\d{12})/', page)))
    return [i for i in ids if i[4:6] in PLACES]

def parse_race_html(race_id, page):
    """結果ページHTML → (各馬のDataFrame, レース情報dict, 払戻のlist)"""
    soup = BeautifulSoup(page, 'html.parser')
    tbl = soup.find('table', class_='race_table_01')
    if tbl is None:
        return None
    header = [th.get_text(strip=True) for th in tbl.find('tr').find_all('th')]

    def col(*names, exact=False):
        for n in names:
            for i, h in enumerate(header):
                if (h == n) if exact else h.startswith(n):
                    return i
        return None

    agari = col('上り', exact=True)
    if agari is None:
        agari = col('上り')
    c = dict(rank=col('着順'), waku=col('枠'), num=col('馬番'), name=col('馬名'),
             sexage=col('性齢'), kin=col('斤量'), jockey=col('騎手'),
             time=col('タイム', exact=True), pos=col('通過'), agari=agari,
             odds=col('単勝'), pop=col('人気'), weight=col('馬体重'), prize=col('賞金'))

    rows = []
    for tr in tbl.find_all('tr')[1:]:
        tds = tr.find_all('td')
        if len(tds) < len(header):
            continue
        def txt(k):
            return tds[c[k]].get_text(strip=True) if c[k] is not None else ''
        a_h = tds[c['name']].find('a', href=re.compile(r'/horse/')) if c['name'] is not None else None
        a_j = tds[c['jockey']].find('a', href=re.compile(r'/jockey/')) if c['jockey'] is not None else None
        m_h = re.search(r'/horse/(\w+)', a_h['href']) if a_h else None
        m_j = re.search(r'/jockey/(?:result/recent/)?(\w+)', a_j['href']) if a_j else None
        sexage = txt('sexage')
        w = re.match(r'(\d+)\(([+-]?\d+)\)', txt('weight'))
        pos = [int(x) for x in re.findall(r'\d+', txt('pos'))]
        rows.append({
            'race_id': race_id,
            'horse_id': m_h.group(1) if m_h else None,
            'horse_name': txt('name'),
            'jockey_id': m_j.group(1) if m_j else None,
            'finish': to_num(txt('rank')),
            'waku': to_num(txt('waku')),
            'umaban': to_num(txt('num')),
            'sex': sexage[:1],
            'age': to_num(sexage[1:]),
            'kinryo': to_num(txt('kin')),
            'time_sec': time_to_sec(txt('time')),
            'pos_first': pos[0] if pos else np.nan,
            'pos_last': pos[-1] if pos else np.nan,
            'agari': to_num(txt('agari')),
            'odds': to_num(txt('odds')),
            'popularity': to_num(txt('pop')),
            'body_weight': float(w.group(1)) if w else np.nan,
            'body_diff': float(w.group(2)) if w else np.nan,
            'prize': to_num(txt('prize')),
        })
    horses = pd.DataFrame(rows)
    if horses.empty:
        return None

    intro = soup.find('div', class_='data_intro')
    itext = intro.get_text(' ', strip=True) if intro else soup.get_text(' ', strip=True)[:3000]
    ttext = soup.title.get_text(' ', strip=True) if soup.title else ''
    m_c = re.search(r'(芝|ダ|障)[^0-9]{0,8}?(\d{3,4})m', itext)
    m_g = re.search(r'(?:芝|ダート)\s*:\s*(良|稍重|稍|重|不良)', itext)
    m_d = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', itext)
    race = {
        'race_id': race_id,
        'date': f'{m_d.group(1)}-{int(m_d.group(2)):02d}-{int(m_d.group(3)):02d}' if m_d else None,
        'place': race_id[4:6],
        'surface_code': SURFACE.get(m_c.group(1)) if m_c else np.nan,
        'distance': to_num(m_c.group(2)) if m_c else np.nan,
        'going_code': GOING.get(m_g.group(1)) if m_g else np.nan,
        'class_code': class_code(ttext + ' ' + itext),
        'handicap': int('ハンデ' in itext),
        'n_runners': int(horses['finish'].notna().sum()),
        'prize1': float(horses['prize'].max()) if horses['prize'].notna().any() else np.nan,
    }

    pays = []
    for pt in soup.find_all('table', class_='pay_table_01'):
        for tr in pt.find_all('tr'):
            th = tr.find('th')
            tds = tr.find_all('td')
            if th is None or len(tds) < 2:
                continue
            kind = th.get_text(strip=True)
            combos = [s for s in tds[0].get_text('\n', strip=True).split('\n') if s]
            amounts = [s for s in tds[1].get_text('\n', strip=True).split('\n') if s]
            for cmb, amt in zip(combos, amounts):
                pays.append({'race_id': race_id, 'kind': kind,
                             'combo': '-'.join(re.findall(r'\d+', cmb)),
                             'payout': to_num(amt)})
    return horses.drop(columns=['prize']), race, pays

def parse_race(race_id):
    page = get(f'https://db.netkeiba.com/race/{race_id}/')
    if not page:
        return None
    return parse_race_html(race_id, page)


# %% セル5：特徴量と確率計算（関数だけ）
FEATURES = [
    'waku', 'umaban', 'age', 'kinryo', 'body_weight', 'body_diff', 'sex_code',
    'surface_code', 'distance', 'going_code', 'class_code', 'handicap', 'n_runners', 'prize1',
    'prev1_finish', 'prev2_finish', 'prev3_finish', 'mean_finish_3',
    'prev1_time_diff', 'prev1_agari_rank', 'prev1_pos_last_ratio',
    'days_since', 'n_starts', 'career_win_rate', 'career_top3_rate',
    'dist_change', 'class_change', 'same_surface_top3_rate', 'heavy_top3_rate',
    'jockey_top3_rate', 'rank_mean_finish_3', 'rank_prev1_time_diff',
]

FEATURES_V3 = FEATURES + [
    'prev1_pos_first_ratio', 'style_early', 'rank_style', 'n_front', 'front_share',   # 脚質と展開
    'prev1_speed', 'max_speed_5', 'mean_speed_3', 'rank_max_speed_5',               # 自作タイム指数
    'course_top3_rate', 'course_starts',                                            # コース適性
    'jockey_change',                                                                # 騎手の乗り替わり
]

def shifted_mean(df, keys, col):
    """同じグループの『過去だけ』の平均（今回のレースは含まない）"""
    return df.groupby(keys, sort=False)[col].transform(lambda s: s.shift(1).expanding().mean())

def shifted_mean_fast(df, keys, col):
    """shifted_mean と同じ値を、グループが多くても速く計算する"""
    v = df[col]
    t = pd.DataFrame({'f': v.fillna(0), 'c': v.notna().astype(int)})
    cs = t.groupby([df[k] for k in keys], sort=False).cumsum()
    s, c = cs['f'] - t['f'], cs['c'] - t['c']
    return (s / c).where(c > 0)

def add_v3_features(df):
    """v3で追加する特徴。すべて『そのレースより前』の情報だけで作る"""
    g = df.groupby('horse_id', sort=False)
    # 脚質：序盤の位置（1コーナーの順位 ÷ 頭数）。小さいほど前に行く馬
    df['pos_first_ratio'] = df['pos_first'] / df['n_runners']
    df['prev1_pos_first_ratio'] = g['pos_first_ratio'].shift(1)
    df['style_early'] = shifted_mean_fast(df, ['horse_id'], 'pos_first_ratio')
    df['rank_style'] = df.groupby('race_id')['style_early'].rank(method='min', na_option='bottom')
    # 展開：このレースで前に行きそうな馬の数
    front = (df['style_early'] <= 0.2).astype(int)
    df['n_front'] = front.groupby(df['race_id']).transform('sum')
    df['front_share'] = df['n_front'] / df['n_runners']
    # 自作タイム指数：同じ競馬場・芝ダ・距離・馬場（良〜稍／重〜不良）の、それまでの勝ちタイム平均との差
    df['going_grp'] = (df['going_code'] >= 3).astype(int)
    key = ['place', 'surface_code', 'distance', 'going_grp']
    w = (df[df['finish'] == 1].drop_duplicates('race_id')[['race_id', 'date'] + key + ['time_sec']]
           .sort_values(['date', 'race_id']))
    w['base'] = shifted_mean_fast(w, key, 'time_sec')
    w['base_n'] = w.groupby(key).cumcount()
    w.loc[w['base_n'] < 5, 'base'] = np.nan
    base = w.set_index('race_id')['base']
    df['speed'] = (df['race_id'].map(base) - df['time_sec']) / df['distance'] * 1000
    sp = [g['speed'].shift(k) for k in range(1, 6)]
    df['prev1_speed'] = sp[0]
    df['max_speed_5'] = pd.concat(sp, axis=1).max(axis=1)
    df['mean_speed_3'] = pd.concat(sp[:3], axis=1).mean(axis=1)
    df['rank_max_speed_5'] = df.groupby('race_id')['max_speed_5'].rank(method='min', ascending=False, na_option='bottom')
    # コース適性：同じ競馬場・芝ダ・距離での過去の3着内率と出走回数
    ck = ['horse_id', 'place', 'surface_code', 'distance']
    df['course_top3_rate'] = shifted_mean_fast(df, ck, 'top3')
    df['course_starts'] = df.groupby(ck, sort=False).cumcount()
    # 騎手の乗り替わり
    prev_j = g['jockey_id'].shift(1)
    df['jockey_change'] = np.where(prev_j.isna(), np.nan, (prev_j != df['jockey_id']).astype(float))
    return df

def build_features(res, races):
    df = res.merge(races, on='race_id', how='inner')
    if 'is_entry' not in df.columns:
        df['is_entry'] = False
    df['is_entry'] = df['is_entry'].fillna(False).astype(bool)
    # 出走取消・競走除外（着順もオッズもない）は、出走しなかったものとして除く
    df = df[~(df['finish'].isna() & df['odds'].isna() & ~df['is_entry'])]
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date', 'horse_id']).sort_values(['date', 'race_id', 'umaban']).reset_index(drop=True)

    df['win'] = np.where(df['finish'].notna(), (df['finish'] == 1).astype(float), np.nan)
    df['top3'] = np.where(df['finish'].notna(), (df['finish'] <= 3).astype(float), np.nan)
    df['sex_code'] = df['sex'].map({'牡': 1, '牝': 2, 'セ': 3})

    win_time = df[df['finish'] == 1].groupby('race_id')['time_sec'].min()
    df['time_diff'] = df['time_sec'] - df['race_id'].map(win_time)
    df['agari_rank'] = df.groupby('race_id')['agari'].rank(method='min')
    df['pos_last_ratio'] = df['pos_last'] / df['n_runners']

    g = df.groupby('horse_id', sort=False)
    for col in ['finish', 'time_diff', 'agari_rank', 'pos_last_ratio', 'distance', 'class_code']:
        df[f'prev1_{col}'] = g[col].shift(1)
    df['prev2_finish'] = g['finish'].shift(2)
    df['prev3_finish'] = g['finish'].shift(3)
    df['mean_finish_3'] = df[['prev1_finish', 'prev2_finish', 'prev3_finish']].mean(axis=1)
    df['days_since'] = (df['date'] - g['date'].shift(1)).dt.days
    df['n_starts'] = g.cumcount()
    df['career_win_rate'] = shifted_mean(df, 'horse_id', 'win')
    df['career_top3_rate'] = shifted_mean(df, 'horse_id', 'top3')
    df['dist_change'] = df['distance'] - df['prev1_distance']
    df['class_change'] = df['class_code'] - df['prev1_class_code']
    df['same_surface_top3_rate'] = shifted_mean(df, ['horse_id', 'surface_code'], 'top3')
    df['heavy'] = (df['going_code'] >= 3).astype(int)
    df['heavy_top3_rate'] = shifted_mean(df, ['horse_id', 'heavy'], 'top3')

    jd = (df.groupby(['jockey_id', 'date'])['top3'].agg(['sum', 'count'])
            .reset_index().sort_values(['jockey_id', 'date']))
    jd['cs'] = jd.groupby('jockey_id')['sum'].cumsum() - jd['sum']
    jd['cc'] = jd.groupby('jockey_id')['count'].cumsum() - jd['count']
    jd['jockey_top3_rate'] = (jd['cs'] + 3) / (jd['cc'] + 10)
    df = df.merge(jd[['jockey_id', 'date', 'jockey_top3_rate']], on=['jockey_id', 'date'], how='left')

    df['rank_mean_finish_3'] = df.groupby('race_id')['mean_finish_3'].rank(method='min', na_option='bottom')
    df['rank_prev1_time_diff'] = df.groupby('race_id')['prev1_time_diff'].rank(method='min', na_option='bottom')
    df = df.sort_values(['date', 'race_id', 'umaban']).reset_index(drop=True)
    df = add_v3_features(df)
    df['eligible'] = eligible_mask(df)
    return df

# ---- 3着までの並びの確率（ハービル式） ----
_PERMS = {}
def _perms(n):
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n), 3)), dtype=int).reshape(-1, 3)
    return _PERMS[n]

def harville(p):
    """各馬の勝率 → すべての(1着,2着,3着)の並びとその確率"""
    p = np.asarray(p, float)
    p = p / p.sum()
    P = _perms(len(p))
    a, b, c = p[P[:, 0]], p[P[:, 1]], p[P[:, 2]]
    pr = a * b / np.clip(1 - a, 1e-12, None) * c / np.clip(1 - a - b, 1e-12, None)
    return P, pr

def top3_probs(p):
    """各馬が3着以内に入る確率"""
    P, pr = harville(p)
    out = np.zeros(len(p))
    for j in range(3):
        np.add.at(out, P[:, j], pr)
    return out

def trio_probs(p):
    """3連複の各組み合わせ（馬の位置の昇順）と確率"""
    P, pr = harville(p)
    key = np.sort(P, axis=1)
    n = len(p)
    code = key[:, 0] * n * n + key[:, 1] * n + key[:, 2]
    uk, inv = np.unique(code, return_inverse=True)
    s = np.zeros(len(uk))
    np.add.at(s, inv, pr)
    combos = np.stack([uk // (n * n), (uk // n) % n, uk % n], axis=1)
    return combos, s

def softmax_by_race(score, race_ids):
    s = pd.Series(score)
    s = s - s.groupby(race_ids).transform('max').values
    e = np.exp(s.values)
    return e / pd.Series(e).groupby(race_ids).transform('sum').values


# %% セル6：計算の関数（検証と同じもの）
import lightgbm as lgb

PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=42)

def predict_lgb(m, X):
    it = getattr(m, 'best_iteration', 0)
    return m.predict(X, num_iteration=it if it and it > 0 else None)

def fit_lgb(data, target, features, valid_start, test_start):
    data = data[data[target].notna()]
    tr = data[data['date'] < valid_start]
    va = data[(data['date'] >= valid_start) & (data['date'] < test_start)]
    return lgb.train(PARAMS, lgb.Dataset(tr[features], tr[target]), num_boost_round=3000,
                     valid_sets=[lgb.Dataset(va[features], va[target])],
                     callbacks=[lgb.early_stopping(100, verbose=False)])

def add_market(df):
    df = df.copy()
    inv = 1 / df['odds']
    df['p_market'] = inv / inv.groupby(df['race_id']).transform('sum')
    return df

MKT_FEATURES = ['log_odds', 'popularity_rank', 'pop_ratio', 'p_market', 'mkt_top3',
                'race_fav_odds', 'race_odds_ratio', 'race_entropy']

def add_mkt_features(df):
    """オッズから作る材料（市場の見方）"""
    df = add_market(df)
    df['log_odds'] = np.log(df['odds'])
    df['popularity_rank'] = df.groupby('race_id')['odds'].rank(method='first')
    df['pop_ratio'] = df['popularity_rank'] / df['n_runners']
    fav = df.loc[df['popularity_rank'] == 1].set_index('race_id')['odds']
    sec = df.loc[df['popularity_rank'] == 2].set_index('race_id')['odds']
    df['race_fav_odds'] = df['race_id'].map(fav)
    df['race_odds_ratio'] = df['race_id'].map(sec) / df['race_fav_odds']
    pm = df['p_market'].clip(1e-9)
    df['race_entropy'] = (-(pm * np.log(pm))).groupby(df['race_id']).transform('sum')
    df['mkt_top3'] = np.nan
    ok = df['p_market'].notna().groupby(df['race_id']).transform('all')
    for rid, idx in df[ok].groupby('race_id').groups.items():
        df.loc[idx, 'mkt_top3'] = top3_probs(df.loc[idx, 'p_market'].values)
    return df

def rate_before(df, keys, target, prior_n):
    t = df[keys + ['date']].copy()
    t['_v'] = df[target].fillna(0).values
    t['_c'] = df[target].notna().astype(int).values
    agg = (t.groupby(keys + ['date'], sort=False)[['_v', '_c']].sum().reset_index().sort_values(keys + ['date']))
    gb = agg.groupby(keys, sort=False)
    agg['_cs'] = gb['_v'].cumsum() - agg['_v']
    agg['_cc'] = gb['_c'].cumsum() - agg['_c']
    agg['_r'] = agg['_cs'] / (agg['_cc'] + prior_n)
    out = t[keys + ['date']].merge(agg[keys + ['date', '_r', '_cc']], on=keys + ['date'], how='left')
    return out['_r'].values, out['_cc'].values

def add_v4_features(df):
    df = df.copy()
    df['top3_ex'] = df['top3'] - 3 / df['n_runners']
    df['jockey_place_ex'], df['jockey_place_n'] = rate_before(df, ['jockey_id', 'place'], 'top3_ex', 20)
    df['jockey_course_ex'], _ = rate_before(df, ['jockey_id', 'place', 'distance'], 'top3_ex', 20)
    df['jockey_horse_ex'], df['jockey_horse_n'] = rate_before(df, ['jockey_id', 'horse_id'], 'top3_ex', 3)
    df['waku_bias'], _ = rate_before(df, ['place', 'distance', 'waku'], 'top3_ex', 50)
    df['umaban_ratio'] = df['umaban'] / df['n_runners']
    df['race_level'] = df.groupby('race_id')['career_top3_rate'].transform('mean')
    df['fin_ratio'] = df['finish'] / df['n_runners']
    g = df.groupby('horse_id', sort=False)
    df['prev1_race_level'] = g['race_level'].shift(1)
    df['level_change'] = df['race_level'] - df['prev1_race_level']
    f = [g['fin_ratio'].shift(k) for k in (1, 2, 3)]
    df['form_trend'] = f[0] - pd.concat(f[1:], axis=1).mean(axis=1)
    s = [g['speed'].shift(k) for k in (1, 2, 3)]
    df['speed_trend'] = s[0] - pd.concat(s[1:], axis=1).mean(axis=1)
    b = [g['body_weight'].shift(k) for k in (1, 2, 3)]
    df['prev_bw_mean'] = pd.concat(b, axis=1).mean(axis=1)
    df['weight_vs_avg'] = df['body_weight'] - df['prev_bw_mean']
    df['prev1_pop_gap'] = g['popularity'].shift(1) - g['finish'].shift(1)
    return df

V4_NEW = ['jockey_place_ex', 'jockey_place_n', 'jockey_course_ex', 'jockey_horse_ex', 'jockey_horse_n',
          'waku_bias', 'umaban_ratio', 'race_level', 'prev1_race_level', 'level_change',
          'form_trend', 'speed_trend', 'weight_vs_avg', 'prev1_pop_gap']
U_FEATS = FEATURES_V3 + V4_NEW + MKT_FEATURES


def wide_dict():
    d = {}
    x = PAY[PAY['kind'].astype(str).map(nfkc).str.strip() == 'ワイド']
    for rid, c, p in zip(x['race_id'], x['combo'], x['payout']):
        nums = tuple(sorted(int(v) for v in re.findall(r'\d+', str(c))))
        if len(nums) == 2 and p == p:
            d.setdefault(rid, {})[nums] = float(p)
    return d

def pair_mats(p):
    """勝つ確率 → (馬連の確率の表, ワイドの確率の表)。表は n×n で、i<j の所を使う"""
    P, pr = harville(p)
    n = len(p)
    U = np.zeros((n, n)); W = np.zeros((n, n))
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    np.add.at(U, (lo, hi), pr)
    for x, y in [(a, b), (a, c), (b, c)]:
        np.add.at(W, (np.minimum(x, y), np.maximum(x, y)), pr)
    return U, W

def fit_k(d):
    """オッズからの確率と実際の払戻の関係（学習期間だけで決める）"""
    ku, kw = [], []
    for rid, g in d.groupby('race_id'):
        g = g[g['odds'].notna() & g['umaban'].notna() & g['p_market'].notna()]
        if rid not in WDICT or rid not in UDICT or len(g) < 6:
            continue
        nums = g['umaban'].astype(int).values
        U, W = pair_mats(g['p_market'].values)
        idx = {u: i for i, u in enumerate(nums)}
        for t, amt in UDICT[rid].items():
            if t[0] in idx and t[1] in idx:
                i, j = sorted((idx[t[0]], idx[t[1]]))
                ku.append(amt / 100 * U[i, j])
        for t, amt in WDICT[rid].items():
            if t[0] in idx and t[1] in idx:
                i, j = sorted((idx[t[0]], idx[t[1]]))
                kw.append(amt / 100 * W[i, j])
    return (float(np.median(ku)) if ku else 0.75), (float(np.median(kw)) if kw else 0.75)



def fit_kw(d):
    """オッズからのワイドの見込み確率と実際の払戻の関係（学習期間だけで決める）"""
    WD = wide_dict()
    k = []
    for rid, g in d.groupby('race_id'):
        g = g[g['odds'].notna() & g['umaban'].notna() & g['p_market'].notna()]
        if rid not in WD or len(g) < 6:
            continue
        nums = g['umaban'].astype(int).values
        _, W = pair_mats(g['p_market'].values)
        idx = {u: i for i, u in enumerate(nums)}
        for t, amt in WD[rid].items():
            if t[0] in idx and t[1] in idx:
                i, j = sorted((idx[t[0]], idx[t[1]]))
                k.append(amt / 100 * W[i, j])
    return float(np.median(k)) if k else 0.75


# %% セルA：これまでのデータを読み込む（いつものデータ＋過去データ）
def _csv(path):
    return path if os.path.exists(path) else (path + '.gz' if os.path.exists(path + '.gz') else None)
def load_dir(d):
    p = {k: _csv(f'{d}/{k}.csv') for k in ['results', 'races', 'payouts']}
    if not all(p.values()):
        return None
    res = pd.read_csv(p['results'], dtype={'race_id': str, 'horse_id': str, 'jockey_id': str}, low_memory=False)
    races = pd.read_csv(p['races'], dtype={'race_id': str, 'place': str})
    pay = pd.read_csv(p['payouts'], dtype={'race_id': str, 'combo': str})
    return res, races, pay

def load_new():
    import glob
    out = []
    for k in ['results', 'races', 'payouts']:
        fs = sorted(glob.glob(f'{NEW_DIR}/{k}_*.csv'))
        out.append(pd.concat([pd.read_csv(f, dtype={'race_id': str, 'horse_id': str, 'jockey_id': str, 'place': str, 'combo': str}) for f in fs], ignore_index=True) if fs else None)
    return tuple(out) if all(x is not None for x in out) else None
parts = [x for x in (load_dir(MAIN_DIR), load_dir(HIST_DIR), load_new()) if x is not None]
if not parts:
    raise SystemExit('データが見つかりません。')
RES = pd.concat([p[0] for p in parts], ignore_index=True).drop_duplicates(['race_id', 'umaban'], keep='last')
RACES = pd.concat([p[1] for p in parts], ignore_index=True).drop_duplicates('race_id', keep='last')
PAY = pd.concat([p[2] for p in parts], ignore_index=True).drop_duplicates()
if 'prize1' not in RACES.columns:
    RACES['prize1'] = np.nan
RACES['place'] = RACES['race_id'].str[4:6]
RACES = RACES[RACES['place'].isin(PLACES)]
print(f'これまでのデータ：{len(RACES):,}レース（〜{pd.to_datetime(RACES["date"]).max().date()}）')


_need(f'{S5_DIR}/info.json', f'{S5_DIR}/m_win.txt', f'{S5_DIR}/m1_blend.json', BIG_JSON, BIG_MODEL)


# %% セルB：S5のモデル（初回だけ作って凍結。2回目からは読み込むだけ）
def train_s5():
    cache = f'{HIST_DIR}/wide2_df.pkl'
    DF = None
    if os.path.exists(cache):
        try:
            c = pd.read_pickle(cache)
            if c.attrs.get('n_races') == len(RACES):
                DF = c
                print('  保存済みの材料を使います')
        except Exception:
            DF = None
    if DF is None:
        print('  材料を作っています（10〜20分）…')
        DF = add_mkt_features(add_v4_features(build_features(RES, RACES)).query('eligible').copy())
    end = DF['date'].max()
    vs = end - pd.DateOffset(months=6)
    m = fit_lgb(DF, 'win', U_FEATS, vs, end + pd.Timedelta(days=1))
    kw = fit_kw(DF[DF['date'] < vs])
    os.makedirs(S5_DIR, exist_ok=True)
    m.save_model(f'{S5_DIR}/m_win.txt', num_iteration=m.best_iteration or None)
    info = {'freeze_date': str(end.date()), 'k_w': kw, 'rule': 'S5：ワイドの確率上位10組のうち期待値1.0以上を最大3組・1組1,000円・1番人気2.0倍以上'}
    json.dump(info, open(f'{S5_DIR}/info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return m, info

if os.path.exists(f'{S5_DIR}/info.json'):
    MS5 = lgb.Booster(model_file=f'{S5_DIR}/m_win.txt')
    INFO5 = json.load(open(f'{S5_DIR}/info.json', encoding='utf-8'))
    print(f'S5モデル：凍結済みのものを使います（{INFO5["freeze_date"]}までのデータで学習）')
else:
    print('S5モデル：初回なので、全部のデータで作って凍結します（10〜30分）…')
    MS5, INFO5 = train_s5()
    print(f'S5モデル：凍結しました（{INFO5["freeze_date"]}までのデータで学習）')
KW = INFO5['k_w']


# %% セルB2：M1の混ぜ方（モデルの確率とオッズの確率。初回だけ決めて保存）
from scipy.optimize import minimize
M1_JSON = f'{S5_DIR}/m1_blend.json'

def fit_blend():
    cache = f'{HIST_DIR}/wide2_df.pkl'
    DF = pd.read_pickle(cache) if os.path.exists(cache) else None
    if DF is None:
        print('  材料を作っています（10〜20分）…')
        DF = add_mkt_features(add_v4_features(build_features(RES, RACES)).query('eligible').copy())
    end = pd.Timestamp(INFO5['freeze_date'])
    DF['date'] = pd.to_datetime(DF['date'])
    d = DF[(DF['date'] > end - pd.DateOffset(months=6)) & (DF['date'] <= end) & DF['win'].notna()
           & DF['odds'].notna() & DF['p_market'].notna()].copy()
    raw = MS5.predict(d[U_FEATS])
    d['pw'] = raw / pd.Series(raw, index=d.index).groupby(d['race_id']).transform('sum').values
    lp, lm = np.log(d['pw'].clip(1e-6)).values, np.log(d['p_market'].clip(1e-6)).values
    codes = pd.factorize(d['race_id'])[0]; y = d['win'].values == 1
    def nll(w):
        s_ = w[0] * lp + w[1] * lm
        s_ = s_ - pd.Series(s_).groupby(codes).transform('max').values
        Z = np.bincount(codes, np.exp(s_))
        return -(s_[y].sum() - np.log(Z[codes[y]]).sum()) / max(y.sum(), 1)
    w = minimize(nll, [1.0, 1.0], method='Nelder-Mead').x
    return {'a': float(w[0]), 'b': float(w[1]), 'freeze_date': INFO5['freeze_date'], 'races': int(d['race_id'].nunique())}

def fit_k3():
    """オッズからの3連複の見込み確率と、実際の払戻の関係（モデルを作った日より前の2年で決める）"""
    end = pd.Timestamp(INFO5['freeze_date'])
    rr = RACES[(pd.to_datetime(RACES['date']) > end - pd.DateOffset(years=2)) & (pd.to_datetime(RACES['date']) <= end)]
    x = PAY[PAY['kind'].astype(str).map(nfkc).str.strip().isin(['3連複', '三連複']) & PAY['race_id'].isin(rr['race_id'])]
    tp = {}
    for rid, c, p in zip(x['race_id'], x['combo'], x['payout']):
        nums = tuple(sorted(int(v) for v in re.findall(r'\d+', str(c))))
        if len(nums) == 3 and p == p:
            tp.setdefault(rid, []).append((nums, float(p)))
    res = RES[RES['race_id'].isin(tp) & RES['odds'].notna() & RES['umaban'].notna()]
    k = []
    for rid, g in res.groupby('race_id'):
        if len(g) < 6:
            continue
        nums = g['umaban'].astype(int).values
        inv = 1 / g['odds'].values
        cmb, pr = trio_probs(inv / inv.sum())
        look = {tuple(sorted((nums[a], nums[b], nums[c]))): q for (a, b, c), q in zip(cmb, pr)}
        for key, amt in tp[rid]:
            if key in look:
                k.append(amt / 100 * look[key])
    return float(np.median(k)) if k else 0.73

BL = None
if os.path.exists(M1_JSON):
    BL = json.load(open(M1_JSON, encoding='utf-8'))
    if BL.get('freeze_date') != INFO5['freeze_date']:
        BL = None
if BL is None:
    print('M1の混ぜ方：初回なので、モデルを作った直前の半年のレースで決めます…')
    BL = fit_blend()
if 'k3' not in BL:
    print('3連複の払戻の見込みの係数を決めています（数分）…')
    BL['k3'] = fit_k3()
json.dump(BL, open(M1_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
K3 = BL['k3']
print(f'M1の混ぜ方：モデル {BL["a"]:.2f}・オッズ {BL["b"]:.2f}（{BL["races"]:,}レースで決定）／3連複の係数 {K3:.2f}')

def picks(g):
    """買い目を決める（ワイド：上位10組から期待値0.9以上を最大3組／3連複：上位20組から期待値1.0以上を最大3点）"""
    nums = g['umaban'].astype(int).values
    _, Wm = pair_mats(g['p_market'].values); _, Wd = pair_mats(g['pw'].values)
    iu = np.triu_indices(len(nums), 1); wm, wd = Wm[iu], Wd[iu]
    pool = np.argsort(-wd)[:10]
    ev = wd * KW / np.clip(wm, 1e-12, None)
    W = [(f'{min(nums[iu[0][k]], nums[iu[1][k]])}-{max(nums[iu[0][k]], nums[iu[1][k]])}', float(wd[k]), float(ev[k]))
         for k in pool[np.argsort(-ev[pool])] if ev[k] >= EV_TH][:3]
    cd, pd_ = trio_probs(g['pw'].values); cm, pm_ = trio_probs(g['p_market'].values)
    mk = {tuple(c): q for c, q in zip(cm, pm_)}
    top = np.argsort(-pd_)[:20]
    t = []
    for k in top:
        c = tuple(cd[k]); e3 = pd_[k] * K3 / max(mk.get(c, 1e-12), 1e-12)
        if e3 >= EV3_TH:
            t.append(('-'.join(str(x) for x in sorted(int(nums[i]) for i in c)), float(pd_[k]), float(e3)))
    T = sorted(t, key=lambda z: -z[2])[:3]
    return W, T

def to_m1(pw, pm):
    s_ = BL['a'] * np.log(np.clip(pw, 1e-6, None)) + BL['b'] * np.log(np.clip(pm, 1e-6, None))
    e = np.exp(s_ - s_.max())
    return e / e.sum()


# %% セルB3：3連単の配当の見込みと、荒れるレースのモデル（初回だけ作って保存）
GAM, DEL = 0.85, 0.70
_TP = {}
def tri_prob(p):
    p = np.clip(np.asarray(p, float), 1e-9, None); p = p / p.sum()
    p2 = p ** GAM; p2 = p2 / p2.sum(); p3 = p ** DEL; p3 = p3 / p3.sum()
    n = len(p)
    if n not in _TP:
        _TP[n] = np.array(list(itertools.permutations(range(n), 3)), dtype=int)
    P = _TP[n]; a, b, c = P[:, 0], P[:, 1], P[:, 2]
    pr = p[a] * p2[b] / np.clip(1 - p2[a], 1e-12, None) * p3[c] / np.clip(1 - p3[a] - p3[b], 1e-12, None)
    return P, pr / pr.sum()
def tri_harv(p):
    p = np.clip(np.asarray(p, float), 1e-9, None); p = p / p.sum(); n = len(p)
    if n not in _TP:
        _TP[n] = np.array(list(itertools.permutations(range(n), 3)), dtype=int)
    P = _TP[n]; a, b, c = P[:, 0], P[:, 1], P[:, 2]
    pr = p[a] * p[b] / np.clip(1 - p[a], 1e-12, None) * p[c] / np.clip(1 - p[a] - p[b], 1e-12, None)
    return P, pr / pr.sum()
def trio_tab(P, q, n):
    s_ = np.sort(P, 1); key = s_[:, 0] * n * n + s_[:, 1] * n + s_[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    return uk, np.bincount(inv, q)
def wide_tab(P, q, n):
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    def k_(x, y):
        return np.minimum(x, y) * n + np.maximum(x, y)
    iu = np.triu_indices(n, 1)
    return iu, sum(np.bincount(k_(x, y), q, n * n) for x, y in [(a, b), (a, c), (b, c)])[iu[0] * n + iu[1]]
def est_trio(q):
    return np.exp(np.polyval(BIG['cf3'], np.log(np.clip(q, 1e-9, None)))) * BIG['sm3']
def est_odds(q):
    return np.exp(np.polyval(BIG['cf'], np.log(np.clip(q, 1e-9, None)))) * BIG['smear']
FEAT_COLS = ['n', 'o1', 'o2', 'o3', 'n10', 'ent_k', 'ent_m', 'kl', 'long_m', 'long_k', 'max_ratio', 'n_ratio', 'k10', 'k50', 'm10', 'm50',
             'cls', 'surf', 'dist', 'going', 'hcap', 'place', 'month']
def race_feat(odds, pk, pm, qk, qm, O, info, rid, date):
    pk = pk / pk.sum(); pm = pm / pm.sum()
    pop = pd.Series(odds).rank(method='first').values; long_ = pop >= 6
    ratio = np.where(long_, pm / pk, 0); so = np.sort(odds)
    return dict(n=len(odds), o1=so[0], o2=so[1], o3=so[2], n10=(odds < 10).sum(), ent_k=-(pk * np.log(pk)).sum(), ent_m=-(pm * np.log(pm)).sum(),
                kl=(pm * np.log(pm / pk)).sum(), long_m=pm[long_].sum(), long_k=pk[long_].sum(), max_ratio=ratio.max(),
                n_ratio=(ratio > 1.3).sum(), k10=qk[O >= 1000].sum(), k50=qk[O >= 5000].sum(), m10=qm[O >= 1000].sum(), m50=qm[O >= 5000].sum(),
                cls=info.get('class_code'), surf=info.get('surface_code'), dist=info.get('distance'), going=info.get('going_code'),
                hcap=info.get('handicap'), place=int(rid[4:6]), month=int(str(date)[5:7]))

def build_big():
    pr = pd.read_csv(_csv(f'{HIST_DIR}/pred6.csv'), dtype={'race_id': str, 'fold': str})
    pr = pr[pr['odds'].notna() & pr['umaban'].notna() & pr['M1'].notna() & pr['p_market'].notna()].copy()
    pr['umaban'] = pr['umaban'].astype(int); pr['date'] = pd.to_datetime(pr['date'])
    k_ = PAY['kind'].astype(str).map(nfkc).str.strip()
    x = PAY[k_.isin(['3連単', '三連単'])]
    tri = {}
    for rid, c, p in zip(x['race_id'], x['combo'], x['payout']):
        nums = [int(v) for v in re.findall(r'\d+', str(c))]
        if len(nums) == 3 and p == p:
            tri.setdefault(rid, {})[tuple(nums)] = float(p)
    trio, wide = {}, {}
    for d_, words, ln in [(trio, ['3連複', '三連複'], 3), (wide, ['ワイド'], 2)]:
        x = PAY[k_.isin(words)]
        for rid, c, p in zip(x['race_id'], x['combo'], x['payout']):
            nums = tuple(sorted(int(v) for v in re.findall(r'\d+', str(c))))
            if len(nums) == ln and p == p:
                d_.setdefault(rid, {})[nums] = float(p)
    info = RACES.set_index('race_id')[['class_code', 'surface_code', 'distance', 'going_code', 'handicap']].apply(pd.to_numeric, errors='coerce').to_dict('index')
    groups = []
    for rid, g in pr.groupby('race_id'):
        g = g.sort_values('umaban'); inf = info.get(rid, {})
        if rid not in tri or len(g) < 10 or g['finish'].notna().sum() < 3 or inf.get('surface_code') == 3 or inf.get('class_code') == 1 or g['odds'].min() < 2.0:
            continue
        groups.append((rid, g))
    print(f'  {len(groups):,}レースで作ります')
    dts = pd.Series([g['date'].iloc[0] for _, g in groups])
    sp = SPLIT if (dts < SPLIT).sum() >= 200 else dts.median()   # 動作確認用（2024年6月より前が少ないとき）
    lq, ly, lq3, ly3, k3h, kwh = [], [], [], [], [], []
    for rid, g in groups:                                     # 配当の見込みと係数（2024年6月までのレースで学ぶ＝検証と同じ）
        if g['date'].iloc[0] >= sp:
            continue
        nums = list(g['umaban']); idx = {u: i for i, u in enumerate(nums)}; n = len(nums)
        P, qk = tri_prob(g['p_market'].values); look = dict(zip(P[:, 0] * n * n + P[:, 1] * n + P[:, 2], qk))
        for key, amt in tri[rid].items():
            if all(u in idx for u in key):
                q = look.get(idx[key[0]] * n * n + idx[key[1]] * n + idx[key[2]])
                if q:
                    lq.append(np.log(q)); ly.append(np.log(amt / 100))
        uk, tk = trio_tab(P, qk, n); lk = dict(zip(uk, tk))
        Ph, hk = tri_harv(g['p_market'].values); ukh, tkh = trio_tab(Ph, hk, n); lkh = dict(zip(ukh, tkh))
        for key, amt in trio.get(rid, {}).items():
            if all(u in idx for u in key):
                a_, b_, c_ = sorted(idx[u] for u in key); kk = a_ * n * n + b_ * n + c_
                if lk.get(kk):
                    lq3.append(np.log(lk[kk])); ly3.append(np.log(amt / 100))
                if lkh.get(kk):
                    k3h.append(amt / 100 * lkh[kk])
        iu, wk = wide_tab(Ph, hk, n); lw = {(min(nums[x], nums[y]), max(nums[x], nums[y])): q for x, y, q in zip(iu[0], iu[1], wk)}
        for key, amt in wide.get(rid, {}).items():
            if key in lw:
                kwh.append(amt / 100 * lw[key])
    lq, ly = np.array(lq), np.array(ly); lq3, ly3 = np.array(lq3), np.array(ly3)
    cf = np.polyfit(lq, ly, 2); smear = float(np.mean(np.exp(ly - np.polyval(cf, lq))))
    cf3 = np.polyfit(lq3, ly3, 2); sm3 = float(np.mean(np.exp(ly3 - np.polyval(cf3, lq3))))
    global BIG
    BIG = {'cf': [float(v) for v in cf], 'smear': smear, 'cf3': [float(v) for v in cf3], 'sm3': sm3,
           'k3h': float(np.median(k3h)), 'kwh': float(np.median(kwh))}
    rows = []
    for i, (rid, g) in enumerate(groups):
        P, qk = tri_prob(g['p_market'].values); _, qm = tri_prob(g['M1'].values); O = est_odds(qk)
        f = race_feat(g['odds'].values, g['p_market'].values, g['M1'].values, qk, qm, O, info.get(rid, {}), rid, g['date'].iloc[0])
        f.update(race_id=rid, date=g['date'].iloc[0], y50=int(max(tri[rid].values()) >= 500_000))
        rows.append(f)
        if (i + 1) % 2000 == 0:
            print(f'  {i + 1:,}レース')
    D = pd.DataFrame(rows)
    prm = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=1)
    tr = D[D['date'] < sp]                                 # 検証の「確かめる期間」と同じく、2024年6月までで学ぶ
    m = lgb.train(prm, lgb.Dataset(tr[FEAT_COLS], tr['y50']), num_boost_round=400)
    m.save_model(BIG_MODEL)
    last = D[D['date'] > D['date'].max() - pd.Timedelta(days=365)]   # 線は直近1年のレースで決める
    BIG.update(cut_a=float(np.nanquantile(m.predict(last[FEAT_COLS]), 1 - QCUT)), cut_b=float(np.nanquantile(last['m10'], 1 - QCUT)),
               until=str(D['date'].max().date()), n_last=int(len(last)))
    json.dump(BIG, open(BIG_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return m

if os.path.exists(BIG_JSON) and os.path.exists(BIG_MODEL):
    BIG = json.load(open(BIG_JSON, encoding='utf-8')); MBIG = lgb.Booster(model_file=BIG_MODEL)
    print(f'準備：保存済みのものを使います（荒れるレースの線は{BIG["until"]}までの1年・{BIG["n_last"]:,}レースで決定／'
          f'係数 3連複 {BIG["k3h"]:.2f}・ワイド {BIG["kwh"]:.2f}）')
else:
    if not _csv(f'{HIST_DIR}/pred6.csv'):
        raise SystemExit('先に⑭（実例の方法を試す）を最後まで動かしてください。')
    print('準備：初回なので、配当の見込み・係数・荒れるレースのモデルを作ります（10〜20分）…')
    MBIG = build_big()
    print(f'準備：できました（荒れるレースの線は{BIG["until"]}までの1年・{BIG["n_last"]:,}レースで決定／係数 3連複 {BIG["k3h"]:.2f}・ワイド {BIG["kwh"]:.2f}）')

def big_picks(g, info, rid):
    """A・Bの判定と、予想3000倍以上の3連単をM1の確率順に20点（A）／15点（B）"""
    nums = g['umaban'].astype(int).values
    P, qk = tri_prob(g['p_market'].values); _, qm = tri_prob(g['pw'].values); O = est_odds(qk)
    f = race_feat(g['odds'].values, g['p_market'].values, g['pw'].values, qk, qm, O, info, rid, DAY.isoformat())
    sa = float(MBIG.predict(pd.DataFrame([f])[FEAT_COLS])[0])
    isA = sa >= BIG['cut_a'] and len(nums) >= 16; isB = f['m10'] >= BIG['cut_b']
    k = NA if isA else (NB if isB else 0)
    sc = np.where(O >= LO, qm, -np.inf); top = [j for j in np.argsort(-sc)[:k] if np.isfinite(sc[j])]
    picks = [(f'{nums[P[j, 0]]}-{nums[P[j, 1]]}-{nums[P[j, 2]]}', float(qm[j]), float(O[j])) for j in top]
    return isA, isB, sa, f['m10'], picks

def all_picks(g, info, rid):
    """①〜④の買い目：[(券種, 組, 金額, 区分, 説明)]"""
    nums = g['umaban'].astype(int).values; n = len(nums); out = []
    isA, isB, sa, m10, P1 = big_picks(g, info, rid)
    tag = 'A・B両方' if isA and isB else ('A' if isA else ('B' if isB else ''))
    for c, p, o in P1:
        out.append(('3連単', c, 100, f'①3連単{tag}', f'予想{o:,.0f}倍・来る確率{p * 100:.3f}%'))
    # ②3連複（前の形）：補正なし
    Ph, hk = tri_harv(g['p_market'].values); _, hm = tri_harv(g['pw'].values)
    uk, tk = trio_tab(Ph, hk, n); _, tm = trio_tab(Ph, hm, n)
    def lab3(k):
        a_, b_, c_ = k // (n * n), (k // n) % n, k % n
        return '-'.join(str(v) for v in sorted((nums[a_], nums[b_], nums[c_])))
    top = np.argsort(-tm)[:20]; ev = tm[top] * BIG['k3h'] / np.clip(tk[top], 1e-12, None)
    for j, e in sorted([(j, e) for j, e in zip(top, ev) if e >= 1.0], key=lambda z: -z[1])[:3]:
        out.append(('3連複', lab3(uk[j]), 100, '②3連複', f'期待値{e:.2f}・来る確率{tm[j] * 100:.2f}%'))
    # ③3連複（期待値0.9・1点）：2着・3着の補正つき
    P, qk = tri_prob(g['p_market'].values); _, qm = tri_prob(g['pw'].values)
    uk2, tk2 = trio_tab(P, qk, n); _, tm2 = trio_tab(P, qm, n)
    top = np.argsort(-tm2)[:20]; ev = tm2[top] * est_trio(tk2[top])
    if (ev >= 0.9).any():
        j = top[int(np.argmax(np.where(ev >= 0.9, ev, -np.inf)))]
        c3 = lab3(uk2[j]); same = [i for i, o_ in enumerate(out) if o_[0] == '3連複' and o_[1] == c3]
        if same:                                              # ②と同じ組は一口にまとめる（②の印に③も重なったと書く）
            i = same[0]; out[i] = out[i][:4] + (out[i][4] + '（③とも重なり）',)
        else:
            out.append(('3連複', c3, 100, '③3連複', f'期待値{ev.max():.2f}・来る確率{tm2[j] * 100:.2f}%'))
    # ④ワイド（1組だけ）：補正なし
    iu, wk = wide_tab(Ph, hk, n); _, wm = wide_tab(Ph, hm, n)
    top = np.argsort(-wm)[:8]; ev = wm[top] * BIG['kwh'] / np.clip(wk[top], 1e-12, None)
    if (ev >= 0.9).sum() == 1:
        j = top[int(np.argmax(ev))]
        c = f'{min(nums[iu[0][j]], nums[iu[1][j]])}-{max(nums[iu[0][j]], nums[iu[1][j]])}'
        out.append(('ワイド', c, 1000, '④ワイド', f'期待値{ev.max():.2f}・来る確率{wm[j] * 100:.1f}%'))
    return isA, isB, sa, m10, out


# %% セルC：その日のレースを読み込む（出馬表。1日1回、10〜20分）
def race_list(d):
    page = get(f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={d:%Y%m%d}')
    if not page:
        return []
    return sorted({i for i in re.findall(r'race_id=(\d{12})', page) if i[4:6] in PLACES})

def fetch_entry(race_id, d):
    page = get(f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}')
    if not page:
        return None, None
    soup = BeautifulSoup(page, 'html.parser')
    rows = []
    for tr in soup.select('tr.HorseList'):
        txt_all = tr.get_text()
        if '取消' in txt_all or '除外' in txt_all:
            continue
        um = tr.find('td', class_=re.compile(r'^Umaban'))
        wk = tr.find('td', class_=re.compile(r'^Waku'))
        a_h = tr.find('a', href=re.compile(r'/horse/'))
        a_j = tr.find('a', href=re.compile(r'/jockey/'))
        ba = tr.find('td', class_='Barei')
        kin = ba.find_next_sibling('td') if ba else None
        wt = tr.find('td', class_='Weight')
        m_h = re.search(r'/horse/(\w+)', a_h['href']) if a_h else None
        m_j = re.search(r'/jockey/(?:result/recent/)?(\w+)', a_j['href']) if a_j else None
        m_w = re.search(r'(\d{3})\s*\(([+-]?\d+)\)', wt.get_text(' ', strip=True)) if wt else None
        sexage = ba.get_text(strip=True) if ba else ''
        rows.append({'race_id': race_id, 'umaban': to_num(um.get_text(strip=True)) if um else np.nan,
                     'waku': to_num(wk.get_text(strip=True)) if wk else np.nan,
                     'horse_id': m_h.group(1) if m_h else None, 'horse_name': a_h.get_text(strip=True) if a_h else '',
                     'jockey_id': m_j.group(1) if m_j else None, 'sex': sexage[:1], 'age': to_num(sexage[1:]),
                     'kinryo': to_num(kin.get_text(strip=True)) if kin else np.nan,
                     'body_weight': float(m_w.group(1)) if m_w else np.nan,
                     'body_diff': float(m_w.group(2)) if m_w else np.nan})
    texts = [soup.find(class_=c) for c in ['RaceName', 'RaceData01', 'RaceData02']]
    itext = ' '.join(t.get_text(' ', strip=True) for t in texts if t is not None)
    t_n = nfkc(itext)
    m_c = re.search(r'(芝|ダ|障)(\d{3,4})m', t_n)
    m_g = re.search(r'馬場\s*:\s*(良|稍|重|不)', t_n)
    m_p = re.search(r'本賞金\s*:\s*([\d.,]+)', t_n)
    m_t = re.search(r'(\d{1,2}:\d{2})\s*発走', t_n)
    entry = pd.DataFrame(rows)
    race = {'race_id': race_id, 'date': d.isoformat(), 'place': race_id[4:6],
            'surface_code': SURFACE.get(m_c.group(1)) if m_c else np.nan,
            'distance': to_num(m_c.group(2)) if m_c else np.nan,
            'going_code': GOING.get(m_g.group(1)) if m_g else np.nan,
            'class_code': class_code(itext), 'handicap': int('ハンデ' in itext),
            'n_runners': len(entry), 'prize1': to_num(m_p.group(1).split(',')[0]) if m_p else np.nan,
            'post': m_t.group(1) if m_t else ''}
    return entry, race

def prepare_day(day):
    global DAY, DAYF, DAYR
    DAY = day; pk = f'{S5_DIR}/day_{day:%Y%m%d}.pkl'
    if os.path.exists(pk):
        DAYF, DAYR = pd.read_pickle(pk); return True
    ids = race_list(day)
    if not ids:
        print(f'{day}：中央競馬のレースはありません。'); return False
    print(f'{day}：{len(ids)}レースの出馬表を読み込みます…')
    E, RR = [], []
    for rid in ids:
        try:
            e, r = fetch_entry(rid, day)
        except BlockedError:
            raise
        except Exception as ex:
            print(f'  {rid}：読めませんでした（{type(ex).__name__}）'); continue
        if e is not None and len(e):
            E.append(e.drop_duplicates('umaban', keep='last')); RR.append(r)
    if not E:
        print('出馬表が読めませんでした。'); return False
    E = pd.concat(E, ignore_index=True)
    for c in ['finish', 'time_sec', 'pos_first', 'pos_last', 'agari', 'odds', 'popularity']:
        E[c] = np.nan
    E['is_entry'] = True
    DAYR = pd.DataFrame(RR)
    print('材料を作っています（10〜20分）…')
    allres = pd.concat([RES[~RES['race_id'].isin(DAYR['race_id'])], E], ignore_index=True)
    allraces = pd.concat([RACES[~RACES['race_id'].isin(DAYR['race_id'])], DAYR.drop(columns='post')], ignore_index=True)
    f = add_v4_features(build_features(allres, allraces))
    DAYF = f[f['race_id'].isin(DAYR['race_id'])].copy()
    pd.to_pickle((DAYF, DAYR), pk)
    print(f'{day} の対象レース：{int(DAYF.drop_duplicates("race_id")["eligible"].sum())}レース（{EXCLUDE_NOTE}）')
    return True


# %% セルD：当日の予想（レースを選ぶ → 最新の単勝オッズで買い目を出す）
def fetch_odds(rid):
    """netkeibaの単勝オッズを読む（読めなければ None）"""
    try:
        txt = get(f'https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=update')
        j = json.loads(txt) if txt else {}
        o = (j.get('data') or {}).get('odds', {}).get('1', {})
        out = {}
        for k, v in o.items():
            val = v[0] if isinstance(v, list) else v
            try:
                x = float(val)
                if x > 0:
                    out[int(k)] = x
            except (TypeError, ValueError):
                pass
        return out if len(out) >= 2 else None
    except Exception:
        return None

def parse_odds(text):
    nums = re.findall(r'\d+(?:\.\d+)?', nfkc(text))
    out = {}
    for a, b in zip(nums[0::2], nums[1::2]):
        try:
            out[int(float(a))] = float(b)
        except ValueError:
            pass
    return out

def s5_race(rid, quiet=False):
    g = DAYF[DAYF['race_id'] == rid].drop_duplicates('umaban', keep='last').copy()
    r = DAYR.drop_duplicates('race_id', keep='last').set_index('race_id').loc[rid]
    name = f'{PLACE_NAMES.get(rid[4:6], rid[4:6])}{int(rid[-2:])}R'
    if not bool(g['eligible'].iloc[0]):
        if not quiet:
            print(f'{name}：検証の対象外のレース（{EXCLUDE_NOTE}）なので見送りです。')
        return
    # 馬体重を最新にする（発走の約1時間前に発表）
    e, _ = fetch_entry(rid, DAY)
    if e is not None and len(e):
        e = e[e['umaban'].notna()].drop_duplicates('umaban', keep='last')
        wmap = e.set_index('umaban')[['body_weight', 'body_diff']]
        g = g[g['umaban'].isin(e['umaban'])]
        g['body_weight'] = g['umaban'].map(wmap['body_weight'])
        g['body_diff'] = g['umaban'].map(wmap['body_diff'])
        g['weight_vs_avg'] = g['body_weight'] - g['prev_bw_mean']
    odds = fetch_odds(rid)
    src = 'netkeibaから自動で読み込み'
    if not odds or len(set(g['umaban'].astype(int)) - set(odds)) > 0:
        print(f'{name}：単勝オッズを読めなかったので見送り'); return
    g['odds'] = g['umaban'].astype(int).map(odds)
    if g['odds'].isna().any():
        print(f'{name}：オッズが入っていない馬がいるので、計算できません。'); return
    g = add_mkt_features(g)
    raw = MS5.predict(g[U_FEATS])
    g['pw'] = to_m1(raw / raw.sum(), g['p_market'].values)          # M1：モデルとオッズを混ぜ直した勝つ確率
    fav = g['odds'].min()
    print(f'\n■ {name}（{r.get("post", "")}発走・{len(g)}頭・単勝オッズ：{src}）')
    if fav < 2.0:
        print(f'  1番人気が{fav:.1f}倍（2.0倍未満）なので見送りです。'); return
    names = dict(zip(g['umaban'].astype(int).values, g['horse_name'].fillna('')))
    isA, isB, sa, m10, P_ = all_picks(g, r.to_dict(), rid)
    print(f'  荒れるレースの判定：A {"○" if isA else "×"}（モデル {sa:.3f}／線 {BIG["cut_a"]:.3f}・16頭以上）'
          f'　B {"○" if isB else "×"}（M1の10万円以上の確率 {m10 * 100:.1f}%／線 {BIG["cut_b"] * 100:.1f}%）')
    if not P_:
        print('  ①〜④のどれにも当てはまらないので、見送りです。'); return
    rows = []
    now = dt.datetime.now(JST).strftime('%Y-%m-%d %H:%M')
    for grp in ['①', '②', '③', '④']:
        z = [p for p in P_ if p[3].startswith(grp)]
        if not z:
            continue
        kind, stake = z[0][0], z[0][2]
        print(f'  {z[0][3]}（{len(z)}{"組" if kind == "ワイド" else "点"}・1{"組" if kind == "ワイド" else "点"}{stake:,}円＝{len(z) * stake:,}円）'
              + ('　着順どおり：1着→2着→3着' if kind == '3連単' else ''))
        for kind, c, stake, sec, note in z:
            sep = '→' if kind == '3連単' else '-'
            nm = sep.join(names.get(int(x), '')[:6] for x in c.split('-'))
            print(f'   {c.replace("-", sep):<10} {nm}　{note}')
            rows.append({'記録日時': now, 'date': str(DAY), 'race_id': rid, 'レース': name, '券種': kind, '組': c, '区分': sec,
                         '金額': stake, 'メモ': note, 'オッズの出どころ': src})
    print(f'  このレースの合計：{sum(p[2] for p in P_):,}円')
    print('  ※オッズは発走まで動くので、買う直前（締切10分前ごろ）にもう一度出し直すのがおすすめです。')
    log = pd.DataFrame(rows)
    if os.path.exists(LOG_CSV):
        old = pd.read_csv(LOG_CSV, dtype={'race_id': str})
        old = old[~((old['race_id'] == rid) & (old['date'] == str(DAY)))]           # 同じレースは最新の予想で上書き
        log = pd.concat([old, log], ignore_index=True)
    log.to_csv(LOG_CSV, index=False, encoding='utf-8-sig')
    print('  （この買い目を記録しました。あとで「答え合わせ」で成績を確かめられます）')

def pay_dicts(pays):
    """払戻の表 → {券種: {組: 100円あたりの払戻}}"""
    out = {'ワイド': {}, '3連複': {}, '3連単': {}}
    for kd, c, a in pays:
        k_ = nfkc(kd).strip().replace('三連複', '3連複').replace('三連単', '3連単')
        if k_ in out and a == a:
            nums = [int(v) for v in re.findall(r'\d+', str(c))]
            out[k_]['-'.join(str(x) for x in (nums if k_ == '3連単' else sorted(nums)))] = float(a)
    return out

def summary_table(D):
    """券種ごとと合計の成績"""
    rows = []
    col = '区分' if '区分' in D.columns else '券種'
    D = D.copy()
    if col == '区分':
        D['区分'] = D['区分'].astype(str).str.replace(r'(①3連単).*', r'\1', regex=True)
    for nm, z in sorted(D.groupby(col, sort=False), key=lambda t: str(t[0])) + [('合計', D)]:
        rr = z.groupby('race_id')['払戻'].sum()
        rows.append({'買い方': nm, 'レース': z['race_id'].nunique(), '組': len(z), '投資': f'{z["金額"].sum():,.0f}円',
                     '払戻': f'{z["払戻"].sum():,.0f}円', '収支': f'{z["払戻"].sum() - z["金額"].sum():+,.0f}円',
                     '回収率': f'{z["払戻"].sum() / z["金額"].sum() * 100:.0f}%', '的中レース': int((rr > 0).sum())})
    print(pd.DataFrame(rows).to_string(index=False))

def check_log():
    """記録した買い目の答え合わせ（結果が出たレースの払戻を読み込んで、成績を出す）"""
    if not os.path.exists(LOG_CSV):
        print('まだ記録がありません。'); return
    log = pd.read_csv(LOG_CSV, dtype={'race_id': str})
    if '券種' not in log.columns:
        log['券種'] = '3連単'
    log['券種'] = log['券種'].fillna('3連単')
    done = log[pd.to_datetime(log['date']).dt.date < TODAY].copy()
    if done.empty:
        print('結果が出ているレースの記録がまだありません（当日のレースは翌日以降に答え合わせできます）。'); return
    pdict = {}
    for rid in done['race_id'].unique():
        p = PAY[PAY['race_id'] == rid]
        if p.empty:
            out = parse_race(rid)
            p = pd.DataFrame(out[2]) if out else pd.DataFrame(columns=['kind', 'combo', 'payout'])
        pdict[rid] = pay_dicts(zip(p['kind'], p['combo'], p['payout']))
    done['払戻'] = [pdict.get(r, {}).get(k_, {}).get(c, 0.0) * y / 100
                    for r, k_, c, y in zip(done['race_id'], done['券種'], done['組'], done['金額'])]
    print(f'■ 記録の答え合わせ（{done["date"].min()}〜{done["date"].max()}）')
    summary_table(done)
    print('\n  日ごと')
    print(done.groupby('date').apply(lambda x: pd.Series({'レース': x['race_id'].nunique(), '投資': f'{x["金額"].sum():,.0f}円',
                                                            '払戻': f'{x["払戻"].sum():,.0f}円'}), include_groups=False).to_string())

def review_day():
    """終わった日の振り返り：確定オッズで買い目を出し、実際の払戻と照らし合わせる"""
    elig = DAYF.drop_duplicates('race_id').set_index('race_id')['eligible']
    rows, n_r = [], 0
    print(f'\n■ {DAY} の振り返り（確定オッズで買い目を出して、実際の払戻と照らし合わせます。1レース3秒ほど）')
    for rid in DAYR['race_id']:
        name = f'{PLACE_NAMES.get(rid[4:6], rid[4:6])}{int(rid[-2:])}R'
        if not bool(elig.get(rid, False)):
            continue
        try:
            out = parse_race(rid)
        except BlockedError:
            raise
        except Exception as e:
            print(f'  {name}：結果を読めませんでした（{type(e).__name__}）'); continue
        if not out:
            print(f'  {name}：結果がまだ読めません'); continue
        horses, _, pays = out
        horses = horses[horses['odds'].notna() & horses['umaban'].notna()]
        g = DAYF[DAYF['race_id'] == rid].drop_duplicates('umaban', keep='last').copy()
        g = g[g['umaban'].isin(horses['umaban'])]
        horses = horses.drop_duplicates('umaban', keep='last')
        hm = horses.set_index('umaban')
        g['odds'] = g['umaban'].map(hm['odds'])
        g['body_weight'] = g['umaban'].map(hm['body_weight'])
        g['body_diff'] = g['umaban'].map(hm['body_diff'])
        g['weight_vs_avg'] = g['body_weight'] - g['prev_bw_mean']
        if g['odds'].isna().any() or len(g) < 6:
            continue
        n_r += 1
        g = add_mkt_features(g)
        raw = MS5.predict(g[U_FEATS])
        g['pw'] = to_m1(raw / raw.sum(), g['p_market'].values)
        fav = g['odds'].min()
        pd_ = pay_dicts((p['kind'], p['combo'], p['payout']) for p in pays)
        if fav < 2.0:
            print(f'  {name}：1番人気{fav:.1f}倍で見送り'); continue
        isA, isB, sa, m10, P_ = all_picks(g, DAYR.drop_duplicates('race_id', keep='last').set_index('race_id').loc[rid].to_dict(), rid)
        if not P_:
            print(f'  {name}：見送り'); continue
        txt = []
        for kind, c, stake, sec, note in P_:
            py = pd_[kind].get(c, 0.0) * stake / 100
            rows.append({'date': str(DAY), 'race_id': rid, 'レース': name, '券種': kind, '組': c, '区分': sec[:1] + sec[1:].split('A')[0].split('B')[0],
                         '金額': stake, '払戻': py})
        for grp in ['①', '②', '③', '④']:
            z = [r_ for r_ in rows if r_['race_id'] == rid and r_['区分'].startswith(grp)]
            if z:
                hit = sum(r_['払戻'] for r_ in z); txt.append(f'{z[0]["区分"]}{len(z)}' + (f'（的中 {hit:,.0f}円）' if hit else ''))
        print(f'  {name}：' + '　'.join(txt))
    if not rows:
        print(f'  買い目が出たレースはありませんでした（対象 {n_r}レース）'); return
    R = pd.DataFrame(rows)
    print(f'\n  対象 {n_r}レース中、買ったのは {R["race_id"].nunique()}レース（3連単・3連複1点100円、ワイド1組1,000円）')
    summary_table(R)
    R.to_csv(f'{S5_DIR}/all_review_{DAY:%Y%m%d}.csv', index=False, encoding='utf-8-sig')
    print('  ※確定オッズでの計算です（実際に買う締切前のオッズとは少し違います）。')

def menu():
    rl = DAYR.copy()
    rl['名前'] = [f'{PLACE_NAMES.get(i[4:6], i[4:6])}{int(i[-2:])}R' for i in rl['race_id']]
    elig = DAYF.drop_duplicates('race_id').set_index('race_id')['eligible']
    while True:
        print('\n' + '=' * 40)
        for i, r in enumerate(rl.itertuples(), 1):
            mark = '' if elig.get(r.race_id, False) else '（対象外）'
            print(f'{i:>2}. {r.名前} {r.post}発走 {int(r.n_runners)}頭{mark}', end='　' if i % 3 else '\n')
        print()
        t = input('予想するレースの番号（例：11）。「全部」で全レース、「振り返り」で終わった日の成績、「答え合わせ」で記録の成績。終わるときは空のままEnter：').strip()
        if not t:
            print('おわります'); break
        if '答え' in t:
            check_log(); continue
        if '振り返' in t:
            try:
                review_day()
            except BlockedError as e:
                print(e); break
            continue
        if '全部' in t:
            print('対象の全レースを、今のオッズでまとめて出します（オッズは発走まで動くので、買う直前にもう一度出し直すのがおすすめです）')
            for rid in rl['race_id']:
                try:
                    s5_race(rid, quiet=True)
                except BlockedError as e:
                    print(e); return
                except Exception as e:
                    print(f'  {rid}：計算できませんでした（{type(e).__name__}）')
            continue
        try:
            k = int(nfkc(t)) - 1
            s5_race(rl['race_id'].iloc[k])
        except (ValueError, IndexError):
            print('番号が見つかりませんでした。')
        except BlockedError as e:
            print(e); break


# %% 自動で動かす部分
def done_path(day):
    return f'{STATE_DIR}/done_{day:%Y%m%d}.csv'
def mark_done(day, rid, status):
    p = done_path(day); now = dt.datetime.now(JST).strftime('%Y-%m-%d %H:%M')
    d = pd.read_csv(p, dtype={'race_id': str}) if os.path.exists(p) else pd.DataFrame(columns=['race_id', '状態', '時刻'])
    d = pd.concat([d[d['race_id'] != rid], pd.DataFrame([{'race_id': rid, '状態': status, '時刻': now}])], ignore_index=True)
    d.to_csv(p, index=False, encoding='utf-8-sig')

PRED_MIN = 10                                  # 発走の何分前に予想するか

def post_time(r):
    hh, mm = map(int, str(r.post).split(':'))
    return dt.datetime.combine(TODAY, dt.time(hh, mm), JST)

def predict_one(rid):
    try:
        n_before = len(pd.read_csv(LOG_CSV)) if os.path.exists(LOG_CSV) else 0
        s5_race(rid, quiet=True)
        n_after = len(pd.read_csv(LOG_CSV)) if os.path.exists(LOG_CSV) else 0
        mark_done(TODAY, rid, '買い目あり' if n_after > n_before else '見送り')
    except BlockedError as e:
        print(e); raise
    except Exception as e:
        print(f'{rid}：計算できませんでした（{type(e).__name__}）'); mark_done(TODAY, rid, 'エラー')

OWNER = 'ojplmokn-rgb'
def gh(*a):
    import subprocess
    return subprocess.run(['gh', *a], capture_output=True, text=True)
def notify(day, text):
    """GitHubのIssueにコメントして、スマホに通知する（地方の keiba-wide と同じ）"""
    if not os.environ.get('GITHUB_ACTIONS'):
        print('（通知）' + text[:200]); return
    fp = f'{STATE_DIR}/issue_{str(day).replace("-", "")}.txt'
    num = open(fp).read().strip() if os.path.exists(fp) else None
    if not num:
        r = gh('issue', 'create', '--title', f'{day} 中央競馬 ①〜④の予想', '--assignee', OWNER,
               '--body', f'@{OWNER} {day} の中央競馬の予想です。各レースの発走{PRED_MIN}分前ごろのオッズで計算し、買い目が出たレースをコメントで届けます。答え合わせの結果もここに届きます。\n\n※お金はかけない記録です。')
        m = re.search(r'/issues/(\d+)', r.stdout)
        if not m:
            print('通知を作れませんでした：', r.stderr[:200]); return
        num = m.group(1); open(fp, 'w').write(num)
    r = gh('issue', 'comment', num, '--body', f'@{OWNER} ' + text)
    if r.returncode != 0:
        print('通知できませんでした：', r.stderr[:200])

def notify_race(rid):
    log = pd.read_csv(LOG_CSV, dtype={'race_id': str})
    z = log[(log['race_id'] == rid) & (log['date'] == str(TODAY))]
    if z.empty:
        return
    post = DAYR.set_index('race_id').loc[rid, 'post']
    L = [f'**{z["レース"].iloc[0]}**（{post}発走）　合計 {z["金額"].sum():,.0f}円']
    for sec, g in z.groupby('区分', sort=True):
        sep = '→' if g['券種'].iloc[0] == '3連単' else '-'
        L.append(f'- {sec}（{len(g)}{"組" if g["券種"].iloc[0] == "ワイド" else "点"}・{g["金額"].sum():,.0f}円）：' + '　'.join(c.replace('-', sep) for c in g['組']))
    notify(TODAY, '\n'.join(L))

def notify_result(log, day):
    z = log[(log['date'] == day) & log['払戻'].notna()].copy()
    if z.empty:
        return
    z['区分'] = z['区分'].astype(str).str.replace(r'(①3連単).*', r'\1', regex=True)
    L = [f'**{day} の答え合わせ**', '', '| 買い方 | レース | 投資 | 払戻 | 回収率 |', '|---|---|---|---|---|']
    for nm, g in sorted(z.groupby('区分'), key=lambda t: t[0]) + [('合計', z)]:
        L.append(f'| {nm} | {g["race_id"].nunique()} | {g["金額"].sum():,.0f}円 | {g["払戻"].sum():,.0f}円 | {g["払戻"].sum() / g["金額"].sum() * 100:.0f}% |')
    hits = z[z['払戻'] > 0]
    L += ['', '当たり：' + ('、'.join(f'{r.レース} {r.区分} {r.組}（{r.払戻:,.0f}円）' for r in hits.itertuples()) if len(hits) else 'なし')]
    pend = log[(log['date'] == day) & log['払戻'].isna()]['race_id'].nunique()
    if pend:
        L.append(f'（まだ結果が読めていないレース：{pend}）')
    notify(day, '\n'.join(L))

def save_to_github(msg):
    """GitHub Actionsで動いているときだけ、記録をその場で保存する"""
    if not os.environ.get('GITHUB_ACTIONS'):
        return
    import subprocess
    def git(*a):
        return subprocess.run(['git', *a], capture_output=True, text=True)
    git('config', 'user.name', 'github-actions'); git('config', 'user.email', 'actions@github.com')
    os.makedirs(NEW_DIR, exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)
    git('add', '-A', NEW_DIR, STATE_DIR, S5_DIR, os.environ.get('REPORT_MD', 'report_jra.md'))
    if git('diff', '--cached', '--quiet').returncode == 0:
        return
    git('commit', '-m', msg)
    for _ in range(3):
        if git('pull', '--rebase').returncode == 0 and git('push').returncode == 0:
            print('記録を保存しました'); return
        time.sleep(5)
    print('記録の保存に失敗しました（あとでまとめて保存します）')

def run_live(window):
    """「開始-終了」の時間帯に予想するレースを、各レースの発走10分前に予想して記録する（keiba-wide と同じ形）
    例：'-13:50' は13:50まで、'13:50-16:45' は予想時刻が13:50以降のレースを16:45まで担当"""
    import subprocess
    pk = f'{S5_DIR}/day_{TODAY:%Y%m%d}.pkl'
    if os.environ.get('GITHUB_ACTIONS'):
        for _ in range(18):                                   # 朝の準備がまだなら、最大90分待つ（二重に作らないため）
            if os.path.exists(pk) or not race_list(TODAY):
                break
            print('朝の準備を待っています…'); time.sleep(300)
            subprocess.run(['git', 'pull', '--rebase', '-q'], capture_output=True)
    if not prepare_day(TODAY):
        return
    a, _, b = window.rpartition('-') if '-' in window else ('', '', window)
    def hm(x):
        h_, m_ = map(int, x.split(':')); return dt.datetime.combine(TODAY, dt.time(h_, m_), JST)
    start = hm(a) if a else None
    stop = hm(b)
    print(f'{"予想時刻が" + a + "以降のレースを" if a else ""}{stop:%H:%M} まで担当します（各レースの発走{PRED_MIN}分前に予想）')
    while True:
        now = dt.datetime.now(JST)
        if now >= stop:
            print('止める時刻になりました'); break
        done = set(pd.read_csv(done_path(TODAY), dtype={'race_id': str})['race_id']) if os.path.exists(done_path(TODAY)) else set()
        todo = sorted((post_time(r), r.race_id) for r in DAYR.itertuples() if r.race_id not in done and r.post
                      and (start is None or post_time(r) - dt.timedelta(minutes=PRED_MIN) >= start))
        if not todo:
            print('今日のレースは全部終わりました'); break
        changed = False
        for post, rid in todo:
            if (post - now).total_seconds() < 120:
                mark_done(TODAY, rid, '間に合わず'); print(f'{rid}：発走に間に合わなかった'); changed = True
            elif post - dt.timedelta(minutes=PRED_MIN) <= now:
                print(f'\n{now:%H:%M} {rid}（{post:%H:%M}発走）を予想します'); predict_one(rid); changed = True
                try:
                    notify_race(rid)
                except Exception as e:
                    print('通知でエラー：', type(e).__name__)
        if changed:
            write_report(); save_to_github(f'競馬（中央） 予想 {now:%Y-%m-%d %H:%M}'); continue
        nxt = min(p for p, _ in todo) - dt.timedelta(minutes=PRED_MIN)
        wait = max(10, min((nxt - now).total_seconds(), (stop - now).total_seconds(), 600))
        time.sleep(wait)

def run_predict():
    if not prepare_day(TODAY):
        return
    now = dt.datetime.now(JST)
    done = pd.read_csv(done_path(TODAY), dtype={'race_id': str}) if os.path.exists(done_path(TODAY)) else pd.DataFrame(columns=['race_id'])
    done_ids = set(done['race_id'])
    for r in DAYR.itertuples():
        if r.race_id in done_ids or not r.post:
            continue
        hh, mm = map(int, str(r.post).split(':'))
        post = dt.datetime.combine(TODAY, dt.time(hh, mm), JST)
        mins = (post - now).total_seconds() / 60
        if mins < 2:
            mark_done(TODAY, r.race_id, '間に合わず'); print(f'{r.race_id}：発走に間に合わなかった'); continue
        if mins > 25:
            continue
        try:
            n_before = len(pd.read_csv(LOG_CSV)) if os.path.exists(LOG_CSV) else 0
            s5_race(r.race_id, quiet=True)
            n_after = len(pd.read_csv(LOG_CSV)) if os.path.exists(LOG_CSV) else 0
            mark_done(TODAY, r.race_id, '買い目あり' if n_after > n_before else '見送り')
        except BlockedError as e:
            print(e); raise
        except Exception as e:
            print(f'{r.race_id}：計算できませんでした（{type(e).__name__}）'); mark_done(TODAY, r.race_id, 'エラー')

GROUP = {'単勝': 1, '複勝': 1, '枠連': 2, '馬連': 2, 'ワイド': 2, '馬単': 2, '3連複': 3, '3連単': 3}
def race_payouts(rid):
    """race.netkeiba の結果ページから払戻を読む（db.netkeiba が読めないとき用）"""
    try:
        page = get(f'https://race.netkeiba.com/race/result.html?race_id={rid}')
    except BlockedError as e:
        print(f'{rid}：{e}'); return None
    if not page:
        return None
    soup = BeautifulSoup(page, 'html.parser'); out = []
    for tr in soup.find_all('tr'):
        th = tr.find('th')
        if not th:
            continue
        kind = nfkc(th.get_text()).strip().replace('三連複', '3連複').replace('三連単', '3連単')
        if kind not in GROUP:
            continue
        td = tr.find('td', class_=re.compile('Result'))
        if td is None:
            continue
        nums = [int(x) for x in re.findall(r'\d+', td.get_text(' '))]
        pays = [int(x.replace(',', '')) for x in re.findall(r'([\d,]+)円', tr.get_text(' '))]
        k = GROUP[kind]
        for i, pv in enumerate(pays):
            c = nums[k * i:k * i + k]
            if len(c) == k:
                c = c if kind in ('3連単', '馬単') else sorted(c)
                out.append({'race_id': rid, 'kind': kind, 'combo': '-'.join(map(str, c)), 'payout': float(pv)})
    return out or None

DB_NG = [False]

def run_settle():
    """記録したレースと、その日の全レースの結果を集めて、払戻を書き込む"""
    import glob
    log = pd.read_csv(LOG_CSV, dtype={'race_id': str}) if os.path.exists(LOG_CSV) else pd.DataFrame()
    if len(log) and '払戻' not in log.columns:
        log['払戻'] = np.nan
    for pk in sorted(glob.glob(f'{S5_DIR}/day_*.pkl')):
        day = dt.datetime.strptime(os.path.basename(pk)[4:12], '%Y%m%d').date()
        if (TODAY - day).days > 7:
            continue
        rp = f'{NEW_DIR}/payouts_{day:%Y%m%d}.csv'
        have = set(pd.read_csv(rp, dtype={'race_id': str})['race_id']) if os.path.exists(rp) else set()
        _, dr = pd.read_pickle(pk)
        H, R_, P_ = [], [], []
        for rid in dr['race_id']:
            if rid in have:
                continue
            out = None
            if not DB_NG[0]:
                try:
                    out = parse_race(rid)
                except BlockedError as e:
                    print(f'db.netkeiba が読めません（{e}）→ 結果ページの払戻だけで答え合わせします'); DB_NG[0] = True
                except Exception:
                    out = None
            if out:
                h, rc, pays = out; H.append(h); R_.append(rc); P_ += pays
            else:
                pp = race_payouts(rid)
                if pp:
                    P_ += pp
        if P_:
            parts_ = [('payouts', pd.DataFrame(P_))] + ([('results', pd.concat(H, ignore_index=True)), ('races', pd.DataFrame(R_))] if H else [])
            for nm, df in parts_:
                p = f'{NEW_DIR}/{nm}_{day:%Y%m%d}.csv'
                old = pd.read_csv(p, dtype={'race_id': str, 'combo': str}) if os.path.exists(p) else None
                pd.concat([old, df], ignore_index=True).to_csv(p, index=False, encoding='utf-8-sig')
            print(f'{day}：{len(set(x["race_id"] for x in P_))}レースの払戻を集めました')
    if not len(log):
        return
    pp = f'{NEW_DIR}/payouts_*.csv'
    pay = pd.concat([pd.read_csv(f, dtype={'race_id': str, 'combo': str}) for f in glob.glob(pp)], ignore_index=True) if glob.glob(pp) else pd.DataFrame(columns=['race_id', 'kind', 'combo', 'payout'])
    pd_by = {rid: pay_dicts(zip(z['kind'], z['combo'], z['payout'])) for rid, z in pay.groupby('race_id')}
    todo = log['払戻'].isna() & log['race_id'].isin(pd_by)
    log.loc[todo, '払戻'] = [pd_by[r].get(k, {}).get(str(c), 0.0) * y / 100 for r, k, c, y in
                            zip(log.loc[todo, 'race_id'], log.loc[todo, '券種'], log.loc[todo, '組'], log.loc[todo, '金額'])]
    log.to_csv(LOG_CSV, index=False, encoding='utf-8-sig')
    print(f'答え合わせ：{int(todo.sum())}点を確定しました')
    for d_ in sorted(log.loc[todo, 'date'].unique()):
        notify_result(log, d_)

def md_table(df):
    cols = list(df.columns)
    return '\n'.join(['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)])

def write_report():
    import glob
    L = ['# 競馬 自動予想の成績（①〜④・お金はかけない記録）', '', f'更新：{dt.datetime.now(JST):%Y-%m-%d %H:%M}', '']
    if not os.path.exists(LOG_CSV):
        L.append('まだ記録がありません。')
    else:
        log = pd.read_csv(LOG_CSV, dtype={'race_id': str})
        if '払戻' not in log.columns:
            log['払戻'] = np.nan
        log['区分'] = log['区分'].astype(str).str.replace(r'(①3連単).*', r'\1', regex=True)
        D = log[log['払戻'].notna()]
        def tab(z):
            rows = []
            for nm, g in sorted(z.groupby('区分'), key=lambda t: t[0]) + [('合計', z)]:
                rr = g.groupby('race_id')['払戻'].sum()
                rows.append({'買い方': nm, 'レース': g['race_id'].nunique(), '点': len(g), '投資': f'{g["金額"].sum():,.0f}円',
                             '払戻': f'{g["払戻"].sum():,.0f}円', '収支': f'{g["払戻"].sum() - g["金額"].sum():+,.0f}円',
                             '回収率': f'{g["払戻"].sum() / g["金額"].sum() * 100:.0f}%' if g['金額'].sum() else '―', '的中レース': int((rr > 0).sum())})
            return pd.DataFrame(rows)
        if len(D):
            L += [f'## 全期間（{D["date"].min()}〜{D["date"].max()}）', '', md_table(tab(D)), '']
            D = D.assign(月=D['date'].str[:7])
            L += ['## 月ごと（合計）', '', md_table(pd.DataFrame([{'月': m, '投資': f'{g["金額"].sum():,.0f}円', '払戻': f'{g["払戻"].sum():,.0f}円',
                   '回収率': f'{g["払戻"].sum() / g["金額"].sum() * 100:.0f}%'} for m, g in D.groupby('月')])), '']
            L += ['## 開催日ごと（新しい順・10日分）', '', md_table(pd.DataFrame([{'日付': d, 'レース': g['race_id'].nunique(), '投資': f'{g["金額"].sum():,.0f}円',
                   '払戻': f'{g["払戻"].sum():,.0f}円'} for d, g in sorted(D.groupby('date'), reverse=True)[:10]])), '']
            hits = D[D['払戻'] > 0].sort_values('払戻', ascending=False).head(10)
            if len(hits):
                L += ['## 大きかった当たり（上位10）', '', md_table(hits[['date', 'レース', '区分', '組', '金額', '払戻']].assign(払戻=hits['払戻'].map(lambda v: f'{v:,.0f}円'))), '']
        pend = log[log['払戻'].isna()]
        if len(pend):
            L += [f'答え合わせ待ち：{pend["race_id"].nunique()}レース', '']
    st = sorted(glob.glob(f'{STATE_DIR}/done_*.csv'))
    if st:
        z = pd.concat([pd.read_csv(f).assign(日付=os.path.basename(f)[5:13]) for f in st], ignore_index=True)
        L += ['## 予想できたか（開催日ごと・新しい順10日分）', '', md_table(z.groupby('日付')['状態'].value_counts().unstack(fill_value=0).reset_index().sort_values('日付', ascending=False).head(10)), '']
    L += ['※成績は確定の払戻での記録です。買い目は発走の2〜25分前の単勝オッズで出しています。①は偶然の可能性が高く、信頼できるのは②と④です。']
    open(os.environ.get('REPORT_MD', f'{BASE}/../report_jra.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('成績表（report_jra.md）を更新しました')

try:
    if MODE == 'prepare':
        run_settle(); prepare_day(TODAY)
    elif MODE == 'live':
        run_live(sys.argv[2] if len(sys.argv) > 2 else '-16:45')
    elif MODE == 'predict':
        run_predict()
    elif MODE == 'settle':
        run_settle()
    if MODE in ('prepare', 'settle'):
        write_report(); save_to_github(f'競馬（中央） {MODE} {dt.datetime.now(JST):%Y-%m-%d %H:%M}')
except BlockedError as e:
    print(e)
write_report()
