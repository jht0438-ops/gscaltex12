
from pathlib import Path
import json
import re
import time
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="GS칼텍스 Retail/Commercial 영업 분석",
    page_icon="⛽",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SNAPSHOT_DATE = "2026-09-13"

# =========================================================
# 1. 스타일
# =========================================================
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .small-note {font-size: 0.88rem; color: #6b7280;}
      .conclusion-box {
          border: 1px solid #d9dee7;
          border-radius: 12px;
          padding: 16px 18px;
          margin-top: 18px;
          background: #fafbfc;
      }
      .conclusion-title {
          font-weight: 700;
          margin-bottom: 8px;
          font-size: 1.02rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 2. 데이터 로드
# =========================================================
@st.cache_data
def load_csv(name):
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


price = load_csv("price_cap_weekly_2026.csv")
yearly = load_csv("seoul_station_change_yearly.csv")
district_change = load_csv("seoul_station_change_by_district.csv")
district_market = load_csv("district_market_2026Q2.csv")
trade_area = load_csv("trade_area_market.csv")
opinet = load_csv("opinet_station_snapshot_20260913.csv")
opinet_full = load_csv("opinet_station_full_20260913.csv")
station_strategy = load_csv("station_strategy_20260913.csv")

# 정상 조회본만 사용
if not opinet.empty and "api_status" in opinet.columns:
    opinet = opinet[opinet["api_status"].astype(str).eq("SUCCESS")].copy()

# =========================================================
# 3. 파생지표
# =========================================================
def yn_count(s):
    return s.astype(str).str.upper().eq("Y").sum()


def pct(n, d):
    return 0 if d == 0 else n / d * 100


facility_by_district = pd.DataFrame()

if not opinet.empty:
    opinet["has_any_facility"] = (
        opinet[["CAR_WASH_YN", "MAINT_YN", "CVS_YN"]]
        .astype(str)
        .apply(lambda c: c.str.upper())
        .eq("Y")
        .any(axis=1)
    )

    facility_by_district = (
        opinet.groupby("district", dropna=False)
        .agg(
            gs_stations=("station_id", "count"),
            wash=("CAR_WASH_YN", yn_count),
            maint=("MAINT_YN", yn_count),
            cvs=("CVS_YN", yn_count),
            no_facility=("has_any_facility", lambda s: (~s).sum()),
        )
        .reset_index()
    )

    facility_by_district["wash_rate"] = (
        facility_by_district["wash"] / facility_by_district["gs_stations"] * 100
    )
    facility_by_district["maint_rate"] = (
        facility_by_district["maint"] / facility_by_district["gs_stations"] * 100
    )
    facility_by_district["cvs_rate"] = (
        facility_by_district["cvs"] / facility_by_district["gs_stations"] * 100
    )
    facility_by_district["no_facility_rate"] = (
        facility_by_district["no_facility"] / facility_by_district["gs_stations"] * 100
    )

market = district_market.copy()

if not market.empty:
    if not facility_by_district.empty:
        market = market.merge(facility_by_district, on="district", how="left")

    for c in ["gs_stations", "wash", "maint", "cvs", "no_facility",
              "wash_rate", "maint_rate", "cvs_rate", "no_facility_rate"]:
        if c not in market.columns:
            market[c] = 0
        market[c] = pd.to_numeric(market[c], errors="coerce").fillna(0)

    demand_cols = [
        "vehicles_202607",
        "floating_population",
        "resident_population",
        "worker_population",
        "apartment_households",
    ]

    for c in demand_cols:
        market[c] = pd.to_numeric(market[c], errors="coerce")
        market[f"{c}_rank"] = market[c].rank(pct=True) * 100

    market["demand_score"] = market[
        [f"{c}_rank" for c in demand_cols]
    ].mean(axis=1)

    # "수익성" 점수가 아니라, 기존 Network를 먼저 점검할 우선순위용 스크리닝
    market["network_pressure_score"] = (
        market["순증감"].abs().rank(pct=True) * 100
    )

    market["review_priority"] = (
        market["demand_score"] * 0.60
        + market["network_pressure_score"] * 0.25
        + market["no_facility_rate"].rank(pct=True) * 100 * 0.15
    )



# =========================================================
# 3-1. 최초 1회 오피넷 전체 스냅샷 → 상권매칭 → 주유소별 전략
# =========================================================
def _extract_oil(payload):
    if not isinstance(payload, dict):
        return []
    result = payload.get("RESULT", {})
    if not isinstance(result, dict):
        return []
    oil = result.get("OIL", [])
    if isinstance(oil, dict):
        return [oil]
    return oil or []


def _detail_by_id_once(api_key, station_id):
    """
    공식 문서의 certkey를 우선 사용하고, 과거 스냅샷에서 실제 성공했던 code를
    자동 fallback으로 사용한다. 사용자는 한 번만 실행하면 된다.
    """
    url = "https://www.opinet.co.kr/api/detailById.do"
    last_preview = ""

    for auth_name in ("certkey", "code"):
        params = {"out": "json", "id": station_id, auth_name: api_key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        text = r.text or ""
        last_preview = re.sub(r"\s+", " ", text[:180]).strip()

        try:
            payload = json.loads(text, strict=False)
        except Exception:
            continue

        oils = _extract_oil(payload)
        if oils:
            return oils[0], auth_name

    raise RuntimeError(f"상세정보 없음 · {station_id} · 응답: {last_preview}")


def _oil_price(detail, prodcd):
    x = detail.get("OIL_PRICE", [])
    if isinstance(x, dict):
        x = [x]
    for r in x or []:
        if str(r.get("PRODCD")) == prodcd:
            return pd.to_numeric(r.get("PRICE"), errors="coerce")
    return np.nan


def _coord_pair_from_mapping(obj):
    """GIS 좌표쌍을 안전하게 숫자로 읽는다."""
    if obj is None:
        return np.nan, np.nan
    x = pd.to_numeric(obj.get("GIS_X_COOR"), errors="coerce")
    y = pd.to_numeric(obj.get("GIS_Y_COOR"), errors="coerce")
    return x, y


def _geocode_address_to_katec(address, station_name=""):
    """
    오피넷 상세응답에 좌표가 없을 때만 사용하는 주소 기반 보완.
    OpenStreetMap Nominatim의 WGS84(lat/lon)를 받은 뒤 기존 로직이 사용하는
    KATEC 좌표로 변환한다. 실패하면 (nan, nan, 오류메시지)를 반환한다.
    """
    addr = str(address or "").strip()
    if not addr:
        return np.nan, np.nan, "주소 없음"

    # 서울 주소임을 명시해 동명이인 매칭을 줄인다.
    queries = []
    for q in [addr, f"{station_name} {addr}".strip()]:
        if q and q not in queries:
            queries.append(q)

    headers = {
        "User-Agent": "GSCaltex-Retail-Commercial-Analysis/1.0 (educational portfolio)",
        "Accept-Language": "ko,en;q=0.8",
    }

    last_error = "검색 결과 없음"
    for q in queries:
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": q,
                    "format": "jsonv2",
                    "limit": 1,
                    "countrycodes": "kr",
                    "addressdetails": 0,
                },
                headers=headers,
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                last_error = f"검색 결과 없음: {q}"
                time.sleep(1.05)
                continue

            lon = float(data[0]["lon"])
            lat = float(data[0]["lat"])

            # WGS84 -> 오피넷에서 사용하던 KATEC
            from pyproj import CRS, Transformer

            wgs84 = CRS.from_epsg(4326)
            katec = CRS.from_proj4(
                "+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 "
                "+x_0=400000 +y_0=600000 +ellps=bessel +units=m "
                "+towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43 +no_defs"
            )
            transformer = Transformer.from_crs(wgs84, katec, always_xy=True)
            x, y = transformer.transform(lon, lat)
            return float(x), float(y), ""

        except Exception as e:
            last_error = str(e)
        finally:
            # Nominatim 공개 서비스의 요청 간격을 보수적으로 유지
            time.sleep(1.05)

    return np.nan, np.nan, last_error


def _create_full_opinet_snapshot(api_key, seed_df, progress=None):
    """
    사용자가 버튼을 한 번 누르면 81개를 한 배치로 조회.
    상세정보와 기존 유외시설은 좌표 유무와 관계없이 보존한다.
    좌표는 ① 상세응답 → ② 기존 스냅샷 → ③ 주소 지오코딩 순서로 보완한다.
    완료 후 CSV 저장 → 이후 실행에서는 API를 다시 호출하지 않음.
    """
    rows = []
    total = len(seed_df)
    failures = []

    for i, (_, seed) in enumerate(seed_df.reset_index(drop=True).iterrows(), start=1):
        sid = str(seed["station_id"])
        detail_error = ""

        try:
            d, auth = _detail_by_id_once(api_key, sid)
        except Exception as e:
            # 상세조회가 실패해도 seed에 있는 기본정보/시설정보는 버리지 않는다.
            d = {}
            auth = ""
            detail_error = str(e)
            failures.append(f"{sid}: 상세조회 실패 · {e}")

        row = {
            "station_id": sid,
            "station_name": d.get("OS_NM") or seed.get("station_name"),
            "brand_code": d.get("POLL_DIV_CD") or d.get("POLL_DIV_CO") or seed.get("brand_code"),
            "address": d.get("NEW_ADR") or seed.get("address"),
            "jibun_address": d.get("VAN_ADR") or seed.get("jibun_address"),
            "district": seed.get("district"),
            "telephone": d.get("TEL") or seed.get("telephone"),
            "gasoline_price": _oil_price(d, "B027") if d else np.nan,
            "diesel_price": _oil_price(d, "D047") if d else np.nan,
            "premium_gasoline_price": _oil_price(d, "B034") if d else np.nan,
            "kerosene_price": _oil_price(d, "C004") if d else np.nan,
            "CAR_WASH_YN": d.get("CAR_WASH_YN", seed.get("CAR_WASH_YN")),
            "MAINT_YN": d.get("MAINT_YN", seed.get("MAINT_YN")),
            "CVS_YN": d.get("CVS_YN", seed.get("CVS_YN")),
            "detail_auth": auth,
            "snapshot_date": SNAPSHOT_DATE,
            "detail_status": "SUCCESS" if d else "FAILED",
            "detail_error": detail_error,
        }

        # 1) 오피넷 상세응답 좌표
        x, y = _coord_pair_from_mapping(d)
        coord_source = "OPINET_DETAIL" if pd.notna(x) and pd.notna(y) else ""
        coord_error = ""

        # 2) 기존 스냅샷 좌표가 있다면 재사용
        if pd.isna(x) or pd.isna(y):
            sx, sy = _coord_pair_from_mapping(seed)
            if pd.notna(sx) and pd.notna(sy):
                x, y = sx, sy
                coord_source = "SEED_SNAPSHOT"

        # 3) 그래도 없으면 도로명주소 → 지번주소 순으로 주소 지오코딩
        if pd.isna(x) or pd.isna(y):
            addresses = []
            for a in [row.get("address"), row.get("jibun_address")]:
                a = str(a or "").strip()
                if a and a.lower() != "nan" and a not in addresses:
                    addresses.append(a)

            for a in addresses:
                gx, gy, ge = _geocode_address_to_katec(a, row.get("station_name", ""))
                if pd.notna(gx) and pd.notna(gy):
                    x, y = gx, gy
                    coord_source = "ADDRESS_GEOCODE"
                    coord_error = ""
                    break
                coord_error = ge

        row["GIS_X_COOR"] = x
        row["GIS_Y_COOR"] = y
        row["coord_source"] = coord_source or "UNRESOLVED"
        row["coord_status"] = "SUCCESS" if pd.notna(x) and pd.notna(y) else "FAILED"
        row["coord_error"] = coord_error
        row["api_status"] = "SUCCESS" if d else "FAILED"

        if row["coord_status"] == "FAILED":
            failures.append(
                f"{sid}: 좌표 미확보 · {row.get('station_name')} · {coord_error or '좌표 없음'}"
            )

        rows.append(row)

        if progress is not None:
            progress.progress(i / total, text=f"오피넷 상세정보 수집 {i}/{total}")

    df = pd.DataFrame(rows)

    out = DATA_DIR / "opinet_station_full_20260913.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    return df, failures

def _find_trade_area_zip():
    patterns = ["*영역*상권*.zip", "*상권*영역*.zip"]
    for pat in patterns:
        xs = list(DATA_DIR.glob(pat))
        if xs:
            return xs[0]
    return None


def _load_trade_area_geometry():
    import geopandas as gpd

    zpath = _find_trade_area_zip()
    if zpath is None:
        raise FileNotFoundError("data 폴더에 서울시 상권분석서비스(영역-상권) ZIP이 없습니다.")

    tmpdir = Path(tempfile.mkdtemp(prefix="seoul_trade_area_"))
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(tmpdir)

    shps = list(tmpdir.rglob("*.shp"))
    if not shps:
        raise FileNotFoundError("영역-상권 ZIP 안에서 .shp 파일을 찾지 못했습니다.")

    # 상권코드 열이 실제로 들어 있는 shp를 선택
    for shp in shps:
        g = gpd.read_file(shp)
        cols = {str(c).upper(): c for c in g.columns}
        code_col = None
        for cand in ["TRDAR_CD", "상권_코드", "TRDAR_CODE"]:
            if cand.upper() in cols:
                code_col = cols[cand.upper()]
                break
        if code_col is None:
            for c in g.columns:
                u = str(c).upper()
                if "TRDAR" in u and "CD" in u and "NM" not in u:
                    code_col = c
                    break
        if code_col is not None:
            name_col = None
            for cand in ["TRDAR_CD_N", "TRDAR_CD_NM", "상권_코드_명", "TRDAR_NM"]:
                if cand.upper() in cols:
                    name_col = cols[cand.upper()]
                    break
            return g, code_col, name_col

    raise KeyError("영역-상권 shapefile에서 상권코드 열을 찾지 못했습니다.")


def _spatial_match_station_trade_area(full_df, trade_df):
    import geopandas as gpd
    from shapely.geometry import Point
    from pyproj import CRS

    # 오피넷/KATEC 좌표계
    katec = CRS.from_proj4(
        "+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 "
        "+x_0=400000 +y_0=600000 +ellps=bessel +units=m "
        "+towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43 +no_defs"
    )

    areas, code_col, name_col = _load_trade_area_geometry()
    if areas.crs is None:
        areas = areas.set_crs(epsg=5181)

    base = full_df.copy()
    valid_mask = base["GIS_X_COOR"].notna() & base["GIS_Y_COOR"].notna()
    valid = base.loc[valid_mask].copy()
    invalid = base.loc[~valid_mask].copy()

    matched_parts = []

    if not valid.empty:
        pts = gpd.GeoDataFrame(
            valid,
            geometry=[
                Point(float(x), float(y))
                for x, y in zip(valid["GIS_X_COOR"], valid["GIS_Y_COOR"])
            ],
            crs=katec,
        ).to_crs(areas.crs)

        keep = [code_col, "geometry"]
        if name_col is not None:
            keep.insert(1, name_col)

        joined = gpd.sjoin(pts, areas[keep], how="left", predicate="within")
        joined["match_method"] = np.where(joined[code_col].notna(), "상권 내부", "")

        # 상권 경계 밖 주유소는 최근접 상권을 연결하되 거리 기록.
        missing = joined.index[joined[code_col].isna()].tolist()
        if missing:
            nearest = gpd.sjoin_nearest(
                pts.loc[missing],
                areas[keep],
                how="left",
                distance_col="trade_area_distance_m",
            )
            nearest = nearest.sort_values("trade_area_distance_m").groupby(level=0).first()
            for idx, rr in nearest.iterrows():
                joined.loc[idx, code_col] = rr.get(code_col)
                if name_col is not None:
                    joined.loc[idx, name_col] = rr.get(name_col)
                joined.loc[idx, "trade_area_distance_m"] = rr.get("trade_area_distance_m")
                joined.loc[idx, "match_method"] = "최근접 상권"

        joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
        joined = joined.rename(columns={code_col: "상권_코드"})
        if name_col is not None:
            joined = joined.rename(columns={name_col: "matched_trade_area_name"})
        matched_parts.append(joined)

    # 좌표가 끝까지 없는 주유소도 결과에서 제거하지 않는다.
    if not invalid.empty:
        invalid["상권_코드"] = pd.Series(pd.NA, index=invalid.index, dtype="Int64")
        invalid["matched_trade_area_name"] = np.nan
        invalid["trade_area_distance_m"] = np.nan
        invalid["match_method"] = "좌표 미확보"
        matched_parts.append(invalid)

    if not matched_parts:
        return base.assign(
            상권_코드=pd.Series(pd.NA, index=base.index, dtype="Int64"),
            match_method="좌표 미확보",
            match_status="NO_COORD",
            match_quality="미확인",
        )

    joined_all = pd.concat(matched_parts, axis=0, ignore_index=False).sort_index()
    joined_all["상권_코드"] = pd.to_numeric(joined_all["상권_코드"], errors="coerce").astype("Int64")

    trade = trade_df.copy()
    trade["상권_코드"] = pd.to_numeric(trade["상권_코드"], errors="coerce").astype("Int64")

    merged = joined_all.merge(
        trade,
        on="상권_코드",
        how="left",
        suffixes=("", "_trade"),
    )

    has_coord = merged["GIS_X_COOR"].notna() & merged["GIS_Y_COOR"].notna()
    matched_ok = merged["상권_코드"].notna() & merged["상권_코드_명"].notna()
    merged["match_status"] = np.select(
        [~has_coord, matched_ok],
        ["NO_COORD", "MATCHED"],
        default="UNMATCHED",
    )

    def q(r):
        if r["match_status"] != "MATCHED":
            return "미확인"
        if r.get("match_method") == "상권 내부":
            return "높음"
        d = pd.to_numeric(r.get("trade_area_distance_m"), errors="coerce")
        if pd.isna(d):
            return "확인 필요"
        if d <= 300:
            return "보통"
        return "낮음"

    merged["match_quality"] = merged.apply(q, axis=1)
    return merged

def _rank_pct(base, value):
    s = pd.to_numeric(base, errors="coerce").dropna()
    v = pd.to_numeric(value, errors="coerce")
    if s.empty or pd.isna(v):
        return np.nan
    return float((s <= float(v)).mean() * 100)


def _top_label(p):
    if pd.isna(p):
        return "미확인"
    return f"서울 상권 상위 {max(1, int(round(100 - p)))}%"


def _create_station_strategies(matched_df, trade_df):
    metric_cols = [
        "floating_population", "resident_population", "worker_population",
        "store_auto_beauty", "store_auto_repair", "store_convenience",
        "sales_auto_beauty", "sales_auto_repair", "sales_convenience",
    ]

    out = matched_df.copy()

    for c in metric_cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c + "_pct"] = out[c].apply(
            lambda v: _rank_pct(trade_df[c], v) if c in trade_df.columns else np.nan
        )

    def yn(v):
        return str(v).strip().upper() == "Y"

    def calc(r):
        if r.get("match_status") != "MATCHED":
            return pd.Series({
                "strategy_1": np.nan, "reason_1": np.nan,
                "strategy_2": np.nan, "reason_2": np.nan,
                "strategy_3": np.nan, "reason_3": np.nan,
            })

        def p(c):
            v = pd.to_numeric(r.get(c + "_pct"), errors="coerce")
            return 0.0 if pd.isna(v) else float(v)

        wash_score = (
            .55*p("sales_auto_beauty")
            + .15*p("store_auto_beauty")
            + .30*p("floating_population")
        )
        repair_score = (
            .55*p("sales_auto_repair")
            + .15*p("store_auto_repair")
            + .30*p("worker_population")
        )
        convenience_score = (
            .45*p("sales_convenience")
            + .15*p("store_convenience")
            + .25*p("floating_population")
            + .15*p("worker_population")
        )
        pickup_score = (
            .45*p("floating_population")
            + .35*p("worker_population")
            + .20*p("resident_population")
        )

        candidates = []

        # 1) 신규 CAPEX보다 기존 시설 활용에 높은 우선순위
        if yn(r.get("CAR_WASH_YN")):
            candidates.append((
                wash_score + 25,
                "기존 세차시설 활성화",
                f"세차장을 이미 보유하고 있으며 자동차미용 추정매출은 {_top_label(r.get('sales_auto_beauty_pct'))}, "
                f"유동인구는 {_top_label(r.get('floating_population_pct'))} 수준입니다. "
                "신규 설치보다 세차 회전율·운영시간·주유 연계율을 우선 개선합니다."
            ))
        elif wash_score >= 82:
            candidates.append((
                wash_score - 8,
                "세차 제휴형 도입 검토",
                f"자동차미용 수요가 {_top_label(r.get('sales_auto_beauty_pct'))} 수준이지만 현재 세차장은 없습니다. "
                "신규 CAPEX보다 외부 운영사 제휴와 실제 세차수요 검증을 먼저 진행합니다."
            ))

        if yn(r.get("MAINT_YN")):
            candidates.append((
                repair_score + 25,
                "기존 경정비 활성화",
                f"경정비시설을 이미 보유하고 있으며 자동차수리 추정매출은 {_top_label(r.get('sales_auto_repair_pct'))}, "
                f"직장인구는 {_top_label(r.get('worker_population_pct'))} 수준입니다. "
                "엔진오일·소모품 교체 등 기존 시설 이용률과 주유 연계를 우선 높입니다."
            ))
        elif repair_score >= 85:
            candidates.append((
                repair_score - 10,
                "경정비 외부사업자 제휴 검토",
                f"자동차수리 수요가 {_top_label(r.get('sales_auto_repair_pct'))} 수준이지만 경정비시설은 없습니다. "
                "직접 투자보다 외부 정비사업자의 공간·운영 제휴를 먼저 검토합니다."
            ))

        if yn(r.get("CVS_YN")):
            candidates.append((
                convenience_score + 25,
                "기존 편의점 매출 활성화",
                f"편의점을 이미 보유하고 있으며 편의점 추정매출은 {_top_label(r.get('sales_convenience_pct'))}, "
                f"유동인구는 {_top_label(r.get('floating_population_pct'))} 수준입니다. "
                "상품구성·운영시간·주유고객 교차구매 개선을 우선합니다."
            ))
        elif convenience_score >= 83 and pickup_score >= 72:
            candidates.append((
                convenience_score - 8,
                "편의·픽업형 파트너십 검토",
                f"편의점 수요는 {_top_label(r.get('sales_convenience_pct'))}, "
                f"유동인구는 {_top_label(r.get('floating_population_pct'))} 수준입니다. "
                "직접 편의점 신설보다 택배·상품픽업 등 외부 파트너 서비스를 우선 검토합니다."
            ))

        if pickup_score >= 76:
            bonus = 8 if not (yn(r.get("CAR_WASH_YN")) or yn(r.get("MAINT_YN")) or yn(r.get("CVS_YN"))) else 0
            candidates.append((
                pickup_score + bonus,
                "유휴공간 기반 픽업·생활물류 제휴",
                f"유동인구는 {_top_label(r.get('floating_population_pct'))}, "
                f"직장인구는 {_top_label(r.get('worker_population_pct'))} 수준입니다. "
                "유휴공간이 있다면 외부 물류·픽업 사업자 제휴로 저CAPEX 유외수익을 검토합니다."
            ))

        candidates.append((
            35,
            "신규 CAPEX 보류·현장수익성 확인",
            "공개데이터만으로 신규투자를 결정하지 않습니다. "
            "실제 판매량·마진·시설 이용률·운영비·부지동선·계약조건을 현장에서 확인한 뒤 투자 여부를 결정합니다."
        ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:3]
        while len(top) < 3:
            top.append(("", "", ""))

        return pd.Series({
            "strategy_1": top[0][1], "reason_1": top[0][2],
            "strategy_2": top[1][1], "reason_2": top[1][2],
            "strategy_3": top[2][1], "reason_3": top[2][2],
        })

    rec = out.apply(calc, axis=1)
    out = pd.concat([out, rec], axis=1)

    out_path = DATA_DIR / "station_strategy_20260913.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def _build_everything_once(api_key, seed_df, trade_df, progress=None):
    full_df, failures = _create_full_opinet_snapshot(
        api_key, seed_df, progress=progress
    )

    # 좌표가 일부 없어도 전체 작업을 중단하지 않는다.
    # 확보된 주유소는 상권매칭/전략 생성을 진행하고, 미확보 주유소는 NO_COORD로 남긴다.
    matched = _spatial_match_station_trade_area(full_df, trade_df)

    if (matched["match_status"] == "MATCHED").sum() == 0:
        raise RuntimeError(
            "상권 매칭이 0개입니다. 좌표계/영역파일 문제이므로 전략 생성을 중단했습니다."
        )

    strategy_df = _create_station_strategies(matched, trade_df)
    return full_df, strategy_df, failures


# =========================================================
# 4. 공통 UI 함수
# =========================================================
def section_intro(why, how):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 왜 분석하나요?")
        st.write(why)
    with c2:
        st.markdown("#### 어떻게 보나요?")
        st.write(how)


def conclusion_box(lines):
    html = "<div class='conclusion-box'><div class='conclusion-title'>이 탭에서 얻을 수 있는 결론</div>"
    html += "<br>".join([f"• {x}" for x in lines])
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def district_conclusion(row):
    d = row["district"]
    vehicles = int(row.get("vehicles_202607", 0) or 0)
    net = int(row.get("순증감", 0) or 0)
    gs = int(row.get("gs_stations", 0) or 0)
    wash = int(row.get("wash", 0) or 0)
    maint = int(row.get("maint", 0) or 0)
    cvs = int(row.get("cvs", 0) or 0)
    none = int(row.get("no_facility", 0) or 0)
    demand = float(row.get("demand_score", 0) or 0)

    if demand >= 75:
        demand_text = "서울 내 수요지표가 높은 편이어서 기존 Network의 생산성을 우선 점검할 가치가 큽니다."
    elif demand >= 50:
        demand_text = "수요지표가 중상위권이어서 주변 상권과 실제 판매량을 함께 확인할 필요가 있습니다."
    else:
        demand_text = "수요지표만으로 공격적 투자를 결정하기보다 개별 주유소의 실제 판매·운영 데이터를 먼저 봐야 합니다."

    if net <= -10:
        net_text = f"2015~2025년 주유소 순증감이 {net}개로 감소 폭이 커, 신규 출점보다 남아 있는 Network의 유지·효율화가 더 중요합니다."
    elif net < 0:
        net_text = f"2015~2025년 주유소 순증감이 {net}개로 감소했으므로 기존 거점의 경쟁력 관리가 필요합니다."
    else:
        net_text = "주유소 수 변화만으로 구조적 감소를 단정하기보다 개별 Network의 생산성을 함께 확인해야 합니다."

    if gs == 0:
        fac_text = "현재 스냅샷에서 확인되는 GS칼텍스 주유소가 없어 유외시설 전략을 직접 비교하기 어렵습니다."
    else:
        fac_text = f"GS칼텍스 {gs}개 중 세차장 {wash}개·경정비 {maint}개·편의점 {cvs}개가 확인됩니다."

    if gs > 0 and none > 0:
        strategy = f"유외시설이 확인되지 않는 주유소가 {none}개 있어, 신규 CAPEX보다 유휴공간·외부 파트너 활용 가능성을 우선 점검할 수 있습니다."
    elif wash > 0:
        strategy = "이미 보유한 세차 등 기존 시설의 이용률·객단가를 높이는 방안이 신규 시설 투자보다 우선입니다."
    else:
        strategy = "시설 보유 여부만으로 투자 결론을 내리지 않고, 실제 부지 여건과 파트너형 운영 가능성을 추가 확인해야 합니다."

    return [
        f"{d} 자동차 등록대수는 약 {vehicles:,}대입니다. {demand_text}",
        net_text,
        fac_text,
        strategy,
    ]


# =========================================================
# 5. 헤더
# =========================================================
st.title("⛽ GS칼텍스 Retail/Commercial 영업 · Network 수익성 분석")

st.caption(
    "공개자료와 오피넷 2026-09-13 스냅샷을 활용해 "
    "가격 운용 제약과 서울 Network 변화 속에서 기존 DC·AC의 경쟁력과 유외수익 기회를 점검합니다."
)

tabs = st.tabs([
    "① 분석 개요",
    "② 최고가격제",
    "③ Network 변화",
    "④ 지역 수요",
    "⑤ 유외수익",
    "⑥ 상권 상세",
    "⑦ GS 주유소",
    "⑧ 최종결론",
])

# =========================================================
# ① 분석 개요
# =========================================================
with tabs[0]:
    st.subheader("분석 질문")

    st.info(
        "직접 보유 자산 효율화와 가격 운용 제약이 존재하는 상황에서, "
        "어떻게 기존 Network의 판매물량과 수익성을 방어할 수 있을까?"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "서울 GS칼텍스 주유소",
        f"{len(opinet):,}개" if not opinet.empty else "-"
    )

    if not opinet.empty:
        wash_total = yn_count(opinet["CAR_WASH_YN"])
        maint_total = yn_count(opinet["MAINT_YN"])
        cvs_total = yn_count(opinet["CVS_YN"])

        c2.metric(
            "세차장 보유",
            f"{wash_total:,}개",
            f"{pct(wash_total, len(opinet)):.1f}%"
        )
        c3.metric(
            "경정비 / 편의점",
            f"{maint_total:,} / {cvs_total:,}개"
        )

    st.markdown("### 분석 흐름")

    st.markdown(
        """
        **① 가격 운용 제약 확인**  
        국제제품가격 → GS 공급가격 → 주유소 평균판매가격의 흐름을 비교합니다.

        **② 서울 Network 구조 변화 확인**  
        신규등록보다 폐업·등록취소가 누적되는지 확인합니다.

        **③ 지역별 수요 확인**  
        자동차 등록대수와 유동·상주·직장인구, 아파트 수요를 결합합니다.

        **④ 기존 GS Network의 유외시설 확인**  
        세차장·경정비·편의점 보유 여부를 주유소 단위로 확인합니다.

        **⑤ 영업 우선순위 도출**  
        신규 CAPEX보다 기존 시설 활용과 외부 파트너형 수익원부터 검토합니다.
        """
    )


# =========================================================
# ② 최고가격제
# =========================================================
with tabs[1]:
    st.subheader("최고가격제와 가격 전가 제약")

    section_intro(
        "국내 판매에서 가격 조정 여력이 제한될 때, 단순한 가격지원만으로 Network 경쟁력을 유지하기 어려울 수 있기 때문입니다.",
        "국제제품가격, GS 공급가격, 전국 주유소 평균판매가격을 동일 기준의 지수로 변환해 방향성을 비교합니다."
    )

    if price.empty:
        st.warning("가격 데이터가 없습니다.")
    else:
        x = price.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")

        fuel = st.radio("제품 선택", ["휘발유", "경유"], horizontal=True)

        if fuel == "휘발유":
            raw_cols = ["intl_gasoline_92ron", "gs_gasoline_supply", "retail_gasoline"]
            labels = ["국제제품가격", "GS 공급가격", "주유소 평균판매가격"]
        else:
            raw_cols = ["intl_diesel_0001", "gs_diesel_supply", "retail_diesel"]
            labels = ["국제제품가격", "GS 공급가격", "주유소 평균판매가격"]

        idx = x[["date"] + raw_cols].copy()

        for c in raw_cols:
            idx[c] = pd.to_numeric(idx[c], errors="coerce")
            first_valid = idx[c].dropna()
            if not first_valid.empty:
                idx[c] = idx[c] / first_valid.iloc[0] * 100

        long = idx.melt(
            id_vars="date",
            value_vars=raw_cols,
            var_name="series",
            value_name="index"
        )
        long["series"] = long["series"].map(dict(zip(raw_cols, labels)))

        fig = px.line(
            long,
            x="date",
            y="index",
            color="series",
            markers=True,
            labels={"date": "시점", "index": "기준시점=100", "series": "구분"},
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "국제제품가격과 국내 공급·판매가격은 단위, 환율, 세금, 재고시점이 다르므로 "
            "단순 차액을 정유사 또는 주유소 마진으로 계산하지 않습니다."
        )

        conclusion_box([
            "국제제품가격 급등 구간에서도 국내 공급가격은 동일한 속도로 움직이지 않아 가격 전가의 제약을 확인할 수 있습니다.",
            "따라서 DC·AC 경쟁력 강화 방안을 공급가격 인하나 반복적 판촉비 투입에만 의존하기 어렵습니다.",
            "Network 수익성 개선은 가격 외 요소인 기존 시설 활용, 운영 효율, 외부 파트너십까지 함께 검토할 필요가 있습니다.",
            "이 탭은 '마진 얼마'를 추정하는 분석이 아니라, 영업전략의 제약조건을 확인하는 용도로 사용합니다.",
        ])


# =========================================================
# ③ Network 변화
# =========================================================
with tabs[2]:
    st.subheader("서울 주유소 Network 변화")

    section_intro(
        "서울의 물리적 주유소 Network가 장기적으로 줄고 있다면 신규 출점보다 기존 거점의 생산성과 유지가 더 중요해질 수 있기 때문입니다.",
        "2015~2025년 신규등록과 폐업·등록취소를 비교하고, 자치구별 순감소 폭을 확인합니다."
    )

    if yearly.empty:
        st.warning("Network 변화 데이터가 없습니다.")
    else:
        y = yearly.copy()

        fig = go.Figure()
        fig.add_bar(x=y["year"], y=y["신규등록"], name="신규등록")
        fig.add_bar(x=y["year"], y=-y["폐업·등록취소"], name="폐업·등록취소")
        fig.add_scatter(
            x=y["year"],
            y=y["순증감"],
            name="순증감",
            mode="lines+markers"
        )
        fig.update_layout(
            barmode="relative",
            xaxis_title="연도",
            yaxis_title="건수"
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("2015~2025 신규등록", f"{int(y['신규등록'].sum()):,}건")
        c2.metric("폐업·등록취소", f"{int(y['폐업·등록취소'].sum()):,}건")
        c3.metric("누적 순증감", f"{int(y['순증감'].sum()):+,}개")

        if not district_change.empty:
            d = district_change.sort_values("순증감").copy()
            fig2 = px.bar(
                d,
                x="district",
                y="순증감",
                labels={"district": "자치구", "순증감": "순증감"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        conclusion_box([
            "2015~2025년 서울은 신규등록보다 폐업·등록취소가 훨씬 많아 물리적 주유소 Network가 구조적으로 축소됐습니다.",
            "Network 감소는 곧바로 GS칼텍스 판매량 감소를 의미하지 않지만, 남아 있는 거점의 생산성과 유지 중요성을 높입니다.",
            "특히 감소 폭이 큰 자치구는 신규 출점 후보라기보다 기존 DC·AC의 경쟁력과 수익성을 먼저 점검할 지역입니다.",
            "따라서 이후 분석은 '어디에 새로 지을까'보다 '어디의 기존 Network를 우선 지킬까'에 초점을 둡니다.",
        ])


# =========================================================
# ④ 지역 수요
# =========================================================
with tabs[3]:
    st.subheader("자치구별 수요와 Network 압력")

    section_intro(
        "주유소 감소 폭이 크더라도 자동차·생활수요가 충분하지 않으면 Network 강화의 우선순위가 낮을 수 있기 때문입니다.",
        "자동차 등록대수, 유동·상주·직장인구, 아파트 가구를 함께 보고 자치구별 수요를 비교합니다."
    )

    if market.empty:
        st.warning("자치구 데이터가 없습니다.")
    else:
        districts = sorted(market["district"].dropna().unique().tolist())
        selected_d = st.selectbox("자치구 선택", districts, key="district_demand")

        row = market.loc[market["district"].eq(selected_d)].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("자동차 등록대수", f"{int(row['vehicles_202607']):,}대")
        c2.metric("상주인구", f"{int(row['resident_population']):,}명")
        c3.metric("유동인구", f"{int(row['floating_population']):,}")
        c4.metric("2015~2025 Network 순증감", f"{int(row['순증감']):+}개")

        scatter = px.scatter(
            market,
            x="vehicles_202607",
            y="순증감",
            size="resident_population",
            text="district",
            hover_data=["floating_population", "worker_population", "apartment_households"],
            labels={
                "vehicles_202607": "자동차 등록대수",
                "순증감": "2015~2025 주유소 순증감",
                "resident_population": "상주인구",
            }
        )
        scatter.update_traces(textposition="top center")
        st.plotly_chart(scatter, use_container_width=True)

        conclusion_box(district_conclusion(row))


# =========================================================
# ⑤ 유외수익
# =========================================================
with tabs[4]:
    st.subheader("기존 시설 기반 유외수익 기회")

    section_intro(
        "가격지원과 반복적 판촉은 비용이 발생합니다. 따라서 기존 주유소가 이미 가진 시설에서 추가 수익을 만들 수 있는지를 먼저 보는 것이 중요합니다.",
        "GS칼텍스 서울 주유소 81개의 세차장·경정비·편의점 보유 여부를 자치구별로 집계하고 지역 수요와 함께 봅니다."
    )

    if market.empty or opinet.empty:
        st.warning("오피넷 또는 자치구 데이터가 없습니다.")
    else:
        total = len(opinet)
        wash_total = yn_count(opinet["CAR_WASH_YN"])
        maint_total = yn_count(opinet["MAINT_YN"])
        cvs_total = yn_count(opinet["CVS_YN"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GS 주유소", f"{total:,}개")
        c2.metric("세차장", f"{wash_total:,}개", f"{pct(wash_total,total):.1f}%")
        c3.metric("경정비", f"{maint_total:,}개", f"{pct(maint_total,total):.1f}%")
        c4.metric("편의점", f"{cvs_total:,}개", f"{pct(cvs_total,total):.1f}%")

        fd = facility_by_district.sort_values("gs_stations", ascending=False).copy()
        plot = fd.melt(
            id_vars=["district", "gs_stations"],
            value_vars=["wash", "maint", "cvs"],
            var_name="facility",
            value_name="count"
        )
        plot["facility"] = plot["facility"].map(
            {"wash": "세차장", "maint": "경정비", "cvs": "편의점"}
        )

        fig = px.bar(
            plot,
            x="district",
            y="count",
            color="facility",
            barmode="group",
            labels={"district": "자치구", "count": "보유 주유소 수", "facility": "시설"},
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_d = st.selectbox(
            "자치구별 유외수익 결론 보기",
            sorted(market["district"].unique().tolist()),
            key="district_nonfuel"
        )

        row = market.loc[market["district"].eq(selected_d)].iloc[0]
        conclusion_box(district_conclusion(row))


# =========================================================
# ⑥ 상권 상세
# =========================================================
with tabs[5]:
    st.subheader("상권 단위 세부 확인")

    section_intro(
        "자치구 평균만으로는 같은 구 안에서도 다른 생활·상업수요를 설명하기 어렵기 때문입니다.",
        "상권별 유동·상주·직장인구와 편의점·자동차수리·자동차미용 점포 및 추정매출을 함께 확인합니다."
    )

    if trade_area.empty:
        st.warning("상권 데이터가 없습니다.")
    else:
        search = st.text_input("상권명 검색", placeholder="예: 잠실, 강남, 마곡")

        z = trade_area.copy()
        if search:
            z = z[
                z["상권_코드_명"]
                .astype(str)
                .str.contains(search, case=False, na=False)
            ]

        show_cols = [
            "상권_코드_명", "상권_구분_코드_명",
            "floating_population", "resident_population", "worker_population",
            "store_auto_beauty", "store_auto_repair", "store_convenience",
            "sales_auto_beauty", "sales_auto_repair", "sales_convenience"
        ]
        show_cols = [c for c in show_cols if c in z.columns]

        st.dataframe(
            z[show_cols].head(300),
            hide_index=True,
            use_container_width=True,
            height=480,
        )

        conclusion_box([
            "같은 자치구라도 상권별 유동·거주·직장 수요와 자동차 관련 업종 분포가 달라 주유소별 전략을 동일하게 적용하기 어렵습니다.",
            "자동차미용·수리 수요가 이미 형성된 상권은 기존 세차·경정비 시설의 이용률 확대 가능성을 우선 검토할 수 있습니다.",
            "편의점 매출이나 점포 수가 높은 상권도 곧바로 주유소 내 편의점 신설을 의미하지 않으며, 부지·동선·임대조건 확인이 필요합니다.",
            "따라서 상권 데이터는 투자 결론이 아니라 현장 방문과 실제 매출 확인의 우선순위를 정하는 자료로 사용합니다.",
        ])


# =========================================================
# ⑦ GS 주유소 · 주유소별 유외전략
# =========================================================
with tabs[6]:
    st.subheader(f"GS칼텍스 주유소별 유외전략 · {SNAPSHOT_DATE}")

    section_intro(
        "자치구 평균이 아니라 실제 A주유소 단위로 '어떤 유외전략을 우선할 것인가'까지 내려가기 위해서입니다.",
        "오피넷 좌표·기존시설과 서울시 상권의 유동·상주·직장인구, 자동차미용·수리·편의점 점포·추정매출을 연결합니다."
    )

    # 전략 파일이 없을 때만 최초 1회 API 수집 버튼 표시
    if station_strategy.empty:
        st.info(
            "최초 1회만 실행합니다. 아래 버튼을 누르면 오피넷 상세정보를 한 번의 배치 작업으로 수집해 "
            "`opinet_station_full_20260913.csv`와 `station_strategy_20260913.csv`를 저장합니다. "
            "그 다음부터는 API를 호출하지 않고 저장된 파일만 사용합니다."
        )

        area_zip = _find_trade_area_zip()
        if area_zip is None:
            st.error("data 폴더에 `서울시 상권분석서비스(영역-상권).zip`을 넣어주세요.")
        elif opinet.empty:
            st.error("기존 `opinet_station_snapshot_20260913.csv`가 없습니다.")
        elif trade_area.empty:
            st.error("`trade_area_market.csv`가 없습니다.")
        else:
            api_key = st.text_input(
                "오피넷 API Key",
                type="password",
                help="키는 저장하지 않습니다. 최초 1회 데이터 생성에만 사용됩니다."
            )

            if st.button("최초 1회 생성: 81개 GS 주유소 → 상권 → 유외전략", type="primary"):
                if not api_key.strip():
                    st.error("오피넷 API Key를 입력해주세요.")
                else:
                    prog = st.progress(0, text="오피넷 상세정보 수집 준비")
                    try:
                        full_df, generated, failures = _build_everything_once(
                            api_key.strip(),
                            opinet.copy(),
                            trade_area.copy(),
                            progress=prog,
                        )
                        prog.progress(1.0, text="완료")

                        matched_n = int(generated["match_status"].eq("MATCHED").sum())
                        no_coord_n = int(generated["match_status"].eq("NO_COORD").sum())
                        geocoded_n = int(full_df.get("coord_source", pd.Series(dtype=str)).astype(str).eq("ADDRESS_GEOCODE").sum())
                        st.success(
                            f"완료 · 오피넷 {len(full_df)}개 수집 · 상권매칭 {matched_n}/{len(generated)}개 · "
                            f"주소기반 좌표보완 {geocoded_n}개 · 좌표 미확보 {no_coord_n}개"
                        )
                        if no_coord_n > 0:
                            st.warning(
                                f"좌표를 끝까지 확보하지 못한 {no_coord_n}개 주유소는 임의의 0값을 넣지 않고 "
                                "NO_COORD로 보존했습니다. 나머지 주유소의 전략은 정상 생성했습니다."
                            )

                        st.download_button(
                            "완성된 station_strategy_20260913.csv 다운로드",
                            generated.to_csv(index=False).encode("utf-8-sig"),
                            file_name="station_strategy_20260913.csv",
                            mime="text/csv",
                        )
                        st.caption(
                            "로컬 실행에서는 data 폴더에 파일이 이미 저장됩니다. "
                            "GitHub/Streamlit Cloud에 배포할 때만 생성된 CSV를 data 폴더에 한 번 올려두면 이후 API 호출이 없습니다."
                        )
                        st.rerun()

                    except Exception as e:
                        st.error(f"생성 중단: {e}")
                        st.caption(
                            "좌표가 일부 실패해도 0으로 대체하지 않고 NO_COORD로 보존합니다. "
                            "전체 상권매칭이 0개인 경우처럼 분석 자체가 불가능할 때만 중단합니다."
                        )

    else:
        s = station_strategy.copy()
        valid = s["match_status"].astype(str).eq("MATCHED") if "match_status" in s.columns else pd.Series(False, index=s.index)

        c1, c2, c3 = st.columns(3)
        c1.metric("전체 GS 주유소", f"{len(s)}개")
        c2.metric("상권 매칭 성공", f"{int(valid.sum())}개")
        c3.metric("매칭률", f"{valid.mean()*100:.1f}%")

        if valid.sum() == 0:
            st.error("상권 매칭 성공 주유소가 0개라 전략을 표시하지 않습니다.")
        else:
            districts = sorted(s.loc[valid, "district"].dropna().astype(str).unique().tolist())
            selected_d = st.selectbox("자치구", districts, key="district_station_strategy")

            g = s[valid & s["district"].astype(str).eq(selected_d)].copy()

            station = st.selectbox(
                "주유소 선택",
                sorted(g["station_name"].astype(str).unique().tolist()),
                key="station_strategy_select"
            )
            r = g[g["station_name"].astype(str).eq(station)].iloc[0]

            def yn_label(v):
                return "있음" if str(v).strip().upper() == "Y" else "없음"

            def nfmt(v):
                x = pd.to_numeric(v, errors="coerce")
                return "미확인" if pd.isna(x) else f"{x:,.0f}"

            def pctlabel(v):
                x = pd.to_numeric(v, errors="coerce")
                return "미확인" if pd.isna(x) else f"서울 상권 상위 {max(1, int(round(100-x)))}%"

            st.markdown("### 1. 현재 주유소")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("세차장", yn_label(r.get("CAR_WASH_YN")))
            c2.metric("경정비", yn_label(r.get("MAINT_YN")))
            c3.metric("편의점", yn_label(r.get("CVS_YN")))
            c4.metric("상권매칭", str(r.get("match_quality", "미확인")))

            st.write("**주소**", r.get("address", ""))
            st.write("**연결 상권**", r.get("상권_코드_명", r.get("matched_trade_area_name", "미확인")))
            if r.get("match_method") == "최근접 상권":
                d = pd.to_numeric(r.get("trade_area_distance_m"), errors="coerce")
                if pd.notna(d):
                    st.caption(f"주유소가 상권영역 경계 밖에 있어 약 {d:,.0f}m 거리의 최근접 상권을 연결했습니다.")

            st.markdown("### 2. 주변 수요")
            c1, c2, c3 = st.columns(3)
            c1.metric("유동인구", nfmt(r.get("floating_population")))
            c2.metric("상주인구", nfmt(r.get("resident_population")))
            c3.metric("직장인구", nfmt(r.get("worker_population")))

            st.markdown("### 3. 주변 유외시장")
            market_rows = pd.DataFrame([
                {
                    "구분": "자동차미용",
                    "점포 수": nfmt(r.get("store_auto_beauty")),
                    "추정매출": nfmt(r.get("sales_auto_beauty")),
                    "서울 상권 내 수준": pctlabel(r.get("sales_auto_beauty_pct")),
                },
                {
                    "구분": "자동차수리",
                    "점포 수": nfmt(r.get("store_auto_repair")),
                    "추정매출": nfmt(r.get("sales_auto_repair")),
                    "서울 상권 내 수준": pctlabel(r.get("sales_auto_repair_pct")),
                },
                {
                    "구분": "편의점",
                    "점포 수": nfmt(r.get("store_convenience")),
                    "추정매출": nfmt(r.get("sales_convenience")),
                    "서울 상권 내 수준": pctlabel(r.get("sales_convenience_pct")),
                },
            ])
            st.dataframe(market_rows, hide_index=True, use_container_width=True)

            st.markdown("### 4. 그래서 이 주유소에서는 무엇을 할 것인가")
            st.success(f"**1순위 · {r.get('strategy_1','')}**\n\n{r.get('reason_1','')}")
            if pd.notna(r.get("strategy_2")) and str(r.get("strategy_2")).strip():
                st.info(f"**2순위 · {r.get('strategy_2','')}**\n\n{r.get('reason_2','')}")
            if pd.notna(r.get("strategy_3")) and str(r.get("strategy_3")).strip():
                st.info(f"**3순위 · {r.get('strategy_3','')}**\n\n{r.get('reason_3','')}")

            st.markdown("### 5. 실제 영업담당자가 추가 확인할 것")
            st.write(
                "유류 판매량·마진, 기존 시설 이용률과 객단가, 운영비·인력, 유휴공간 면적과 차량동선, "
                "계약조건, 인근 경쟁점의 가격·서비스 수준을 현장에서 확인한 뒤 실행 여부를 결정합니다."
            )

            st.caption(
                "상권 추정매출은 해당 주유소의 실제 매출이 아니라 상권 수요를 보는 스크리닝 지표입니다."
            )

            with st.expander(f"{selected_d} 전체 GS 주유소 1순위 전략"):
                show_cols = [
                    "station_name", "상권_코드_명",
                    "CAR_WASH_YN", "MAINT_YN", "CVS_YN",
                    "strategy_1", "match_quality"
                ]
                show_cols = [c for c in show_cols if c in g.columns]
                st.dataframe(g[show_cols], hide_index=True, use_container_width=True)

# =========================================================
# ⑧ 최종결론
# =========================================================
with tabs[7]:
    st.subheader("프로젝트 최종결론")

    st.markdown(
        """
        ### 1. 가격보다 Network 생산성을 먼저 본다
        최고가격제 환경에서는 가격을 통한 마진 확보와 거래처 지원의 운용폭이 제한될 수 있습니다.
        따라서 단순 가격경쟁보다 **기존 DC·AC가 스스로 수익을 만들 수 있는 구조**가 중요합니다.

        ### 2. 서울에서는 신규 출점보다 기존 거점 유지가 우선이다
        2015~2025년 서울 주유소는 신규등록보다 폐업·등록취소가 훨씬 많았습니다.
        이는 신규 주유소 건설의 필요성을 의미하는 것이 아니라,
        **고수요 지역에 남아 있는 기존 Network의 판매물량과 생산성을 지킬 필요성**을 보여줍니다.

        ### 3. 유외수익은 '있는 시설부터' 활용한다
        2026-09-13 스냅샷 기준 서울 GS칼텍스 주유소 81개 중
        세차장은 58개, 경정비는 16개, 편의점은 9개가 확인됩니다.
        따라서 **기존 세차·경정비·편의점 이용률 확대 → 유휴공간 외부 파트너 활용 → 필요한 경우 신규 CAPEX**
        순서가 합리적입니다.

        ### 4. 주유소마다 동일한 처방을 적용하지 않는다
        실제 GS 주유소 위치를 서울시 상권과 연결해 유동·상주·직장인구와 자동차미용·수리·편의점 수요를 함께 봅니다.
        따라서 각 주유소별로 **기존 세차 활성화 / 기존 경정비 활성화 / 편의·픽업 파트너십 / 신규 CAPEX 보류** 중 우선전략을 다르게 제시합니다.

        ### 5. 공개데이터는 의사결정의 시작점이다
        본 분석은 실제 내부 수익성을 추정하지 않습니다.
        현업에서는 **판매량·마진·시설 이용률·운영비·계약조건·부지 여건**을 추가 확인한 뒤
        최종 지원·투자·Network 전략을 결정해야 합니다.
        """
    )

    if not market.empty:
        st.markdown("### 우선 점검 지역 스크리닝")
        top = (
            market.sort_values("review_priority", ascending=False)
            .head(8)[
                [
                    "district",
                    "vehicles_202607",
                    "순증감",
                    "gs_stations",
                    "wash",
                    "maint",
                    "cvs",
                    "review_priority",
                ]
            ]
            .copy()
        )

        top["review_priority"] = top["review_priority"].round(1)

        top = top.rename(
            columns={
                "district": "자치구",
                "vehicles_202607": "자동차 등록대수",
                "순증감": "Network 순증감",
                "gs_stations": "GS 주유소",
                "wash": "세차장",
                "maint": "경정비",
                "cvs": "편의점",
                "review_priority": "점검 우선순위",
            }
        )

        st.dataframe(top, hide_index=True, use_container_width=True)

        st.caption(
            "점검 우선순위는 수익성 예측값이 아니라, 자동차·생활수요와 Network 감소, "
            "기존 유외시설 공백을 함께 본 스크리닝 지표입니다."
        )
