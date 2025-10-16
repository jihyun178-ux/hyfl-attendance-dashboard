import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import datetime as dt
import gspread
from google.oauth2.service_account import Credentials

# ✅ Streamlit Cloud secrets에서 구글 서비스 계정 불러오기
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_gspread_client():
    sa_info = st.secrets["google_service_account"]
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_data(ttl=60)
def load_sheet(sheet_key: str, worksheet: str) -> pd.DataFrame:
    gc = get_gspread_client()
    sh = gc.open_by_key(sheet_key)
    ws = sh.worksheet(worksheet)

    EXPECTED_HEADERS = ["날짜","교시","좌석번호","이메일","시간","학년","반","번호","이름"]
    rows = ws.get_all_records(head=1, expected_headers=EXPECTED_HEADERS)
    df = pd.DataFrame(rows)

    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce").dt.date
    for c in ["학년", "반", "번호"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "교시" in df.columns:
        df["교시"] = df["교시"].astype(str).str.replace("교시", "", regex=False)
        df["교시"] = pd.to_numeric(df["교시"], errors="coerce").astype("Int64")

    df["소속"] = df.apply(lambda r: f"{r.get('학년','')}-{r.get('반','')}", axis=1)
    weekday_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
    df["요일"] = pd.to_datetime(df["날짜"], errors="coerce").dt.weekday.map(weekday_map)

    return df


# ----------------------------- #
# 지표/표/차트 헬퍼
# ----------------------------- #
def kpi(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    c1.metric("총 체크 수", f"{len(df):,}")
    c2.metric("학생 수", df["이름"].nunique() if "이름" in df.columns else 0)
    c3.metric("운영 일수", df["날짜"].nunique() if "날짜" in df.columns else 0)

def top3_checkin(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["이름","소속","체크인수","출석일수"])
    g = (df.groupby(["이름","소속"])
           .agg(체크인수=("이름","size"), 출석일수=("날짜","nunique"))
           .reset_index()
           .sort_values(["체크인수","출석일수"], ascending=False)
           .head(3))
    return g

def top3_attendance_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["이름","소속","출석일수","체크인수"])
    g = (df.groupby(["이름","소속"])
           .agg(출석일수=("날짜","nunique"), 체크인수=("이름","size"))
           .reset_index()
           .sort_values(["출석일수","체크인수"], ascending=False)
           .head(3))
    return g

def top3_each_grade(df: pd.DataFrame) -> dict:
    """학년별 TOP3 테이블 dict[학년] = DataFrame"""
    out = {}
    if df.empty: return out
    for g in sorted(df["학년"].dropna().unique().tolist()):
        sub = df[df["학년"] == g]
        t = (sub.groupby(["이름","반"])
               .agg(체크인수=("이름","size"), 출석일수=("날짜","nunique"))
               .reset_index()
               .sort_values(["체크인수","출석일수"], ascending=False)
               .head(3))
        out[int(g)] = t
    return out

def weekday_bar_for_class(df: pd.DataFrame):
    """선택 학년/반의 요일 분포 막대"""
    if df.empty:
        st.info("선택한 학급의 데이터가 없습니다.")
        return
    order = ["월","화","수","목","금","토","일"]
    g = (df.groupby("요일").size()
           .reindex(order)
           .fillna(0).astype(int)
           .reset_index(name="건수"))
    chart = alt.Chart(g).mark_bar().encode(
        x=alt.X("요일:O", sort=order, axis=alt.Axis(labelAngle=0)),   # ← 요일 라벨 가로 유지
        y=alt.Y("건수:Q", axis=alt.Axis(title="출석 건수")),
        tooltip=["요일","건수"]
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

def weekday_pivot_by_student(df: pd.DataFrame):
    """선택 학년/반에서 학생별 요일 빈도표"""
    if df.empty:
        st.dataframe(pd.DataFrame({"정보":"없음"}))
        return
    order = ["월","화","수","목","금","토","일"]
    p = (pd.crosstab(df["이름"], df["요일"])
           .reindex(columns=order, fill_value=0)
           .sort_index())
    p["합계"] = p.sum(axis=1)
    p = p.sort_values("합계", ascending=False)
    st.dataframe(p, use_container_width=True)

# ----------------------------- #
# UI
# ----------------------------- #
st.title("📊 야간자율학습 출석 대시보드")

with st.sidebar:
    st.header("데이터 연결")
    sheet_key = st.text_input("Google Sheet Key", "1LH_AI8jvW-vNn9I8wsj8lIot16vuLzqyjbZfDqcNgM8")
    worksheet = st.text_input("워크시트 이름", "출석기록")

    st.header("필터")
    df0 = pd.DataFrame()
    if sheet_key and worksheet:
        try:
            df0 = load_sheet(sheet_key, worksheet)
        except Exception as e:
            st.error(f"시트 불러오기 실패: {e}")

    # 기간 선택 (기본: 이번달)
    today = dt.date.today()
    first_day = today.replace(day=1)
    date_range = st.date_input("기간 선택 (기본: 이번 달)", [first_day, today])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = first_day, today

    # 기본 목록
    grades_all  = sorted(df0.get("학년", pd.Series(dtype="Int64")).dropna().unique().tolist())
    classes_all = sorted(df0.get("반",   pd.Series(dtype="Int64")).dropna().unique().tolist())

    f_grades  = st.multiselect("학년", grades_all, default=grades_all)
    f_classes = st.multiselect("반", classes_all)

    # 교시: 2·3교시를 묶어 보여주기
    periods_raw = sorted(df0.get("교시", pd.Series(dtype="Int64")).dropna().unique().tolist())
    label_to_periods = {}
    if periods_raw:
        if 1 in periods_raw: label_to_periods["1교시"] = [1]
        if 2 in periods_raw or 3 in periods_raw:
            label_to_periods["2~3교시"] = [p for p in [2,3] if p in periods_raw]
        for p in periods_raw:
            if p not in [1,2,3]:
                label_to_periods[f"{p}교시"] = [p]
    period_labels_all = list(label_to_periods.keys())
    f_periods_labels = st.multiselect("교시", period_labels_all, default=period_labels_all)

    f_query = st.text_input("검색(이름/이메일)")

if df0.empty:
    st.info("왼쪽에서 시트 키/워크시트를 입력하고, 같은 폴더에 service_account.json 파일이 있어야 합니다.")
    st.stop()

# ----------------------------- #
# 공통 필터 적용
# ----------------------------- #
mask = (
    (pd.to_datetime(df0["날짜"]) >= pd.to_datetime(start_date))
    & (pd.to_datetime(df0["날짜"]) <= pd.to_datetime(end_date))
)
df = df0.loc[mask].copy()

if f_grades:
    df = df[df["학년"].isin(f_grades)]
if f_classes:
    df = df[df["반"].isin(f_classes)]
if f_periods_labels:
    sel = []
    for lab in f_periods_labels:
        sel.extend(label_to_periods.get(lab, []))
    if sel:
        df = df[df["교시"].isin(sel)]
if f_query:
    s1 = df.get("이름", pd.Series([""]*len(df))).astype(str)
    s2 = df.get("이메일", pd.Series([""]*len(df))).astype(str)
    df = df[s1.str.contains(f_query, case=False, na=False) | s2.str.contains(f_query, case=False, na=False)]

# ----------------------------- #
# 본문: 요약/Top3 + 담임용 보기 + 추이
# ----------------------------- #
st.success(f"연결된 시트: https://docs.google.com/spreadsheets/d/{sheet_key} (탭: {worksheet})")

tab_summary, tab_homeroom, tab_trend = st.tabs(
    ["🏆 요약·TOP3", "👩‍🏫 담임용 보기", "📈 전체 추이"]
)

with tab_summary:
    st.subheader("요약")
    kpi(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 체크인수 TOP3")
        st.dataframe(top3_checkin(df), use_container_width=True)
    with col2:
        st.markdown("### 출석일수 TOP3")
        st.dataframe(top3_attendance_days(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### 학년별 TOP3")
    tops = top3_each_grade(df)
    if not tops:
        st.info("표시할 데이터가 없습니다.")
    else:
        for g in sorted(tops.keys()):
            with st.expander(f"🔹 {g}학년 TOP3", expanded=False):
                st.dataframe(tops[g], use_container_width=True)

with tab_homeroom:
    st.markdown("### 담임용 보기 — 우리 반은 무슨 요일에 많이 올까?")
    c1, c2 = st.columns(2)
    with c1:
        hr_grade = st.selectbox("학년 선택", sorted(df0["학년"].dropna().unique().tolist()))
    with c2:
        hr_class = st.selectbox("반 선택", sorted(df0[df0["학년"]==hr_grade]["반"].dropna().unique().tolist()))

    my_class_df = df[(df["학년"]==hr_grade) & (df["반"]==hr_class)].copy()
    st.markdown(f"선택한 학급: **{hr_grade}학년 {hr_class}반**  (기간: {start_date} ~ {end_date})")

    st.markdown("#### 요일 분포")
    weekday_bar_for_class(my_class_df)

    st.markdown("#### 학생별 요일표")
    weekday_pivot_by_student(my_class_df)

with tab_trend:
    st.markdown("### 날짜별 총 체크 추이")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        g = df.groupby("날짜").size().reset_index(name="건수")
        chart = alt.Chart(g).mark_line(point=True).encode(
            x="날짜:T", y="건수:Q", tooltip=["날짜:T","건수:Q"]
        )
        st.altair_chart(chart, use_container_width=True)

st.divider()
st.markdown("#### 테이블 미리보기")
st.dataframe(df.head(200), use_container_width=True)


# --- 추가: 로컬/클라우드 겸용 인증 함수 ---
from google.oauth2.service_account import Credentials

def get_creds(scopes):
    # Streamlit Cloud에서는 secrets에 넣어둔 서비스 계정 정보를 사용
    if "gcp_service_account" in st.secrets:
        return Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),  # secrets.toml의 키 이름
            scopes=scopes
        )
    # 로컬에서는 기존 JSON 파일 사용
    return Credentials.from_service_account_file("service_account.json", scopes=scopes)
