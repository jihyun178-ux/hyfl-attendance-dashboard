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

<<<<<<< HEAD
=======
# ----------------------------- #
# 인증 함수 (상단으로 이동: load_sheet에서 바로 사용)
# ----------------------------- #
def get_creds(scopes):
    # Streamlit Cloud: secrets.toml 사용
    if "gcp_service_account" in st.secrets:
        return Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=scopes
        )
    # 로컬: service_account.json 사용
    return Credentials.from_service_account_file("service_account.json", scopes=scopes)

# ----------------------------- #
# 공통 상수/헬퍼
# ----------------------------- #
SCHOOL_GRADES = [1, 2, 3]  # 학교 체계에 맞게 조정
WEEK_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

def safe_table(df: pd.DataFrame,
               placeholder_cols: list | None = None,
               msg: str = "📭 표시할 데이터가 없습니다."):
    """빈 DF일 때도 안전하게 표/안내 출력. 사용 가능하면 True 반환."""
    if df is None or df.empty:
        st.info(msg)
        if placeholder_cols:
            st.dataframe(pd.DataFrame({c: [] for c in placeholder_cols}), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame({"정보": ["없음"]}), use_container_width=True)
        return False
    return True

# ----------------------------- #
# Google Sheets 로드
# ----------------------------- #
>>>>>>> a5cce6c (fix: 빈 데이터 안전 처리 및 3학년 데이터 예외 처리)
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
    if df.empty:
        return pd.DataFrame(columns=["이름","소속","체크인수","출석일수"])
    g = (df.groupby(["이름","소속"])
           .agg(체크인수=("이름","size"), 출석일수=("날짜","nunique"))
           .reset_index()
           .sort_values(["체크인수","출석일수"], ascending=False)
           .head(3))
    return g

def top3_attendance_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["이름","소속","출석일수","체크인수"])
    g = (df.groupby(["이름","소속"])
           .agg(출석일수=("날짜","nunique"), 체크인수=("이름","size"))
           .reset_index()
           .sort_values(["출석일수","체크인수"], ascending=False)
           .head(3))
    return g

def top3_each_grade(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """학년별 TOP3: 존재하지 않는 학년도 빈 DF로 포함"""
    out: dict[int, pd.DataFrame] = {}
    for g in SCHOOL_GRADES:
        sub = df[df["학년"] == g]
        if sub.empty:
            out[g] = pd.DataFrame(columns=["이름","반","체크인수","출석일수"])
        else:
            t = (sub.groupby(["이름","반"])
                   .agg(체크인수=("이름","size"), 출석일수=("날짜","nunique"))
                   .reset_index()
                   .sort_values(["체크인수","출석일수"], ascending=False)
                   .head(3))
            out[g] = t
    return out

def weekday_bar_for_class(df: pd.DataFrame):
    """선택 학년/반의 요일 분포 막대"""
    if df.empty:
        st.info("📭 선택한 학급의 데이터가 없습니다.")
        return
    g = (df.groupby("요일").size()
           .reindex(WEEK_ORDER)
           .fillna(0).astype(int)
           .reset_index(name="건수"))
    chart = alt.Chart(g).mark_bar().encode(
        x=alt.X("요일:O", sort=WEEK_ORDER, axis=alt.Axis(labelAngle=0)),   # 요일 라벨 가로 유지
        y=alt.Y("건수:Q", axis=alt.Axis(title="출석 건수")),
        tooltip=["요일","건수"]
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

def weekday_pivot_by_student(df: pd.DataFrame):
    """선택 학년/반에서 학생별 요일 빈도표"""
    if df.empty:
        st.info("📭 선택한 학급의 데이터가 없습니다.")
        placeholder = pd.DataFrame({c: [] for c in ["이름", *WEEK_ORDER, "합계"]})
        st.dataframe(placeholder, use_container_width=True)
        return
    p = (pd.crosstab(df["이름"], df["요일"])
           .reindex(columns=WEEK_ORDER, fill_value=0)
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

    # 교시: 2·3교시 묶어 표시
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

tab_summary, tab_homeroom, tab_trend = st.tabs(["🏆 요약·TOP3", "👩‍🏫 담임용 보기", "📈 전체 추이"])

with tab_summary:
    st.subheader("요약")
    kpi(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 체크인수 TOP3")
        t1 = top3_checkin(df)
        safe_table(t1, placeholder_cols=["이름","소속","체크인수","출석일수"], msg="📭 체크인 데이터가 없습니다.")
    with col2:
        st.markdown("### 출석일수 TOP3")
        t2 = top3_attendance_days(df)
        safe_table(t2, placeholder_cols=["이름","소속","출석일수","체크인수"], msg="📭 출석일수 데이터가 없습니다.")

    st.markdown("---")
    st.markdown("### 학년별 TOP3")
    tops = top3_each_grade(df)
    for g in SCHOOL_GRADES:
        with st.expander(f"🔹 {g}학년 TOP3", expanded=False):
            safe_table(tops[g],
                       placeholder_cols=["이름","반","체크인수","출석일수"],
                       msg=f"📭 {g}학년 데이터가 없습니다.")
            if not tops[g].empty:
                st.dataframe(tops[g], use_container_width=True)

with tab_homeroom:
    st.markdown("### 담임용 보기 — 우리 반은 무슨 요일에 많이 올까?")
    grades_opts = sorted(df0.get("학년", pd.Series(dtype="Int64")).dropna().unique().tolist())
    if not grades_opts:
        st.info("📭 학년 데이터가 없습니다.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            hr_grade = st.selectbox("학년 선택", grades_opts)
        class_opts = sorted(
            df0[df0["학년"] == hr_grade].get("반", pd.Series(dtype="Int64")).dropna().unique().tolist()
        )
        with c2:
            hr_class = st.selectbox("반 선택", class_opts) if class_opts else st.selectbox("반 선택", ["(없음)"])

        if not class_opts:
            st.info(f"📭 {hr_grade}학년의 반 데이터가 없습니다.")
        else:
            my_class_df = df[(df["학년"] == hr_grade) & (df["반"] == hr_class)].copy()
            st.markdown(f"선택한 학급: **{hr_grade}학년 {hr_class}반**  (기간: {start_date} ~ {end_date})")

            st.markdown("#### 요일 분포")
            weekday_bar_for_class(my_class_df)

            st.markdown("#### 학생별 요일표")
            weekday_pivot_by_student(my_class_df)

with tab_trend:
    st.markdown("### 날짜별 총 체크 추이")
    if df.empty:
        st.info("📭 표시할 데이터가 없습니다.")
    else:
        g = df.groupby("날짜").size().reset_index(name="건수")
        chart = alt.Chart(g).mark_line(point=True).encode(
            x="날짜:T", y="건수:Q", tooltip=["날짜:T","건수:Q"]
        )
        st.altair_chart(chart, use_container_width=True)

st.divider()
st.markdown("#### 테이블 미리보기")
st.dataframe(df.head(200), use_container_width=True)
<<<<<<< HEAD


# --- 추가: 로컬/클라우드 겸용 인증 함수

# --- 인증 헬퍼: Streamlit Cloud(Secrets) + 로컬(JSON) 둘 다 지원 ---
from google.oauth2.service_account import Credentials
import streamlit as st

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

def get_creds(scopes=None):
    """Streamlit Cloud에서는 st.secrets, 로컬에서는 service_account.json 사용"""
    scopes = scopes or SCOPES

    # ✅ Cloud: Secrets(TOML)에 넣은 서비스 계정 사용
    if "google_service_account" in st.secrets:
        info = dict(st.secrets["google_service_account"])
        return Credentials.from_service_account_info(info, scopes=scopes)

    # ✅ Local: 프로젝트 폴더의 JSON 파일 사용
    return Credentials.from_service_account_file("service_account.json", scopes=scopes)

=======
>>>>>>> a5cce6c (fix: 빈 데이터 안전 처리 및 3학년 데이터 예외 처리)
