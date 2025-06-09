
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.font_manager as fm
import lightgbm as lgb
from datetime import timedelta

# 環境変数をロード
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# フォント設定（日本語用）
font_path = "fonts/ipaexg.ttf"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

BASE_URL = "https://delica-insight-bot.onrender.com"

def clean_numeric_column(df, col):
    return df[col].replace(",", "", regex=True).astype(float)

def prepare_graph_dir():
    dir_path = "static/graphs"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def apply_font(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    ax.title.set_fontproperties(font_prop)
    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)

def generate_graphs(df: pd.DataFrame, dir_path: str):
    sns.set(style="whitegrid")
    urls = []

    # グラフ1
    fig, ax = plt.subplots(figsize=(10, 5))
    daily = df.groupby("日付")["販売金額"].sum()
    daily.plot(marker="o", ax=ax)
    ax.set_title("日別売上金額の推移")
    ax.set_ylabel("売上金額（円）")
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels(daily.index, rotation=45)
    apply_font(ax)
    path = f"{dir_path}/graph1_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "日別売上金額", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    # グラフ2
    fig, ax = plt.subplots(figsize=(6, 6))
    cat = df.groupby("カテゴリ")["販売金額"].sum()
    ax.pie(cat, labels=cat.index, autopct='%1.1f%%', startangle=140, textprops={'fontproperties': font_prop})
    ax.set_title("カテゴリ別売上構成比", fontproperties=font_prop)
    path = f"{dir_path}/graph2_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "カテゴリ別売上構成", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    # グラフ3
    fig, ax = plt.subplots(figsize=(10, 5))
    top = df.groupby("商品名")["販売数量"].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("商品別販売数量ランキング Top10")
    ax.set_xlabel("販売数量")
    apply_font(ax)
    path = f"{dir_path}/graph3_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "商品別販売数量", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    return urls

def create_prompt(df: pd.DataFrame, graphs) -> str:
    top_product = df.groupby("商品名")["販売数量"].sum().sort_values(ascending=False).head(3)
    top_categories = df.groupby("カテゴリ")["販売金額"].sum().sort_values(ascending=False).head(3)
    discount_stats = df["値引き率"].str.replace("%", "").astype(float).describe().to_dict()
    waste_stats = df["廃棄率"].str.replace("%", "").astype(float).describe().to_dict()
    daily = df.groupby("日付")["販売金額"].sum().to_dict()
    category_share = df.groupby("カテゴリ")["販売金額"].sum().to_dict()
    product_quantity = df.groupby("商品名")["販売数量"].sum().sort_values(ascending=False).head(10).to_dict()

    prompt = f"""
あなたは小売部門の売上分析担当アシスタントです。
以下の1週間分の売上データとグラフに基づいて、次のようなアウトプットを生成してください。

- 商品別・カテゴリ別の売上傾向や気づきを分析
- 値引き率や廃棄率が高い商品への改善提案
- 来週に向けた販売戦略（仕入れ強化・POP・販促など）

【出力形式】
・箇条書きで3〜5個にまとめてください
・現場のデリカ担当者がすぐ動けるような視点で書いてください
・300文字以内で

- 売上上位商品: {top_product},
- 売上上位カテゴリ: {top_categories},
- 値引き率の統計情報: {discount_stats},
- 廃棄率の統計情報: {waste_stats},
- 日別売上金額: {daily},
- カテゴリ別売上金額: {category_share},
- 販売数量Top10商品: {product_quantity}

【参考グラフ】
"""
    for graph in graphs:
        prompt += f"\n- {graph['title']}:\n![]({graph['url']})"
    return prompt

# 曜日ダミー列作成
def add_weekday_dummies(df):
    df["日付"] = pd.to_datetime(df["日付"])
    df["曜日"] = df["日付"].dt.dayofweek
    return pd.get_dummies(df, columns=["曜日"], prefix="曜日", drop_first=False)

# 予測関数（販売金額が目的）
def forecast(df, group_col, periods=7):
    df = add_weekday_dummies(df)
    df["販売金額"] = clean_numeric_column(df, "販売金額")
    df["日付"] = pd.to_datetime(df["日付"])
    target_dates = [df["日付"].max() + timedelta(days=i+1) for i in range(periods)]

    results = {}
    for key, group in df.groupby(group_col):
        X = group[[c for c in group.columns if c.startswith("曜日_")]]
        y = group["販売金額"]
        if len(X) < 10: continue
        model = lgb.LGBMRegressor()
        model.fit(X, y)

        last_weekday = group["日付"].max().dayofweek
        future_weekdays = [(last_weekday + i + 1) % 7 for i in range(periods)]
        future_df = pd.DataFrame(pd.get_dummies(future_weekdays, prefix="曜日"))
        for col in X.columns:
            if col not in future_df:
                future_df[col] = 0
        future_df = future_df[X.columns]

        preds = model.predict(future_df)
        results[key] = dict(zip([str(d.date()) for d in target_dates], map(lambda x: round(x, 1), preds)))

    return results

@app.post("/report")
async def generate_weekly_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        df["販売金額"] = clean_numeric_column(df, "販売金額")
        df["販売数量"] = clean_numeric_column(df, "販売数量")
        dir_path = prepare_graph_dir()
        graphs = generate_graphs(df, dir_path)

        prompt = create_prompt(df, graphs)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは熟練のデータアナリストです。"},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content
        markdown_images = "\n\n".join([f"**{g['title']}**\n\n![]({g['url']})" for g in graphs])
        full_text = f"{summary}\n\n---\n\n{markdown_images}"

        return JSONResponse(content={"html": full_text.replace("\n", "<br>")})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict_sales(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        df["販売金額"] = clean_numeric_column(df, "販売金額")

        product_preds = forecast(df, "商品名")
        category_preds = forecast(df, "カテゴリ")
        date_preds = forecast(df.groupby("日付").agg({"販売金額": "sum"}).reset_index(), "日付")

        return {
            "product_forecast": product_preds,
            "category_forecast": category_preds,
            "date_forecast": date_preds
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
