from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import aiofiles
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.font_manager as fm

# 環境変数をロード
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 日本語フォントの指定（Render用）
font_path = "fonts/ipaexg.ttf"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# ベースURL（Dify対応用に完全URLを生成）
BASE_URL = "https://delica-insight-bot.onrender.com"

# ユーティリティ：数値列の前処理
def clean_numeric_column(df, col):
    return df[col].replace(",", "", regex=True).astype(float)

# ユーティリティ：グラフ保存ディレクトリの準備
def prepare_graph_dir():
    dir_path = "static/graphs"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# 共通：フォント適用関数
def apply_font(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    ax.title.set_fontproperties(font_prop)
    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)

# グラフ生成処理
def generate_graphs(df: pd.DataFrame, dir_path: str):
    plt.rcParams["font.family"] = font_prop.get_name()
    sns.set(style="whitegrid")
    urls = []

    # グラフ1: 日別売上金額推移
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

    # グラフ2: カテゴリ別売上構成比
    fig, ax = plt.subplots(figsize=(6, 6))
    cat = df.groupby("カテゴリ")["販売金額"].sum()
    ax.pie(cat, labels=cat.index, autopct='%1.1f%%', startangle=140)
    ax.set_title("カテゴリ別売上構成比")
    path = f"{dir_path}/graph2_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "カテゴリ別売上構成", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    # グラフ3: 商品別販売数量ランキング（Top10）
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

    # グラフ4: 値引き率の分布
    fig, ax = plt.subplots(figsize=(8, 4))
    discount = df["値引き率"].str.replace("%", "").astype(float)
    sns.histplot(discount, bins=10, kde=True, ax=ax)
    ax.set_title("値引き率の分布")
    apply_font(ax)
    path = f"{dir_path}/graph4_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "値引き率の分布", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    # グラフ5: 廃棄率 vs 値引き率（散布図）
    fig, ax = plt.subplots(figsize=(6, 6))
    discount = df["値引き率"].str.replace("%", "").astype(float)
    waste = df["廃棄率"].str.replace("%", "").astype(float)
    sns.scatterplot(x=discount, y=waste, ax=ax)
    ax.set_title("廃棄率 vs 値引き率")
    ax.set_xlabel("値引き率（%）")
    ax.set_ylabel("廃棄率（%）")
    apply_font(ax)
    path = f"{dir_path}/graph5_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "廃棄率 vs 値引き率", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    return urls

# GPT用プロンプト生成
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

# 週次レポートエンドポイント
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
        return JSONResponse(content={"graphs": graphs, "summary": summary})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
