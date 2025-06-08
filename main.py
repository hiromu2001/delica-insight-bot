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
    path = f"{dir_path}/graph5_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(path)
    urls.append({"title": "廃棄率 vs 値引き率", "url": f"{BASE_URL}/static/graphs/{os.path.basename(path)}"})
    plt.close()

    return urls

# GPTに渡す要約プロンプト生成（変更なし）
# ...以下略...
