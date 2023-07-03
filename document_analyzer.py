import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
from PyPDF2 import PdfReader

# AWSの認証情報を設定する
session = boto3.Session(
    #aws_access_key_id="AWS_ACCESS_KEY_ID",
    #aws_secret_access_key="AWS_SECRET_ACCESS_KEY",
    region_name="ap-northeast-1"
)

# Comprehendクライアントを作成する
comprehend = session.client("comprehend")

# ファイルからテキストデータを取得する関数
def get_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    # ページ数の取得
    num_pages = len(pdf_reader.pages)
    text = ""
    # ページごとにテキストを抽出して表示
    for page in range(num_pages):
        pdf_page = pdf_reader.pages[page]
        page_text = pdf_page.extract_text()
        text += page_text
    return text

# テキストデータをキーワード解析して、出現頻度を計算する関数
def analyze_text(text):
    response = comprehend.detect_key_phrases(Text=text, LanguageCode="ja")
    keywords = [kp["Text"] for kp in response["KeyPhrases"]]
    freq = pd.Series(keywords).value_counts().reset_index()
    freq.columns = ["keyword", "count"]
    return freq

# テキストデータを感情分析する関数
def analyze_sentiment(text):
    response = comprehend.detect_sentiment(Text=text, LanguageCode="ja")
    sentiment = response["Sentiment"]
    return sentiment

# Streamlitアプリケーションの設定
st.set_page_config(page_title="Comprehend Analyzer", page_icon=":sunglasses:")

st.title("Comprehend Analyzer")

# PDFファイルのアップロード
file = st.file_uploader("PDFファイルをアップロードしてください", type=['pdf'])

if file is not None:
    # ファイルからテキストデータを取得
    text = get_text_from_pdf(file)

    # キーワード解析して出現頻度を計算
    freq = analyze_text(text)

    # 感情分析を実行
    sentiment = analyze_sentiment(text)
    # 結果を表示
    #st.write("キーワードの出現頻度：")
    #st.write(freq)

    st.write("感情分析の結果：", sentiment)

    # グラフを表示
    fig1 = px.bar(freq, x="keyword", y="count", title="Keyword Frequency")
    st.plotly_chart(fig1)

    #例えば文章毎にセンチメント出してパイチャートで出すのも面白いかもしれない
    # fig2 = px.pie(values=[1, 0, 0, 0, 0], names=["Positive", "Negative", "Neutral", "Mixed", "Error"], title="Sentiment Analysis")
    # fig2.update_traces(values=[int(sentiment == "POSITIVE"), int(sentiment == "NEGATIVE"), int(sentiment == "NEUTRAL"), int(sentiment == "MIXED"), int(sentiment == "ERROR")])
    # st.plotly_chart(fig2)


    st.write(text)