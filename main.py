import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import asyncio
import aiohttp
from multiprocessing import Pool


def calculate_moving_average(df, window=30):
    df = df.sort_values("timestamp")
    df["temp_ma"] = df["temperature"].rolling(window=window, min_periods=1).mean()
    return df


def calc_season_stats(args):
    season, df = args
    mean_temp = df["temperature"].mean()
    std_temp = df["temperature"].std()
    return season, mean_temp, std_temp


def seasonal_statistics(df):
    groups = df.groupby("season")
    start_seq = time.time()
    stats_seq = {}
    for season, group in groups:
        mean_temp = group["temperature"].mean()
        std_temp = group["temperature"].std()
        stats_seq[season] = {"mean": mean_temp, "std": std_temp}
    time_seq = time.time() - start_seq
    start_par = time.time()
    with Pool(processes=4) as pool:
        args = [(season, group) for season, group in groups]
        results = pool.map(calc_season_stats, args)
    stats_par = {season: {"mean": mean, "std": std} for season, mean, std in results}
    time_par = time.time() - start_par
    return stats_seq, time_seq, stats_par, time_par


def detect_anomalies(df, stats):
    df["anomaly"] = df.apply(
        lambda row: (row["temperature"] < stats[row["season"]]["mean"] - 2 * stats[row["season"]]["std"]) or (
                row["temperature"] > stats[row["season"]]["mean"] + 2 * stats[row["season"]]["std"]), axis=1)
    return df


def get_current_temp_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()


async def get_current_temp_async(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()


st.title("Анализ температурных данных и мониторинг через OpenWeatherMap API")
st.sidebar.header("Загрузка данных и настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл с данными", type="csv")
if not uploaded_file:
    st.info("Пожалуйста, загрузите файл 'temperature_data.csv'.")
else:
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    st.subheader("Общие данные (все города, первые 5 строк)")
    st.write(df.head())
    cities = df["city"].unique().tolist()
    selected_city = st.sidebar.selectbox("Выберите город", cities)
    df_city = df[df["city"] == selected_city].copy()
    st.subheader(f"Данные для города: {selected_city} (первые 5 строк)")
    st.write(df_city.head())
    df_city = calculate_moving_average(df_city)
    stats_seq, time_seq, stats_par, time_par = seasonal_statistics(df_city)
    st.subheader("Сезонная статистика (последовательный расчёт)")
    st.write(stats_seq)
    st.write(f"Время последовательного расчёта: {time_seq:.4f} сек")
    st.subheader("Сезонная статистика (параллельный расчёт)")
    st.write(stats_par)
    st.write(f"Время параллельного расчёта: {time_par:.4f} сек")
    stats = stats_seq
    df_city = detect_anomalies(df_city, stats)
    st.subheader("Временной ряд температур с аномалиями")
    import plotly.express as px

    fig = px.scatter(df_city, x="timestamp", y="temperature", color="anomaly",
                     title=f"Температура в {selected_city} (выделены аномалии)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Сезонные профили")
    seasons = list(stats.keys())
    means = [stats[s]["mean"] for s in seasons]
    stds = [stats[s]["std"] for s in seasons]
    fig2 = px.bar(x=seasons, y=means, error_y=stds, labels={'x': "Сезон", 'y': "Температура (°C)"},
                  title="Средняя температура и стандартное отклонение по сезонам")
    st.plotly_chart(fig2, use_container_width=True)
    st.sidebar.header("Настройки OpenWeatherMap API")
    api_key = st.sidebar.text_input("Введите API ключ OpenWeatherMap", type="password")
    request_method = st.sidebar.radio("Метод запроса к API", ("Синхронный", "Асинхронный"))
    if api_key:
        st.subheader("Текущая температура по API")
        if request_method == "Синхронный":
            current_data = get_current_temp_sync(selected_city, api_key)
        else:
            current_data = asyncio.run(get_current_temp_async(selected_city, api_key))
        if current_data.get("cod") != 200:
            st.error(f"Ошибка API: {current_data.get('message')}")
        else:
            current_temp = current_data["main"]["temp"]
            st.write(f"Текущая температура в {selected_city}: {current_temp}°C")
            current_month = pd.Timestamp.now().month
            month_to_season = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
            current_season = month_to_season[current_month]
            season_mean = stats.get(current_season, {}).get("mean")
            season_std = stats.get(current_season, {}).get("std")
            if season_mean is not None and season_std is not None:
                lower_bound = season_mean - 2 * season_std
                upper_bound = season_mean + 2 * season_std
                st.write(f"Исторический диапазон для {current_season}: {lower_bound:.2f}°C – {upper_bound:.2f}°C")
                if current_temp < lower_bound or current_temp > upper_bound:
                    st.warning("Текущая температура аномальна для данного сезона!")
                else:
                    st.success("Текущая температура в норме для данного сезона.")
            else:
                st.info("Нет данных для определения сезонного диапазона.")
