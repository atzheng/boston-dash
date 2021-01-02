from dateutil.parser import parse
from datetime import datetime, timezone
import json
import operator as op
import os
import requests

import altair as alt
import funcy as f
import numpy as np
import pandas as pd
from scipy.stats import skellam
import streamlit as st
st.set_page_config(layout="wide")
import yaml


# UTILS
# -------------------------------------------------------------------------
def relative_ts(ts):
   return int((parse(ts) - datetime.now(timezone.utc)).total_seconds() / 60)


def vectorize(fun):
    def vectorized(l):
        return f.lmap(fun, l)
    return vectorized


def format_pct(p):
    return "{P}%".format(P=round(p * 100))


def pad(x, length, fill=None):
    return (x + [fill] * (length - len(x)))[:length]
   


# BLUEBIKES
# -------------------------------------------------------------------------
def p_stockout(arr_rate, dep_rate, stock, T):
    return skellam.cdf(-stock,
                       max(T * arr_rate, 1e-5),
                       max(T * dep_rate, 1e-5))


@st.cache
def load_bluebikes_static_data():
    return pd.read_csv("rates.csv"), pd.read_csv("stations.csv")


@st.cache(ttl=60)
def fetch_bluebikes_status():
    endpoint = 'https://gbfs.bluebikes.com/gbfs/en/station_status.json'
    response = requests.get(endpoint)
    return {
        int(x['station_id']): {
            "bikes": x['num_bikes_available'],
            "docks": x['num_docks_available']
        }
        for x in json.loads(response.text)['data']['stations']
    }


def get_bluebikes_station_info(station_id):
    rates, stations = load_bluebikes_static_data()
    stock = fetch_bluebikes_status()[station_id]

    name = stations[stations['station_id'] == station_id]["name"].values[0]
    subdf = rates[(rates["station_id"] == station_id) &
                  (rates['hour'] == datetime.now().hour) &
                  (rates['wday'] == datetime.now().isoweekday())]

    arr, dep = (subdf['arrival'].values[0], subdf['departure'].values[0])
    times = [0.25, 0.5, 1]
    bikeout_probs = [
        format_pct(p_stockout(arr, dep, stock['bikes'], T)) for T in times]
    dockout_probs = [
        format_pct(p_stockout(dep, arr, stock['docks'], T)) for T in times]

    return {
        "station": name,
        "bikes": stock['bikes'],
        # 'bike wait': "%d" % round(60 / arr),
        # 'dock wait': "%d" % round(60 / dep),
        "docks": stock['docks'],
        "P(bikeout)": " / ". join(bikeout_probs),
        "P(dockout)": " / ". join(dockout_probs)
    }


def get_bluebikes_info(stations):
    return pd.DataFrame.from_dict(
        f.lmap(get_bluebikes_station_info, stations))


# MBTA
# -------------------------------------------------------------------------
@st.cache(ttl=60)
def query_mbta(endpoint, params):
    response = requests.get(
        os.path.join("https://api-v3.mbta.com/", endpoint + "/"),
        params=params)
    data = json.loads(response.text)['data']
    return f.lmap(op.itemgetter("attributes"), data)


@st.cache
def query_mbta_id(endpoint, id):
    return query_mbta(endpoint, (('filter[id]', id), ))[0]


def get_mbta_station_info(cfg):
    route_info = query_mbta_id("routes", cfg['route'])
    stop_info = query_mbta_id("stops", cfg['stop'])

    params = (('filter[stop]', cfg['stop']),
              ('filter[route]', cfg['route']),
              ('page[limit]', '10'))
    arrivals = query_mbta('predictions', params)
    by_direction = f.walk_values(
        vectorize(f.compose(relative_ts, op.itemgetter('arrival_time'))),
        f.group_by(op.itemgetter('direction_id'), arrivals))

    return [
        f.merge({
            "station": stop_info['name'],
            "route": cfg['route'],
            "direction": route_info['direction_destinations'][k],
        }, dict(zip(range(5), pad(v, 5))))
        for k, v in by_direction.items()]


def get_mbta_info(cfgs):
    return pd.DataFrame.from_dict(f.lcat(map(get_mbta_station_info, cfgs)))


# WEATHER
# -------------------------------------------------------------------------
@st.cache(ttl=60 * 5)
def get_weather_dfs(config):
    response = requests.get("https://api.openweathermap.org/data/2.5/onecall",
                            params=config.items())
    weather = json.loads(response.text)
    hourly = [f.merge(f.omit(hr, 'weather'), hr['weather'][0])
              for hr in weather['hourly']]
    hourly_df = pd.DataFrame(hourly)
    hourly_df['dt_end'] = hourly_df['dt'] + 60 * 60
    hourly_df['dt'] = f.lmap(datetime.fromtimestamp, hourly_df['dt'])
    hourly_df['dt_end'] = f.lmap(datetime.fromtimestamp, hourly_df['dt_end'])
    hourly_df['color'] = f.lmap(get_weather_color, hourly_df['id'])

    current = (
        "Currently {temp}°F, {description}. Wind {wind_speed} MPH"
        .format(temp=round(weather['current']['temp']),
                description=weather['current']['weather'][0]['description'],
                wind_speed=round(weather['current']['wind_speed'])))
    return current, hourly_df


def get_weather_color(id):
    if id >= 200 and id < 300:
        return "#ff3b3b"
    elif id >= 300 and id < 400:
        return "#bad0ff"
    elif id in [500, 520]:
        return "#8ca6de"
    elif id in [501, 521, 531]:
        return "#4777de"
    elif id in [502, 503, 504, 511, 522]:
        return "#1147ba"
    elif id in [600, 615, 620]:
        return "#e3a6ff"
    elif id >= 600 and id < 700:
        return "d06bff"
    elif id == 800:
        return "ffffff"
    elif id == 801:
        return "d9d9d9"
    elif id == 802:
        return "bababa"
    elif id == 803:
        return "9c9c9c"
    elif id == 804:
        return "808080"
    else:
        raise Exception()


def plot_hourly_weather(hourly):
    base = alt.Chart(hourly).encode(x=alt.X('dt:T', axis=alt.Axis(title=None)))
    temp_color = "000000"
    wind_color = "00d2e6"
    temp = (base
            .mark_line(stroke=temp_color)
            .encode(y=alt.Y('temp:Q',
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(title="Temperature (°F)",
                                          titleColor=temp_color))))
    wind = (base
            .mark_line(stroke=wind_color)
            .encode(y=alt.Y('wind_speed:Q',
                            axis=alt.Axis(title="Wind Speed (MPH)",
                                          titleColor=wind_color))))

    precip = (base
              # .mark_line(stroke=wind_color)
              .encode(y=alt.Y('pop',
                              axis=alt.Axis(title=None, labels=False)),
                      color=alt.Color('color', scale=None))
              .mark_area(line=False,
                         interpolate='step-after',
                         opacity=0.5))

    sun = (base.encode(x2='dt_end:T', color=alt.Color('color', scale=None))
           .mark_rect(stroke=None, opacity=0.3))

    sun_text = base.encode(
        text="description").mark_text(opacity=0.15, angle=270, xOffset=8,
                                      y=250, align="left")

    return alt.layer(alt.layer(temp, wind).resolve_scale(y='independent'),
                     precip, sun, sun_text)


# MAIN
# -------------------------------------------------------------------------
with open("config.yaml", "r") as file:
    CONFIG = yaml.load(file)

# st, weather = st.beta_columns(2)
# st.title('Dashboard')


st.header("Weather")
current, hourly = get_weather_dfs(CONFIG['weather'])
st.write(current)
st.altair_chart(plot_hourly_weather(hourly), use_container_width=True)
st.header("Bluebikes")
st.write(get_bluebikes_info(CONFIG['bluebikes']))
st.header("MBTA")
st.write(get_mbta_info(CONFIG['mbta']))
