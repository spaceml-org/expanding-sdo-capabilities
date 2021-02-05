""""
"""
import pandas as pd
import datetime as dt


def split_time(str_time):
    yr, mth, rest = str_time.split('-')
    day, time = rest.split('T')
    hour, minus, _ = time.split(':')
    return {'y': yr, 'mt': mth, 'd': day, 'h': hour, 'm': minus}


def from_row_to_date(x):
    l_date = [x['year'], x['month'], x['day'], x['hour'], x['min'], 0]
    return dt.datetime(*map(int, l_date))


def nearest(items, pivot):
    def convert_nptime_to_datetime(x):
        return dt.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')
    return min(items, key=lambda x: abs(convert_nptime_to_datetime(x) - pivot))


def find_closest_datetimes(start_time, end_time, sel_df):
    nearest_start_time = nearest(sel_df['date'].tolist(), start_time)
    # this is imperfect, we should look into the month, in case the event happens close to midnight
    nearest_end_time = nearest(sel_df['date'].tolist(), end_time)
    return nearest_start_time, nearest_end_time


def get_datetime(time, buffer_h, buffer_m):
    d_time = split_time(time)
    time = dt.datetime(*map(int, list(d_time.values())))
    time = time - dt.timedelta(hours=buffer_h, minutes=buffer_m)
    return time


def select_images_in_the_interval(start_time, end_time, df_inventory, buffer_h=1, buffer_m=0):
    first_datetime = get_datetime(start_time, buffer_h, buffer_m)
    last_datetime = get_datetime(end_time, buffer_h, buffer_m)
    # select all the times of that day
    sel_df = df_inventory[(df_inventory.year == first_datetime.year)
                        & (df_inventory.month == first_datetime.month)
                        & (df_inventory.day == first_datetime.day)]
    if sel_df.shape[0] > 0:
        # this operation is slow if applied to the full dataframe
        sel_df['date'] = sel_df.apply(from_row_to_date, axis=1)
        first_datetime, last_datetime = find_closest_datetimes(first_datetime, last_datetime, sel_df)
        sel_df = sel_df[(sel_df.date >= first_datetime) & (sel_df.date <= last_datetime)]
        return sel_df
    else:
        print('This day is not available')
        return pd.DataFrame(columns=df_inventory.columns)