""""
"""
import pandas as pd
import datetime as dt
from collections import OrderedDict
import logging

_logger = logging.getLogger(__name__)


def split_time(str_time):
    yr, mth, rest = str_time.split('-')
    day, time = rest.split('T')
    hour, minute, _ = time.split(':')
    return OrderedDict([('y',yr), ('mt',mth), ('d',day), ('h',hour), ('m', minute)])


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


def get_datetime(time, buffer_h, buffer_m, add=False):
    """
    Function that convert string into datetime and adds/subtract time to it
    Args:
        time (str):  str in the format 2010-06-12T00:30:00
        buffer_h (int): hours to be subtracted/added
        buffer_m (int): minutes to be subtracted/added
        add (bool): if True time is added. If False is subtracted.

    Returns: (dt.datetime)

    """
    d_time = split_time(time)
    time = dt.datetime(*map(int, list(d_time.values())))
    if add:
        time = time + dt.timedelta(hours=buffer_h, minutes=buffer_m)
    else:
        time = time - dt.timedelta(hours=buffer_h, minutes=buffer_m)
    return time


def select_images_in_the_interval(first_datetime, last_datetime, df_inventory):
    """
    Function to select all the rows in df_inventory that have events happened after
    first_datetime and before last_datetime.
    Args:
        first_datetime (dt.datetime): earliest datetime to be included
        last_datetime (dt.datetime): most recent datetime to be included
        df_inventory (pd.DataFrame): dataframe that contains available datetimes.
            It's assumed to have the cols: 'year', 'month', 'day', 'hour', 'min'

    Returns: pd.DataFrame. It returns an empty dataframe if the closest row is more than
        one day apart from first_datetime.

    """
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
        _logger.warning('This day %s-%s-%s is not available' % (first_datetime.year, 
                                                                 first_datetime.month,
                                                                 first_datetime.day))
        return pd.DataFrame(columns=df_inventory.columns)