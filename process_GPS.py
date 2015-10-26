# process GPS collar records from OPC
# This analysis assumes points were taken near the equator, so that decimal
# degrees can be converted to km with a single conversion factor

import os
import re
import math
import time
import pandas as pd
from pandas.tseries.offsets import *

def euclidean_dist(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points supplied as lat, long
    pairs.  The points are assumed to be in geographic coordinates and the
    distance is returned in km.  Note this function assumes we are near the
    equator."""
    
    a = abs(float(x1) - float(x2))
    b = abs(float(y1) - float(y2))
    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2))
    km = c * 111.32
    return km
    
def read_veg_data(veg_file):
    """Make data frame of veg transect points, including date, lat, and long"""
    
    veg_data = pd.read_table(veg_file, sep=',')
    veg_data = veg_data.where((pd.notnull(veg_data)), None)
    # veg_data['Date'] = pd.to_datetime(veg_data['Date sampled'],
                                          # format='%d-%b-%y')
    return veg_data

def combine_GPS_files(outerdir):
    """Combine GPS records from multiple files into one data frame that can be
    queried by date and lat/long location."""
    
    folders = [f for f in os.listdir(outerdir) if
                                      os.path.isdir(os.path.join(outerdir, f))]
    df_list = []
    for folder in folders:
        dir = os.path.join(outerdir, folder)
        files = [f for f in os.listdir(dir) if re.search('.csv$', f)]
        for file in files:
            rot = str(re.search('rot(.+?)_', file).group(1))
            suf = re.search('_(.+?).csv', file).group(1)
            abbrev = suf[:-2]
            unit_no = suf[-2:]
            filedir = os.path.join(dir, file)
            df = pd.read_table(filedir, sep=',')
            df = df.where((pd.notnull(df)), None)
            for name in df.columns.values:
                if name.strip() != name:
                    df[name.strip()] = df[name]
                    del df[name]
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            except ValueError:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
            df['rot'] = rot
            df['suf'] = suf
            df['abbrev'] = abbrev
            df['unit_no'] = unit_no
            df_list.append(df)
    GPS_data = pd.concat(df_list)
    return GPS_data

def find_matched_records(x1, y1, GPS_data, distance):
    """Find GPS records that were recorded within the specified distance (in
    km) of a point.  Assumes that coordinates are given in latitude and
    longitude and are near the equator."""
    
    xdist = float(distance) / 111.321
    ydist = float(distance) / 110.567
    match1 = GPS_data.loc[GPS_data['Longitude'] < (x1 + xdist)]
    match2 = match1.loc[match1['Longitude'] > (x1 - xdist)]
    match3 = match2.loc[match2['Latitude'] < (y1 + ydist)]
    match4 = match3.loc[match3['Latitude'] > (y1 - ydist)]
    return match4

def match_to_points(points_file, outerdir, result_dir, distance):
    points_data = read_veg_data(points_file)
    GPS_datafile = os.path.join(outerdir, "data_combined.csv")
    if not os.path.exists(GPS_datafile):
        GPS_data = combine_GPS_files(outerdir)
        GPS_data.to_csv(GPS_datafile)
    else:
        GPS_data = pd.read_table(GPS_datafile, sep=',')
        GPS_data['Date'] = pd.to_datetime(GPS_data['Date'],
                                              format='%Y-%m-%d')
    for v_row in xrange(len(points_data)):
        if points_data.iloc[v_row].POINT_X is None or \
                                points_data.iloc[v_row].POINT_Y is None:
            continue
        # x1 = float(points_data.iloc[v_row].Long)
        # y1 = float(points_data.iloc[v_row].Lat)
        x1 = float(points_data.iloc[v_row].POINT_X)
        y1 = float(points_data.iloc[v_row].POINT_Y)
        
        match = find_matched_records(x1, y1, GPS_data, distance)
        if len(match) > 0:
            # property = points_data.iloc[v_row].Property
            # trans = points_data.iloc[v_row].Transect
            station = points_data.iloc[v_row].Name
            # filename = 'matched_0.5km_%s_transect_%s.csv' % (property, trans)
            filename = 'matched_station_%s_%dkm.csv' % (station, distance)
            match.to_csv(os.path.join(result_dir, filename))

def summarize_density(points_file, gps_metadata_file, outerdir, result_dir,
                      distance):
    """Calculate the density of animals within a given distance of points. For
    each point, first find movement records collected near that point. Then
    filter records: what herds were there each day. Retrieve number of animals
    (of each type) that were there each day. Then summarize number of each
    animal type by month by averaging daily records.
    
    Distance must be given in km. Points must be given in geographic
    coordinates and must be near the equator."""
    
    metadata = pd.read_table(gps_metadata_file, sep=',')
    metadata = metadata.where((pd.notnull(metadata)), None)
    metadata.Rotation = metadata.Rotation.astype(str)
    metadata['Date'] = pd.to_datetime(metadata['Date'],
                              format='%d.%m.%y')
    points_data = read_veg_data(points_file)
    GPS_datafile = os.path.join(outerdir, "data_combined.csv")
    if not os.path.exists(GPS_datafile):
        GPS_data = combine_GPS_files(outerdir)
        GPS_data.to_csv(GPS_datafile)
    else:
        GPS_data = pd.read_table(GPS_datafile, sep=',')
        GPS_data.rot.astype(str)
        GPS_data['Date'] = pd.to_datetime(GPS_data['Date'],
                                              format='%Y-%m-%d')
    for v_row in xrange(len(points_data)):
        # if points_data.iloc[v_row].POINT_X is None or \
                                # points_data.iloc[v_row].POINT_Y is None:
        if points_data.iloc[v_row].Long is None or \
                                points_data.iloc[v_row].Lat is None:
            continue
        # point_name = points_data.iloc[v_row].Name
        # x1 = float(points_data.iloc[v_row].POINT_X)
        # y1 = float(points_data.iloc[v_row].POINT_Y)
        point_name = '%s-%s' % (points_data.iloc[v_row].Property,
                                points_data.iloc[v_row].Transect)
        x1 = float(points_data.iloc[v_row].Long)
        y1 = float(points_data.iloc[v_row].Lat)
        
        match = find_matched_records(x1, y1, GPS_data, distance)
        if len(match) > 0:
            df_list = []
            for date in match["Date"].map(lambda t: t.date()).unique():
                herd_dict = {'date': [],
                             'unit': [],
                             'Bulls': [],
                             'Cows': [],
                             'Steers': [],
                             'Heifers': [],
                             'steer/heifer': [],
                             'Weaners': [],
                             'Calves': [],
                }
                year = pd.to_datetime(date).year
                one_day = match.loc[match['Date'] == date]  # for each day where a herd was recorded nearby
                for unit in pd.unique(one_day.unit_no.ravel()):  # for each unit recorded in that day
                    one_unit = one_day.loc[one_day['unit_no'] == unit]
                    for abb in pd.unique(one_unit.abbrev.ravel()):
                        rotation = str(pd.unique(one_unit.rot.ravel())[0])
                        record1 = metadata.loc[metadata['Abbrv'] == abb]
                        record2 = record1.loc[record1['Unit_no'] == unit]
                        record3 = record2.loc[record2['Rotation']\
                                                              == rotation]
                        if len(record3) > 1:
                            record4 = record3.loc[record3['Year'] == year]
                            record = record4
                        else:
                            record = record3
                        assert len(record) <= 1
                        if len(record) > 0:
                            herd_dict['date'].append(date)
                            herd_dict['unit'].append(unit)
                            herd_dict['Bulls'].append(record.iloc[0].Bulls)
                            herd_dict['Cows'].append(record.iloc[0].Cows)
                            herd_dict['Steers'].append(record.iloc[0].Steers)
                            herd_dict['Heifers'].append(record.iloc[0].
                                                        Heifers)
                            herd_dict['steer/heifer'].append(
                                               record.iloc[0]['steer/heifer'])
                            herd_dict['Weaners'].append(record.iloc[0].
                                                        Weaners)
                            herd_dict['Calves'].append(record.iloc[0].Calves)
                date_df = pd.DataFrame(herd_dict)
                # there may have been >1 recorder for one herd; check that we
                # only count each herd once
                date_df = date_df.drop_duplicates(['Calves', 'Cows', 'Steers'])
                df_list.append(date_df)
            matched_df = pd.concat(df_list)
            filename = 'herds_%s_%dkm.csv' % (point_name, distance)
            matched_df.to_csv(os.path.join(result_dir, filename))
            
            # summarize number of each animal type by month
            df = matched_df
            df = df.where((pd.notnull(df)), None)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df.sort(['date'])
            first = df.iloc[0].date
            start = pd.datetime(first.year, first.month, 1)
            # if the first record in the first month was from the first 10 days
            # in the month, we include that month in our summary
            if first.day > 10:
                start = start + DateOffset(months=1)
            last = df.iloc[-1].date
            end = pd.datetime(last.year, last.month, 1)  # end is INCLUSIVE
            if last.day < 20:
                end = end - DateOffset(months=1)   
            df_i = df.set_index('date')
            ave_dict = {'month': [],
                        'year': [],
                        'Bulls': [],
                        'Calves': [],
                        'Cows': [],
                        'Heifers': [],
                        'Steers': [],
                        'Weaners': [],
                        'steer/heifer': [],
            }
            time = start
            while time <= end:
                records = df_i[((df_i.index.month == time.month) &
                                (df_i.index.year == time.year))]
                ave_dict['month'].append(time.month)
                ave_dict['year'].append(time.year)
                ave_dict['Bulls'].append(records[['Bulls']].mean()[0])
                ave_dict['Calves'].append(records[['Calves']].mean()[0])
                ave_dict['Cows'].append(records[['Cows']].mean()[0])
                ave_dict['Heifers'].append(records[['Heifers']].mean()[0])
                ave_dict['Steers'].append(records[['Steers']].mean()[0])
                ave_dict['Weaners'].append(records[['Weaners']].mean()[0])
                ave_dict['steer/heifer'].append(records[[
                                                    'steer/heifer']].mean()[0])
                time = time + DateOffset(months=1)
            ave_df = pd.DataFrame(ave_dict)
            filename = 'average_animals_%s_%dkm.csv' % (point_name, distance)
            ave_df.to_csv(os.path.join(result_dir, filename))

if __name__ == "__main__":
    outerdir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/GPS_data'
    veg_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/veg_2014_metadata.csv"
    result_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/Matched_GPS_records/Matched_with_weather_stations"
    weather_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/Climate/OPC_weather_stations_coordinates.csv"
    gps_metadata_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/GPS_data/GPS_metadata.csv"
    veg_result_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/Matched_GPS_records/Matched with veg transects"
    distance = 2.
    summarize_density(veg_file, gps_metadata_file, outerdir, veg_result_dir,
                      distance)
