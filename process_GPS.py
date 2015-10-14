# process GPS collar records from OPC
# This analysis assumes points were taken near the equator, so that decimal
# degrees can be converted to km with a single conversion factor

import os
import re
import math
import time
import pandas

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
    
    veg_data = pandas.read_table(veg_file, sep=',')
    veg_data = veg_data.where((pandas.notnull(veg_data)), None)
    veg_data['Date'] = pandas.to_datetime(veg_data['Date sampled'],
                                          format='%d-%b-%y')
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
            rot = re.search('rot(.+?)_', file).group(1)
            suf = re.search('_(.+?).csv', file).group(1)
            filedir = os.path.join(dir, file)
            df = pandas.read_table(filedir, sep=',')
            df = df.where((pandas.notnull(df)), None)
            for name in df.columns.values:
                if name.strip() != name:
                    df[name.strip()] = df[name]
                    del df[name]
            try:
                df['Date'] = pandas.to_datetime(df['Date'], format='%m/%d/%Y')
            except ValueError:
                df['Date'] = pandas.to_datetime(df['Date'], format='%Y/%m/%d')
            df['rot'] = rot
            df['suf'] = suf
            df_list.append(df)
    GPS_data = pandas.concat(df_list)
    return GPS_data

def find_matched_records(x1, y1, GPS_data):
    """Find GPS records that were recorded within 0.5 km of a point"""
    
    match1 = GPS_data.loc[GPS_data['Longitude'] < (x1 + 0.0044915)]
    match2 = match1.loc[match1['Longitude'] > (x1 - 0.0044915)]
    match3 = match2.loc[match2['Latitude'] < (y1 + 0.0044915)]
    match4 = match3.loc[match3['Latitude'] > (y1 - 0.0044915)]
    return match4

if __name__ == "__main__":
    outerdir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/GPS_data'
    veg_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/veg_2014_metadata.csv"
    result_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/Matched_GPS_records"
    
    veg_data = read_veg_data(veg_file)
    GPS_datafile = os.path.join(outerdir, "data_combined.csv")
    if not os.path.exists(GPS_datafile):
        GPS_data = combine_GPS_files(outerdir)
        GPS_data.to_csv(GPS_datafile)
    else:
        GPS_data = pandas.read_table(GPS_datafile, sep=',')
        GPS_data['Date'] = pandas.to_datetime(GPS_data['Date'],
                                              format='%Y-%m-%d')
    for v_row in xrange(len(veg_data)):
        if veg_data.iloc[v_row].Lat is None or veg_data.iloc[v_row].Long is None:
            continue
        x1 = float(veg_data.iloc[v_row].Long)
        y1 = float(veg_data.iloc[v_row].Lat)
        
        match = find_matched_records(x1, y1, GPS_data)
        if len(match) > 0:
            property = veg_data.iloc[v_row].Property
            trans = veg_data.iloc[v_row].Transect
            filename = 'matched_0.5km_%s_transect_%s.csv' % (property, trans)
            match.to_csv(os.path.join(result_dir, filename))