# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:20:13 2022
author: Guoming Yang
School of Electrical Engineering and Automation
Harbin Institute of Technology
email: yangguoming1995@gmail.com
"""

import requests
#The apikey you got after signing up in "https://developer.nrel.gov/docs/solar/nsrdb/site-count/#request-url". The information needed includes your email.
your_apikey = 'NqyiqdbQxEVIbhSTnSlbmq0xB462ha6YLE6g1BHf'
your_email = '1328332147@qq.com'
#Either 10, 30 or 60 minute intervals are available.
interval = '60'
#Options: 2016, 2017, 2018, 2019, 2020.
names = 'tmy-2020'
#Pass true to retrieve data with timestamps in UTC. Pass false to retrieve data with
#timestamps converted to local time of data point (without daylight savings time).
utc = 'false'
wkt = 'POINT(131.07	45.78)'
url = "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-tmy-download.json?api_key={your_apikey}".format(your_apikey=your_apikey)
payload = "wkt={wkt}&names={names}&utc={utc}&leap_day=true&interval={interval}&email={your_email}".format\
    (your_email=your_email,interval=interval,names=names,wkt=wkt, utc=utc)
headers = {
    'content-type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache"
}
response = requests.request("POST", url, data=payload, headers=headers)
print(response.text)