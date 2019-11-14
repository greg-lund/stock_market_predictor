import requests
import json
import csv

api_key = "P998ZOO8W2LFAJ6R"
URL = "https://www.alphavantage.co/query"
file_out = "./testing.csv"

PARAMS = {
        'function': "TIME_SERIES_DAILY",
        'symbol': "WMT",
        'datatype': "csv",
        'outputsize':"full",
        'apikey': api_key
        }

response = requests.get(url=URL, params=PARAMS)
f = open(file_out,'w')
f.write(response.text)
f.close()
