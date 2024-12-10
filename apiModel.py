import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Set up a cached session with retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_historical_data(latitude, longitude, start_date, end_date):
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,  # "2019-01-01"
        "end_date": end_date,    #"2024-01-01"
        "hourly": "temperature_2m"
    }
    
    try:
        # Make the API call
        responses = openmeteo.weather_api(url, params=params)
        
        # Check if the response is valid
        if not responses:
            raise ValueError("No responses from the API")

        # Process the first response
        response = responses[0]
        hourly = response.Hourly()

        # Get the temperature values and timestamps
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        timestamps = pd.to_datetime(hourly.Time(), unit="s", utc=True)

        # Create a DataFrame of hourly data with timestamps and temperature
        hourly_dataframe = pd.DataFrame({
            "date": timestamps,
            "temperature_2m": hourly_temperature_2m
        })

        # Return the DataFrame
        return hourly_dataframe

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
