import sys
from lstm import train_and_predict

def run_cli():
    weather_data_path = "weather2024.csv"
    start_date = input("Enter the date for the week you want to predict (YYYY-MM-DD): ")

    try:
        predictions = train_and_predict(weather_data_path, start_date)
        print("\nPredicted Hourly Temperatures for the Week:")
        for daily in predictions:
            print(f"{daily['date']}:")
            for hour_data in daily["temperatures"]:
                print(f"  Hour {hour_data['hour']:02}: {hour_data['temperature']:.2f}°F")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_cli()
