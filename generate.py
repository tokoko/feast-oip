from feast.driver_test_data import create_orders_df, create_customer_daily_profile_df, create_driver_hourly_stats_df
from datetime import datetime
import os

drivers = [101, 102]
customers = [1, 2, 3]
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 10)

driver_df = create_driver_hourly_stats_df(drivers=drivers, start_date=start_date, end_date=end_date)
customer_df = create_customer_daily_profile_df(customers=customers, start_date=start_date, end_date=end_date)
order_df = create_orders_df(customers=customers, drivers=drivers, start_date=start_date, end_date=end_date, order_count=10000)

if not os.path.exists('data'):
    os.mkdir('data')

driver_df.to_parquet(os.path.join('data', 'driver_hourly_stats.parquet'))
customer_df.to_parquet(os.path.join('data', 'customer_daily_profile.parquet'))
order_df.to_parquet(os.path.join('data', 'orders.parquet'))




