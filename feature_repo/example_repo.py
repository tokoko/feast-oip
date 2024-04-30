from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64


driver = Entity(name="driver", join_keys=["driver_id"])
customer = Entity(name="customer", join_keys=["customer_id"])

driver_stats_source = FileSource(
    name="/home/projects/feast-oip/data/driver_hourly_stats.parquet",
    path="/home/projects/feast-oip/data/driver_hourly_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

customer_profile_source = FileSource(
    name="/home/projects/feast-oip/data/customer_daily_profile.parquet",
    path="/home/projects/feast-oip/data/customer_daily_profile.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created"
)

driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64, description="Average daily trips"),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "driver_performance"},
)

customer_daily_profile_fv = FeatureView(
    name="customer_daily_profile",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="current_balance", dtype=Float32),
        Field(name="avg_passenger_count", dtype=Float32),
        Field(name="lifetime_trip_count", dtype=Int64),
    ],
    online=True,
    source=customer_profile_source,
)

driver_success_service = FeatureService(
    name="driver_success",
    features=[
        driver_stats_fv[
            ["conv_rate", "acc_rate", "avg_daily_trips"]
        ],  # Sub-selects a feature from a feature view
        customer_daily_profile_fv[
            ["avg_passenger_count", "lifetime_trip_count"]
        ]
        # transformed_conv_rate,  # Selects all features from the feature view
    ],
)


# driver_activity_v2 = FeatureService(
#     name="driver_activity_v2", features=[driver_stats_fv, transformed_conv_rate]
# )



# # Define a request data source which encodes features / information only
# # available at request time (e.g. part of the user initiated HTTP request)
# input_request = RequestSource(
#     name="vals_to_add",
#     schema=[
#         Field(name="val_to_add", dtype=Int64),
#         Field(name="val_to_add_2", dtype=Int64),
#     ],
# )


# # Define an on demand feature view which can generate new features based on
# # existing feature views and RequestSource features
# @on_demand_feature_view(
#     sources=[driver_stats_fv, input_request],
#     schema=[
#         Field(name="conv_rate_plus_val1", dtype=Float64),
#         Field(name="conv_rate_plus_val2", dtype=Float64),
#     ],
# )
# def transformed_conv_rate(inputs: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["conv_rate_plus_val1"] = inputs["conv_rate"] + inputs["val_to_add"]
#     df["conv_rate_plus_val2"] = inputs["conv_rate"] + inputs["val_to_add_2"]
#     return df




# # Defines a way to push data (to be available offline, online or both) into Feast.
# driver_stats_push_source = PushSource(
#     name="driver_stats_push_source",
#     batch_source=driver_stats_source,
# )

# # Defines a slightly modified version of the feature view from above, where the source
# # has been changed to the push source. This allows fresh features to be directly pushed
# # to the online store for this feature view.
# driver_stats_fresh_fv = FeatureView(
#     name="driver_hourly_stats_fresh",
#     entities=[driver],
#     ttl=timedelta(days=1),
#     schema=[
#         Field(name="conv_rate", dtype=Float32),
#         Field(name="acc_rate", dtype=Float32),
#         Field(name="avg_daily_trips", dtype=Int64),
#     ],
#     online=True,
#     source=driver_stats_push_source,  # Changed from above
#     tags={"team": "driver_performance"},
# )


# # Define an on demand feature view which can generate new features based on
# # existing feature views and RequestSource features
# @on_demand_feature_view(
#     sources=[driver_stats_fresh_fv, input_request],  # relies on fresh version of FV
#     schema=[
#         Field(name="conv_rate_plus_val1", dtype=Float64),
#         Field(name="conv_rate_plus_val2", dtype=Float64),
#     ],
# )
# def transformed_conv_rate_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["conv_rate_plus_val1"] = inputs["conv_rate"] + inputs["val_to_add"]
#     df["conv_rate_plus_val2"] = inputs["conv_rate"] + inputs["val_to_add_2"]
#     return df


# driver_activity_v3 = FeatureService(
#     name="driver_activity_v3",
#     features=[driver_stats_fresh_fv, transformed_conv_rate_fresh],
# )
