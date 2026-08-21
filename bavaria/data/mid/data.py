import numpy as np

"""
This stage provides some data provided in the MiD 2017 report for Bavaria instead of Munich
"""

def configure(context):
    pass

def execute(context):
    data = {}

    data["car_availability_constraints"] = [
        { "zone": "stadt", "target": 0.89 }, # ländliche Kreise mit Verdichtungsansätzen Changed from 0.47 to 0.89
        { "zone": "landkreis", "target": 0.89 }, # ländliche Kreise mit Verdichtungsansätzen Changed from 0.69 to 0.89
        { "zone": "external", "target": 0.82 }, # Bavaria value
    ]

    data["bicycle_availability_constraints"] = [
        { "zone": "stadt", "target": 0.81 }, # ländlicher Kreis mit Verdichtungsansätzen Changed from 0.87 to 0.81
        { "zone": "landkreis", "target": 0.81 },  # ländlicher Kreis mit Verdichtungsansätzen Changed from 0.87 to 0.81
        { "zone": "external", "target": 0.80 }, # Bavaria value

        { "zone": "stadt", "sex": "male", "target": 0.83 }, # bayern value Changed from 0.85 to 0.83
        { "zone": "stadt", "sex": "female", "target": 0.78 }, # bayern value Changed from 0.82 to 0.78

        { "zone": "stadt", "age": (-np.inf, 17), "target": 0.92 }, # bayern value Changed from 0.92 to 0.92
        { "zone": "stadt", "age": (18, 29), "target": 0.79 }, # bayern value Changed from 0.85 to 0.79
        { "zone": "stadt", "age": (30, 49), "target": 0.87 }, # bayern value Changed from 0.90 to 0.87
        { "zone": "stadt", "age": (50, 64), "target": 0.84 }, # bayern value Changed from 0.87 to 0.84
        { "zone": "stadt", "age": (65, 74), "target": 0.78 }, # bayern value Changed from 0.76 to 0.78
        { "zone": "stadt", "age": (75, np.inf), "target": 0.59 }, # bayern value Changed from 0.57 to 0.59

        # landkreis
        { "zone": "landkreis", "sex": "male", "target": 0.83 }, # bayern value Changed from 0.85 to 0.83
        { "zone": "landkreis", "sex": "female", "target": 0.78 }, # bayern value Changed from 0.82 to 0.78

        { "zone": "landkreis", "age": (-np.inf, 17), "target": 0.92 }, # bayern value Changed from 0.92 to 0.92
        { "zone": "landkreis", "age": (18, 29), "target": 0.79 }, # bayern value Changed from 0.85 to 0.79
        { "zone": "landkreis", "age": (30, 49), "target": 0.87 }, # bayern value Changed from 0.90 to 0.87
        { "zone": "landkreis", "age": (50, 64), "target": 0.84 }, # bayern value Changed from 0.87 to 0.84
        { "zone": "landkreis", "age": (65, 74), "target": 0.78 }, # bayern value Changed from 0.76 to 0.78
        { "zone": "landkreis", "age": (75, np.inf), "target": 0.59 }, # bayern value Changed from 0.57 to 0.59
    ]

    data["pt_subscription_constraints"] = [
        { "zone": "stadt", "target": 0.08 }, # Oberfranken Changed from 0.35 to 0.08
        { "zone": "landkreis", "target": 0.08 }, # Oberfranken Changed from 0.35 to 0.08
        { "zone": "external", "target": 0.17 }, # Bavaria value

        { "zone": "stadt", "sex": "male", "target": 0.23 }, # München Umland Changed from 0.46 to 0.23
        { "zone": "stadt", "sex": "female", "target": 0.21 }, # München Umland Changed from 0.50 to 0.21

        { "zone": "stadt", "age": (-np.inf, 17), "target": 0.49 }, # bayern value Changed from 0.49 to 0.49
        { "zone": "stadt", "age": (18, 29), "target": 0.32 }, # bayern value Changed from 0.65 to 0.32
        { "zone": "stadt", "age": (30, 49), "target": 0.15 }, # bayern value Changed from 0.48 to 0.15
        { "zone": "stadt", "age": (50, 64), "target": 0.12 }, # bayern value Changed from 0.40 to 0.12
        { "zone": "stadt", "age": (65, 74), "target": 0.09 }, # bayern value Changed from 0.37 to 0.09
        { "zone": "stadt", "age": (75, np.inf), "target": 0.09 }, # bayern value Changed from 0.34 to 0.09

        # landkreis
        { "zone": "landkreis", "sex": "male", "target": 0.23 }, # München Umland value stays the same
        { "zone": "landkreis", "sex": "female", "target": 0.21 }, # München Umland value stays the same
        
        { "zone": "landkreis", "age": (-np.inf, 17), "target": 0.49 }, # changed from 0,41 to 0,49
        { "zone": "landkreis", "age": (18, 29), "target": 0.32 }, # changed from 0,39 to 0,32
        { "zone": "landkreis", "age": (30, 49), "target": 0.15 }, # changed from 0,22 to 0,15
        { "zone": "landkreis", "age": (50, 64), "target": 0.12 }, # changed from 0,20 to 0,12
        { "zone": "landkreis", "age": (65, 74), "target": 0.09 }, # changed from 0,11 to 0,09
        { "zone": "landkreis", "age": (75, np.inf), "target": 0.09 }, # changed from 0,11 to 0,09
    ]

    return data

