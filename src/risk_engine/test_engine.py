from fusion import fuse_risks
from alerts import (
    classify_risk,
    get_alert_message,
    apply_escalation
)


print("=" * 50)
print("TEST 1: TEMPORAL ONLY")
print("=" * 50)

risk, contributions = fuse_risks(
    temporal_risk=0.72
)

print("Final Risk:", round(risk, 3))
print("Contributions:", contributions)


print("\n" + "=" * 50)
print("TEST 2: SPATIAL + TEMPORAL")
print("=" * 50)

risk, contributions = fuse_risks(
    spatial_risk=0.81,
    temporal_risk=0.72
)

print("Final Risk:", round(risk, 3))
print("Contributions:", contributions)


print("\n" + "=" * 50)
print("TEST 3: ALL MODELS")
print("=" * 50)

risk, contributions = fuse_risks(
    spatial_risk=0.81,
    temporal_risk=0.72,
    vision_risk=0.65
)

print("Final Risk:", round(risk, 3))
print("Contributions:", contributions)
print("\n" + "=" * 50)
print("TEST 4: ALERT CLASSIFICATION")
print("=" * 50)

test_risks = [
    0.10,
    0.35,
    0.65,
    0.90
]

for score in test_risks:

    level = classify_risk(score)

    message = get_alert_message(level)

    print(f"\nRisk Score: {score}")
    print(f"Risk Level: {level}")
    print(f"Action: {message}")
    print("\n" + "=" * 50)
print("TEST 5: RISK ESCALATION")
print("=" * 50)


test_cases = [

    {
        "name": "Normal Risk",
        "spatial": 0.30,
        "temporal": 0.40,
        "vision": 0.20
    },

    {
        "name": "Two High Signals",
        "spatial": 0.65,
        "temporal": 0.70,
        "vision": 0.30
    },

    {
        "name": "Two Severe Signals",
        "spatial": 0.85,
        "temporal": 0.90,
        "vision": 0.40
    },

    {
        "name": "Single Severe Signal",
        "spatial": 0.85,
        "temporal": 0.30,
        "vision": 0.20
    }

]


for case in test_cases:

    base_risk, _ = fuse_risks(
        spatial_risk=case["spatial"],
        temporal_risk=case["temporal"],
        vision_risk=case["vision"]
    )

    base_level = classify_risk(base_risk)

    final_level, reason = apply_escalation(
        base_level,
        spatial_risk=case["spatial"],
        temporal_risk=case["temporal"],
        vision_risk=case["vision"]
    )

    print(f"\nCase: {case['name']}")
    print(f"Base Risk: {base_risk:.3f}")
    print(f"Base Level: {base_level}")
    print(f"Final Level: {final_level}")

    if reason:
        print(f"Reason: {reason}")