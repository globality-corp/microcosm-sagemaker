from dataclasses import dataclass


@dataclass
class SimplePrediction:
    uri: str
    score: float
