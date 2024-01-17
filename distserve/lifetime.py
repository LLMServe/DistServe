from enum import Enum
import dataclasses
import time
import json
from typing import List

"""
This file (lifetime.py) contains utilities for lifetime monitoring of a request
Used for micro-benchmark in the paper "disaggregate"

A request's lifetime looks like this:

Issued
|
| Context Queuing
|
Context Begin
|
| Context-ing
|
Context End
|
| Waiting in the bridge queue
|
Migration Begin
|
| Migrating
|
Migration End
|
| Decoding Queuing
|
Decoding Begin
|
| Decoding
|
Decoding End
"""

class LifetimeEventType(Enum):
    """
    The type of an event in a request's lifetime
    """
    Issued = "issued"
    ContextBegin = "context_begin"
    ContextEnd = "context_end"
    MigrationBegin = "migration_begin"
    MigrationEnd = "migration_end"
    DecodingBegin = "decoding_begin"
    DecodingEnd = "decoding_end"

    def __str__(self) -> str:
        return self.value

class LifetimeEvent(json.JSONEncoder):
    """
    An event in a request's lifetime
    Contains a timestamp and a type
    """
    
    def __init__(self, event_type: LifetimeEventType, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        self.event_type = event_type
        self.timestamp = timestamp
        
def json_encode_lifetime_events(events: List[LifetimeEvent]) -> List[dict]:
    return [
        {
            "timestamp": event.timestamp,
            "event_type": str(event.event_type)
        } 
        for event in events
    ]

def json_decode_lifetime_events(json_list: List[dict]) -> List[LifetimeEvent]:
    return [
        LifetimeEvent(LifetimeEventType(event["event_type"]), event["timestamp"])
        for event in json_list
    ]