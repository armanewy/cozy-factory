extends "res://scripts/machine.gd"

func _ready() -> void:
    id = "oven"
    cycle_ms = 1500
    power_watts = 200
    inputs = ["dough"]
    outputs = ["croissant"]
    footprint = Vector2i(2,1)
    _add_sprite()
