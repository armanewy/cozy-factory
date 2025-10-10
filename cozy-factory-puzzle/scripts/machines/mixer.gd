extends "res://scripts/machine.gd"

func _ready() -> void:
    id = "mixer"
    cycle_ms = 1200
    power_watts = 80
    inputs = ["flour"]
    outputs = ["dough"]
    footprint = Vector2i(1,1)
    _add_sprite()
