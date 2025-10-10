extends "res://scripts/machine.gd"

func _ready() -> void:
    id = "mill"
    # Defaults; LevelLoader may override from content data
    cycle_ms = 800
    power_watts = 50
    inputs = []
    outputs = ["flour"]
    footprint = Vector2i(1,1)
    _add_sprite()
