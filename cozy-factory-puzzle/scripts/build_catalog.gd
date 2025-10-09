extends Node

class_name BuildCatalog

static var defs := {
    "mill":   {"type":"machine", "script":"res://scripts/machines/mill.gd",   "footprint": Vector2i(1,1), "power":50,  "cost":100},
    "mixer":  {"type":"machine", "script":"res://scripts/machines/mixer.gd",  "footprint": Vector2i(1,1), "power":80,  "cost":150},
    "oven":   {"type":"machine", "script":"res://scripts/machines/oven.gd",   "footprint": Vector2i(2,1), "power":200, "cost":250},
    "belt":   {"type":"belt",    "scene":"res://scenes/Conveyor.tscn",         "footprint": Vector2i(1,1), "power":0,   "cost":50,  "belt_speed": 1.0},
    "seller": {"type":"machine", "script":"res://scripts/machines/seller.gd", "footprint": Vector2i(1,1), "power":0,   "cost":0}
}

static func get_def(id: String) -> Dictionary:
    return defs.get(id, {})

