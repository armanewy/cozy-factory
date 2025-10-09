extends "res://scripts/machine.gd"

@export var accepts: Array[String] = ["croissant"]

var game_state: Node = null

func set_game_state(gs: Node) -> void:
    game_state = gs

func tick(dt_ms: int) -> void:
    # Try to consume as many accepted items as available this tick (bounded)
    var cell := _cell()
    var consumed := 0
    for attempt in range(8):
        if io_bus and io_bus.take_inputs(cell, accepts):
            consumed += 1
        else:
            break
    if consumed > 0 and game_state and game_state.has_method("on_item_sold"):
        game_state.call("on_item_sold", accepts[0], consumed)

func _draw() -> void:
    var size := Vector2(42, 42)
    draw_rect(Rect2(-size*0.5, size), Color(1.0,0.95,0.85))
    draw_rect(Rect2(-size*0.5, size), Color(0.2,0.2,0.2), false, 2.0)
    draw_string(get_tree().root.get_theme_default_font(), Vector2(-10,5), "$", HAlign.LEFT, -1, 12)

