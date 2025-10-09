extends Node2D

@export var direction: Vector2i = Vector2i(1, 0)
@export var belt_speed: float = 1.0

var grid: Node = null
var io_bus: Node = null

func set_refs(grid_in: Node, io_in: Node) -> void:
    grid = grid_in
    io_bus = io_in
    add_to_group("tickables")

func tick(dt_ms: int) -> void:
    if io_bus == null or grid == null:
        return
    var cell := grid.to_cell(global_position)
    io_bus.advance_items(cell, direction, dt_ms, belt_speed)

func rotate_dir() -> void:
    # Right -> Down -> Left -> Up
    direction = Vector2i(-direction.y, direction.x)

