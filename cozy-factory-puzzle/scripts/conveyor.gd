extends Node2D

@export var direction: Vector2i = Vector2i(1, 0)
@export var belt_speed: float = 1.0

var grid: Node = null
var io_bus: Node = null

func set_refs(grid_in: Node, io_in: Node) -> void:
    grid = grid_in
    io_bus = io_in
    add_to_group("tickables")
    queue_redraw()

func tick(dt_ms: int) -> void:
    if io_bus == null or grid == null:
        return
    var cell: Vector2i = grid.to_cell(global_position)
    io_bus.advance_items(cell, direction, dt_ms, belt_speed)

func rotate_dir() -> void:
    # Right -> Down -> Left -> Up
    direction = Vector2i(-direction.y, direction.x)
    queue_redraw()

func _draw() -> void:
    var size := Vector2(48, 32)
    draw_rect(Rect2(-size*0.5, size), Color(0.9,0.9,0.9))
    draw_rect(Rect2(-size*0.5, size), Color(0.3,0.3,0.3), false, 2.0)
    var dirv := Vector2(direction.x, direction.y).normalized()
    var tip := dirv * 18.0
    draw_line(Vector2.ZERO - tip*0.6, tip, Color(0.2,0.2,0.2), 3.0)
    # Arrow head
    var left := tip.rotated(-0.6) * 0.4
    var right := tip.rotated(0.6) * 0.4
    draw_line(tip, tip - left, Color(0.2,0.2,0.2), 3.0)
    draw_line(tip, tip - right, Color(0.2,0.2,0.2), 3.0)
