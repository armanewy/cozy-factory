extends Node2D

class_name Machine

@export var id: String = ""
@export var cycle_ms: int = 1000
@export var power_watts: int = 50
@export var inputs: Array[String] = []
@export var outputs: Array[String] = []
@export var footprint: Vector2i = Vector2i(1,1)
@export var facing: Vector2i = Vector2i(1,0)

var _accum := 0
var grid: Node = null
var io_bus: Node = null

func set_refs(grid_in: Node, io_in: Node) -> void:
	grid = grid_in
	io_bus = io_in

func tick(dt_ms: int) -> void:
    if grid == null or io_bus == null:
        return
    if not is_instance_valid(grid) or not is_inside_tree():
        return
    _accum += dt_ms
	if _accum >= cycle_ms:
		if _process_cycle():
			_accum = 0
	queue_redraw()

func _cell() -> Vector2i:
    if grid != null and is_instance_valid(grid) and "to_cell" in grid:
        return grid.to_cell(global_position)
    return Vector2i.ZERO

func _process_cycle() -> bool:
	# Default cycle: consume inputs -> produce one of each output
	if io_bus and inputs.size() == 0 and outputs.size() > 0:
		for k in outputs:
			io_bus.push_output(_cell(), k, 1, facing)
		return true
	if io_bus and inputs.size() > 0:
		if io_bus.take_inputs(_cell(), inputs):
			for k in outputs:
				io_bus.push_output(_cell(), k, 1, facing)
			return true
	return false

func _draw() -> void:
	var col := Color(0.85, 0.9, 1.0)
	if id == "mill":
		col = Color(0.80, 0.93, 0.82)
	elif id == "mixer":
		col = Color(0.85, 0.85, 0.98)
	elif id == "oven":
		col = Color(0.98, 0.88, 0.78)
	var size := Vector2(48, 48)
	draw_rect(Rect2(-size*0.5, size), col)
	draw_rect(Rect2(-size*0.5, size), Color(0.2,0.2,0.2), false, 2.0)
