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
    _accum += dt_ms
    if _accum >= cycle_ms:
        if _process_cycle():
            _accum = 0

func _cell() -> Vector2i:
    if grid and "to_cell" in grid:
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

