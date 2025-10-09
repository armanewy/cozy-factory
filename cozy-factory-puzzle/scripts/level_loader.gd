extends Node

@export var content_dir := "res://content"

var grid: Node2D = null
var io_bus: Node = null
var game_state: Node = null
var _grid_cols: int = 12
var _grid_rows: int = 8

func _ready() -> void:
	# Load first level by default
	_ensure_level_loaded("res://content/levels/level_001.json")

func _ensure_level_loaded(level_path: String) -> void:
	var level_scene: PackedScene = load("res://scenes/Level.tscn")
	var level: Node = level_scene.instantiate()
	# Avoid adding children while the parent is still initializing
	get_parent().add_child.call_deferred(level)
	await get_tree().process_frame
	grid = level.get_node("Grid")
	io_bus = level.get_node("IoBus")
	game_state = level.get_node("GameState")
	_load_level(level_path)

func _load_level(level_path: String) -> void:
	var f := FileAccess.open(level_path, FileAccess.READ)
	if f == null:
		push_error("Cannot open level: %s" % level_path)
		return
	var data = JSON.parse_string(f.get_as_text())
	if typeof(data) != TYPE_DICTIONARY:
		push_error("Invalid level JSON")
		return
	var grid_conf = data.get("grid", {"cols":12, "rows":8})
	_grid_cols = int(grid_conf.get("cols",12))
	_grid_rows = int(grid_conf.get("rows",8))
	grid.configure(_grid_cols, _grid_rows)
	io_bus.configure(_grid_cols, _grid_rows)
	game_state.target = data.get("target", {})
	# Spawn a minimal starter chain in the center
	_spawn_demo_chain()

func _spawn_demo_chain() -> void:
	var cx: int = _grid_cols >> 1
	var cy: int = _grid_rows >> 1
	var c: Vector2i = Vector2i(cx - 2, cy)
	_place_machine("res://scripts/machines/mill.gd", c)
	_place_conveyor(c + Vector2i(1,0), Vector2i(1,0))
	_place_machine("res://scripts/machines/mixer.gd", c + Vector2i(2,0))
	_place_conveyor(c + Vector2i(3,0), Vector2i(1,0))
	_place_machine("res://scripts/machines/oven.gd", c + Vector2i(4,0))

func _place_machine(script_path: String, cell: Vector2i) -> void:
	var s: Node2D = load("res://scenes/Building.tscn").instantiate() as Node2D
	s.set_script(load(script_path))
	add_child(s)
	s.position = grid.to_world(cell)
	if s.has_method("set_refs"):
		s.call("set_refs", grid, io_bus)
	var fp: Vector2i = s.get("footprint")
	grid.place(s, cell, fp.x, fp.y)

func _place_conveyor(cell: Vector2i, dir: Vector2i) -> void:
	var c: Node2D = load("res://scenes/Conveyor.tscn").instantiate() as Node2D
	add_child(c)
	c.position = grid.to_world(cell)
	c.set("direction", dir)
	if c.has_method("set_refs"):
		c.call("set_refs", grid, io_bus)
	grid.place(c, cell, 1, 1)
