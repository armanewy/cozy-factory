extends Node

@export var content_dir := "res://content"

var grid: Node2D = null
var io_bus: Node = null
var game_state: Node = null
var _grid_cols: int = 12
var _grid_rows: int = 8
var _current_level: Node = null
var _vp_connected: bool = false

func _ready() -> void:
	# Load first level by default
	_ensure_level_loaded("res://content/levels/level_001.json")

func _ensure_level_loaded(level_path: String) -> void:
	var level_scene: PackedScene = load("res://scenes/Level.tscn")
	var level: Node = level_scene.instantiate()
	# Avoid adding children while the parent is still initializing
	get_parent().add_child.call_deferred(level)
	await get_tree().process_frame
	_current_level = level
	grid = level.get_node("Grid")
	io_bus = level.get_node("IoBus")
	game_state = level.get_node("GameState")
	if grid.has_method("set_io_bus"):
		grid.call("set_io_bus", io_bus)
	# Fit grid to current viewport
	if grid.has_method("fit_to_viewport_size"):
		var vp_size: Vector2 = get_viewport().get_visible_rect().size
		grid.call("fit_to_viewport_size", vp_size)
	# Connect viewport resize once
	if not _vp_connected:
		_vp_connected = true
		get_viewport().size_changed.connect(func():
			if grid and grid.has_method("fit_to_viewport_size"):
				var sz: Vector2 = get_viewport().get_visible_rect().size
				grid.call("fit_to_viewport_size", sz)
		)
	# Inform UI of the new GameState
	var ui := get_parent().get_node_or_null("UI")
	if ui and ui.has_method("set_game_state"):
		ui.call("set_game_state", game_state)
	_load_level(level_path)

func load_level_path(level_path: String) -> void:
	var sc := get_parent().get_node("SimClock")
	if sc:
		sc.call("pause_clock")
	if _current_level:
		_current_level.queue_free()
		await get_tree().process_frame
	_ensure_level_loaded(level_path)
	if sc:
		sc.call("start")

func reload_current() -> void:
	# For now, reload level_001; can track path later
	load_level_path("res://content/levels/level_001.json")

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
	if data.has("constraints"):
		game_state.set_constraints(data.get("constraints"))
	# Spawn a minimal starter chain in the center
	_spawn_demo_chain()

func _spawn_demo_chain() -> void:
	var cx: int = _grid_cols >> 1
	var cy: int = _grid_rows >> 1
	var c: Vector2i = Vector2i(cx - 3, cy)
	_place_machine("res://scripts/machines/mill.gd", c)
	_place_conveyor(c + Vector2i(1,0), Vector2i(1,0))
	_place_machine("res://scripts/machines/mixer.gd", c + Vector2i(2,0))
	_place_conveyor(c + Vector2i(3,0), Vector2i(1,0))
	_place_machine("res://scripts/machines/oven.gd", c + Vector2i(4,0))
	_place_conveyor(c + Vector2i(5,0), Vector2i(1,0))
	_place_machine("res://scripts/machines/seller.gd", c + Vector2i(6,0))

func _place_machine(script_path: String, cell: Vector2i) -> void:
	var s: Node2D = load("res://scenes/Building.tscn").instantiate() as Node2D
	s.set_script(load(script_path))
	add_child(s)
	s.position = grid.to_world(cell)
	if s.has_method("set_refs"):
		s.call("set_refs", grid, io_bus)
	# Provide id early so base can pick texture even if derived doesn't call base
	if script_path.find("mill") != -1:
		s.set("id", "mill")
	elif script_path.find("mixer") != -1:
		s.set("id", "mixer")
	elif script_path.find("oven") != -1:
		s.set("id", "oven")
	elif script_path.find("seller") != -1:
		s.set("id", "seller")
	if s.has_method("_add_sprite"):
		s.call("_add_sprite")
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
