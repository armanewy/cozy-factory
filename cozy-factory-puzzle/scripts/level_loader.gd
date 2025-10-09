extends Node

@export var content_dir := "res://content"

var grid: Node2D = null
var io_bus: Node = null
var game_state: Node = null

func _ready() -> void:
    # Load first level by default
    _ensure_level_loaded("res://content/levels/level_001.json")

func _ensure_level_loaded(level_path: String) -> void:
    var level_scene: PackedScene = load("res://scenes/Level.tscn")
    var level: Node = level_scene.instantiate()
    get_tree().get_root().get_node("/root/Main").add_child(level)
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
    grid.configure(int(grid_conf.get("cols",12)), int(grid_conf.get("rows",8)))
    io_bus.configure(int(grid_conf.get("cols",12)), int(grid_conf.get("rows",8)))
    game_state.target = data.get("target", {})
    # Spawn a minimal starter chain in the center
    _spawn_demo_chain()

func _spawn_demo_chain() -> void:
    var cols := grid.cols
    var rows := grid.rows
    var c := Vector2i(cols/2-2, rows/2)
    _place_machine("res://scripts/machines/mill.gd", c)
    _place_conveyor(c + Vector2i(1,0), Vector2i(1,0))
    _place_machine("res://scripts/machines/mixer.gd", c + Vector2i(2,0))
    _place_conveyor(c + Vector2i(3,0), Vector2i(1,0))
    _place_machine("res://scripts/machines/oven.gd", c + Vector2i(4,0))

func _place_machine(script_path: String, cell: Vector2i) -> void:
    var s := load("res://scenes/Building.tscn").instantiate()
    s.set_script(load(script_path))
    add_child(s)
    if s is Node2D:
        s.position = grid.to_world(cell)
    if "set_refs" in s:
        s.set_refs(grid, io_bus)
    grid.place(s, cell, s.footprint.x, s.footprint.y)

func _place_conveyor(cell: Vector2i, dir: Vector2i) -> void:
    var c := load("res://scenes/Conveyor.tscn").instantiate()
    add_child(c)
    c.position = grid.to_world(cell)
    c.direction = dir
    c.set_refs(grid, io_bus)
    grid.place(c, cell, 1, 1)

