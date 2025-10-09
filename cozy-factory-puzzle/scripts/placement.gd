extends Node

@export var default_build: String = "belt"

var level: Node = null
var grid: Node = null
var io_bus: Node = null
var game_state: Node = null
var current: String = "belt"
var dir: Vector2i = Vector2i(1,0)

func _ready() -> void:
    current = default_build
    level = get_parent()
    grid = level.get_node("Grid")
    io_bus = level.get_node("IoBus")
    game_state = level.get_node("GameState")
    set_process_unhandled_input(true)

func _unhandled_input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed and not event.echo:
        match event.keycode:
            KEY_1: current = "mill"
            KEY_2: current = "mixer"
            KEY_3: current = "oven"
            KEY_4: current = "belt"
            KEY_5: current = "seller"
            KEY_0: current = "erase"
            KEY_R:
                # Try to rotate the conveyor under the cursor; otherwise rotate placement dir
                var mp: Vector2 = get_viewport().get_mouse_position()
                var cell: Vector2i = grid.to_cell(mp)
                var node: Node2D = grid.get_at(cell)
                if node and node.has_method("rotate_dir"):
                    node.call("rotate_dir")
                else:
                    dir = Vector2i(-dir.y, dir.x)
                get_viewport().set_input_as_handled()
                return
    if event is InputEventMouseButton and event.pressed and event.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT]:
        var cell: Vector2i = grid.to_cell(event.position)
        if event.button_index == MOUSE_BUTTON_RIGHT or current == "erase":
            var n: Node2D = grid.get_at(cell)
            if n:
                _refund(n)
                grid.remove(n)
                n.queue_free()
            return
        _try_place(cell)

func _refund(n: Node2D) -> void:
    var id_val: String = str(n.get("build_id"))
    var def: Dictionary = BuildCatalog.get_def(id_val)
    if def.size() > 0 and game_state:
        game_state.on_removed(def)

func _try_place(cell: Vector2i) -> void:
    var def: Dictionary = BuildCatalog.get_def(current)
    if def.size() == 0:
        return
    if not game_state.can_place(def):
        return
    var w: int = int(def.get("footprint", Vector2i.ONE).x)
    var h: int = int(def.get("footprint", Vector2i.ONE).y)
    var place_dir: Vector2i = dir
    # rotate non-square machines when facing vertical
    if def.get("type") == "machine" and w != h and (dir.y != 0):
        var tmp: int = w; w = h; h = tmp
    if not grid.free_for(cell, w, h):
        return
    var node: Node2D
    if def.get("type") == "belt":
        node = load("res://scenes/Conveyor.tscn").instantiate() as Node2D
        node.set("direction", place_dir)
        node.call("set_refs", grid, io_bus)
    else:
        node = load("res://scenes/Building.tscn").instantiate() as Node2D
        node.set_script(load(def.get("script")))
        node.set("facing", place_dir)
        node.call("set_refs", grid, io_bus)
        if node.has_method("set_game_state"):
            node.call("set_game_state", game_state)
    node.set("build_id", current)
    if grid.place(node, cell, w, h):
        get_parent().add_child(node)
        game_state.on_placed(def)
