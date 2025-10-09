extends Node2D

@export var cell_size := 64
var cols := 12
var rows := 8
var occupied := {} # key -> Node2D
var io_bus: Node = null

func configure(cols_in: int, rows_in: int) -> void:
    cols = cols_in
    rows = rows_in

func get_size() -> Vector2i:
    return Vector2i(cols, rows)

func set_io_bus(io: Node) -> void:
    io_bus = io
    set_process(true)
    queue_redraw()

func _process(_delta: float) -> void:
    queue_redraw()

func _draw() -> void:
    # Grid lines
    var cs := float(cell_size)
    var col_grid := Color(0.8,0.8,0.85,1.0)
    for x in range(cols+1):
        draw_line(Vector2(x*cs, 0), Vector2(x*cs, rows*cs), col_grid, 1.0)
    for y in range(rows+1):
        draw_line(Vector2(0, y*cs), Vector2(cols*cs, y*cs), col_grid, 1.0)
    # Items debug: small circles moving along direction
    if io_bus and io_bus.has_method("get_items") and io_bus.has_method("in_bounds"):
        var tiles: Dictionary = io_bus.tiles if "tiles" in io_bus else {}
        for k in tiles.keys():
            var parts := str(k).split(",")
            if parts.size() != 2:
                continue
            var cx := int(parts[0])
            var cy := int(parts[1])
            var cell_origin := Vector2(cx*cs, cy*cs)
            var arr: Array = tiles[k]
            for it in arr:
                if typeof(it) != TYPE_DICTIONARY:
                    continue
                var prog := float(it.get("progress", 0.0))
                var dirv: Vector2i = it.get("dir", Vector2i.ZERO)
                var dirf := Vector2(dirv.x, dirv.y)
                var pos := cell_origin + Vector2(cs*0.5, cs*0.5) + dirf * (prog-0.5) * cs * 0.8
                draw_circle(pos, 6.0, Color(0.2,0.2,0.2,1.0))

func to_cell(world_pos: Vector2) -> Vector2i:
    var local := to_local(world_pos)
    return Vector2i(floor(local.x / cell_size), floor(local.y / cell_size))

func to_world(cell: Vector2i) -> Vector2:
    return to_global(Vector2((cell.x + 0.5) * cell_size, (cell.y + 0.5) * cell_size))

func rect_footprint(cell: Vector2i, w: int, h: int) -> Array[Vector2i]:
    var arr: Array[Vector2i] = []
    for y in h:
        for x in w:
            arr.append(Vector2i(cell.x + x, cell.y + y))
    return arr

func free_for(cell: Vector2i, w: int, h: int) -> bool:
    for c in rect_footprint(cell, w, h):
        if c.x < 0 or c.y < 0 or c.x >= cols or c.y >= rows:
            return false
        if occupied.has(_k(c)):
            return false
    return true

func place(node: Node2D, cell: Vector2i, w: int, h: int) -> bool:
    if not free_for(cell, w, h):
        return false
    for c in rect_footprint(cell, w, h):
        occupied[_k(c)] = node
    node.position = Vector2((cell.x + 0.5) * cell_size, (cell.y + 0.5) * cell_size)
    node.add_to_group("tickables")
    return true

func remove(node: Node2D) -> void:
    for k in occupied.keys():
        if occupied[k] == node:
            occupied.erase(k)

func get_at(cell: Vector2i) -> Node2D:
    return occupied.get(_k(cell), null)

func _k(c: Vector2i) -> String:
    return str(c.x, ",", c.y)
