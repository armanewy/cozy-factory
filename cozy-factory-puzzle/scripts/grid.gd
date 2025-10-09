extends Node2D

@export var cell_size := 64
var cols := 12
var rows := 8
var occupied := {} # key -> Node2D

func configure(cols_in: int, rows_in: int) -> void:
    cols = cols_in
    rows = rows_in

func get_size() -> Vector2i:
    return Vector2i(cols, rows)

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

func _k(c: Vector2i) -> String:
    return str(c.x, ",", c.y)
