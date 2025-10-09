extends Node

# Per-tile items: Dictionary(String(cell_key) => Array[Dictionary])
var tiles: Dictionary = {}
var grid_cols: int = 12
var grid_rows: int = 8

func configure(cols: int, rows: int) -> void:
    grid_cols = cols
    grid_rows = rows

func _key(cell: Vector2i) -> String:
    return str(cell.x, ",", cell.y)

func in_bounds(cell: Vector2i) -> bool:
    return cell.x >= 0 and cell.y >= 0 and cell.x < grid_cols and cell.y < grid_rows

func get_items(cell: Vector2i) -> Array:
    var arr: Array = tiles.get(_key(cell), [])
    return arr

func set_items(cell: Vector2i, arr: Array) -> void:
    tiles[_key(cell)] = arr

func push_output(cell: Vector2i, kind: String, n:int=1, dir:Vector2i=Vector2i.ZERO) -> void:
    # Spawn items on a cell with zero progress towards dir
    if not in_bounds(cell):
        return
    var a: Array = get_items(cell)
    for i in range(n):
        a.append({"kind": kind, "dir": dir, "progress": 0.0})
    set_items(cell, a)

func take_inputs(cell: Vector2i, required: Array[String]) -> bool:
    var a: Array = get_items(cell)
    var consumed: Array[int] = []
    for r in required:
        var idx := -1
        for i in range(a.size()):
            if i in consumed: continue
            var it: Dictionary = a[i]
            if typeof(it) == TYPE_DICTIONARY and it.get("kind") == r and float(it.get("progress",0)) <= 0.0:
                idx = i; break
        if idx == -1:
            return false
        consumed.append(idx)
    # Remove in reverse order
    consumed.sort() ; consumed.reverse()
    for idx in consumed:
        a.remove_at(idx)
    set_items(cell, a)
    return true

func advance_items(cell: Vector2i, dir: Vector2i, dt_ms: int, tiles_per_sec: float) -> void:
    if not in_bounds(cell):
        return
    var a: Array = get_items(cell)
    var speed: float = max(tiles_per_sec, 0.001)
    var delta_tiles: float = float(dt_ms) / 1000.0 * speed
    var remain: Array = []
    for it in a:
        var d: Dictionary = it
        d["dir"] = dir
        d["progress"] = float(d.get("progress", 0.0)) + delta_tiles
        if d["progress"] >= 1.0:
            var next: Vector2i = cell + dir
            if in_bounds(next):
                var arrive: Array = get_items(next)
                d["progress"] = 0.0
                arrive.append(d)
                set_items(next, arrive)
            else:
                # Off-grid => drop (treated as sold if a sink handles it)
                pass
        else:
            remain.append(d)
    set_items(cell, remain)
