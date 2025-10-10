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
var _sprite: Sprite2D = null

func set_refs(grid_in: Node, io_in: Node) -> void:
	grid = grid_in
	io_bus = io_in
	if grid and grid.has_signal("cell_size_changed"):
		grid.cell_size_changed.connect(_on_grid_cell_size_changed)


func tick(dt_ms: int) -> void:
	# Guard against level reloads/freeing
	if grid == null or io_bus == null:
		return
	if not is_instance_valid(grid) or not is_instance_valid(io_bus) or not is_inside_tree():
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
	# Fallback (no texture)
	if _sprite != null: return
	var size := Vector2(48, 48)
	draw_rect(Rect2(-size*0.5, size), Color(0.85,0.9,1.0))
	draw_rect(Rect2(-size*0.5, size), Color(0.2,0.2,0.2), false, 2.0)

func _on_grid_cell_size_changed(_cs: int) -> void:
	_fit_sprite()
func _ready() -> void:
	_add_sprite()

func _add_sprite() -> void:
	if _sprite != null: return
	var tex := _texture_for_id()
	if tex == null: return
	_sprite = Sprite2D.new()
	_sprite.name = "Sprite"
	_sprite.texture = tex
	_sprite.centered = true
	_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	add_child(_sprite)
	_fit_sprite()

func _texture_for_id() -> Texture2D:
	var path := ""
	match id:
		"mill": path = "res://assets/cards/mill.png"
		"mixer": path = "res://assets/cards/mixer.png"
		"oven": path = "res://assets/cards/oven.png"
		"seller": path = "res://assets/cards/market.png"
		_: path = ""
    if path != "" and ResourceLoader.exists(path):
        return load(path)
    return null

func _fit_sprite() -> void:
    if _sprite == null or _sprite.texture == null or grid == null: return
    var sz: Vector2i = footprint
    var target_w: float = float(grid.cell_size) * float(sz.x)
    var target_h: float = float(grid.cell_size) * float(sz.y)
    var tw: float = float(_sprite.texture.get_width())
    var th: float = float(_sprite.texture.get_height())
    var s: float = min(target_w / tw, target_h / th) * 0.9
    _sprite.scale = Vector2(s, s)
