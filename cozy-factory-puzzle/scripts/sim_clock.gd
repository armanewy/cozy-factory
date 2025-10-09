extends Node

@export var tick_ms: int = 100
var _accum := 0.0
var _running := false

signal tick(dt_ms: int)

func _ready() -> void:
    set_process(true)
    _running = true

func start() -> void:
    _running = true

func pause_clock() -> void:
    _running = false

func _process(delta: float) -> void:
    if not _running:
        return
    _accum += delta * 1000.0
    while _accum >= tick_ms:
        _accum -= tick_ms
        tick.emit(tick_ms)
        for n in get_tree().get_nodes_in_group("tickables"):
            if "tick" in n:
                n.tick(tick_ms)
