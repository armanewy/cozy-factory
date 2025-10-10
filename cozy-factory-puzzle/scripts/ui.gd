extends CanvasLayer

var main: Node = null
var level_loader: Node = null
var sim_clock: Node = null
var game_state: Node = null

func _ready() -> void:
    main = get_parent()
    level_loader = main.get_node("LevelLoader")
    sim_clock = main.get_node("SimClock")
    _wire_buttons()
    _refresh_info()

func _wire_buttons() -> void:
    var btn_pp: Button = $HUD/TopBar/BtnPlayPause
    var btn_reset: Button = $HUD/TopBar/BtnReset
    btn_pp.pressed.connect(_on_play_pause)
    btn_reset.pressed.connect(_on_reset)
    $HUD/TopBar/Levels/L1.pressed.connect(func(): _load_level("res://content/levels/level_001.json"))
    $HUD/TopBar/Levels/L2.pressed.connect(func(): _load_level("res://content/levels/level_002.json"))
    $HUD/TopBar/Levels/L3.pressed.connect(func(): _load_level("res://content/levels/level_003.json"))
    set_process_unhandled_input(true)

func _unhandled_input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_SPACE:
        _on_play_pause()

func _on_play_pause() -> void:
    if sim_clock == null: return
    var running: bool = sim_clock.get("_running")
    if running:
        sim_clock.call("pause_clock")
    else:
        sim_clock.call("start")

func _on_reset() -> void:
    # Reload current level path tracked by loader
    if level_loader and level_loader.has_method("reload_current"):
        level_loader.call("reload_current")

func _load_level(path: String) -> void:
    if level_loader and level_loader.has_method("load_level_path"):
        level_loader.call("load_level_path", path)
    _refresh_info()

func set_game_state(gs: Node) -> void:
    if gs == null: return
    game_state = gs
    if gs.has_signal("level_passed"):
        gs.level_passed.connect(_on_level_passed)
    if gs.has_signal("stats_changed"):
        gs.stats_changed.connect(_refresh_info)
    _refresh_info()

func _on_level_passed() -> void:
    $HUD/Win.text = "Level Passed!"
    if sim_clock:
        sim_clock.call("pause_clock")

func _refresh_info() -> void:
    if game_state == null:
        var level: Node = _find_level()
        if level:
            set_game_state(level.get_node("GameState"))
    if game_state == null: return
    var info: String = "Budget: %d/%d  Power: %d/%d  Target: %s  Produced: %s" % [game_state.spent, game_state.budget, game_state.power_used, game_state.power_limit, str(game_state.target), str(game_state.produced)]
    $HUD/TopBar/Info.text = info

func _find_level() -> Node:
    for n in get_tree().get_nodes_in_group("root"):
        pass
    # fallback: look under Main
    return get_parent().get_node_or_null("Level")
