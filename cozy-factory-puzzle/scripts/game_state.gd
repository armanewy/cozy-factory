extends Node

signal level_passed

var budget := 999999
var power_limit := 999999
var produced := {}
var target := {}

func reset() -> void:
    produced.clear()

func on_item_sold(kind: String, n:=1) -> void:
    produced[kind] = int(produced.get(kind, 0)) + n
    _check_win()

func _check_win() -> void:
    if target.size() == 0:
        return
    for k in target.keys():
        if int(produced.get(k, 0)) < int(target[k]):
            return
    level_passed.emit()
