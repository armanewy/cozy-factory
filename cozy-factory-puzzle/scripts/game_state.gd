extends Node

signal level_passed
signal stats_changed

var budget := 999999
var power_limit := 999999
var spent := 0
var power_used := 0
var produced := {}
var target := {}

func reset() -> void:
    produced.clear()
    spent = 0
    power_used = 0
    stats_changed.emit()

func on_item_sold(kind: String, n:=1) -> void:
    produced[kind] = int(produced.get(kind, 0)) + n
    stats_changed.emit()
    _check_win()

func _check_win() -> void:
    if target.size() == 0:
        return
    for k in target.keys():
        if int(produced.get(k, 0)) < int(target[k]):
            return
    level_passed.emit()

func set_constraints(c: Dictionary) -> void:
    budget = int(c.get("budget", budget))
    power_limit = int(c.get("power_watts", power_limit))
    stats_changed.emit()

func can_place(def: Dictionary) -> bool:
    var cost := int(def.get("cost", 0))
    var power := int(def.get("power", 0))
    return (spent + cost) <= budget and (power_used + power) <= power_limit

func on_placed(def: Dictionary) -> void:
    spent += int(def.get("cost", 0))
    power_used += int(def.get("power", 0))
    stats_changed.emit()

func on_removed(def: Dictionary) -> void:
    spent -= int(def.get("cost", 0))
    power_used -= int(def.get("power", 0))
    stats_changed.emit()
