Cozy Factory Puzzle (Phase 2 scaffold)

Open `cozy-factory-puzzle/project.godot` in Godot 4.x and press Play.

What’s included
- Fixed-tick `SimClock` (100 ms) that drives all `tickables`.
- Minimal `Level.tscn` with `Grid`, `IoBus`, and `GameState`.
- Machines: `mill`, `mixer`, `oven` (scripts in `scripts/machines/`).
- Conveyor with direction and speed; items advance deterministically.
- Level loader that loads `content/levels/level_001.json` and spawns a demo chain.

Next steps you can add quickly
- Placement and rotation UI (map `R` to `Conveyor.rotate_dir()`).
- Level select that swaps `Level.tscn` and calls the loader with a different JSON.
- Hook `GameState.on_item_sold()` wherever your sink/seller is implemented.

