Cozy Factory Puzzle (Phase 2)

This Godot project provides a deterministic, fixed‑tick factory sandbox used to validate and play with Cozy Factory content.

Run
- Open `project.godot` in Godot 4.3+ and press Play.
- The grid auto‑fits the window and recenters when you resize.

Controls
- Build palette: `1` Mill, `2` Mixer, `3` Oven, `4` Belt, `5` Seller, `0` Erase
- Place/Remove: Left/Right click
- Rotate: `R` rotates a belt under the cursor; otherwise rotates the placement direction
- Sim: Space toggles Play/Pause; Reset reloads; Level 1/2/3 buttons switch levels

Architecture
- Scenes
  - `Main.tscn`: root scene with `SimClock`, `LevelLoader`, and `UI`
  - `Level.tscn`: holds `Grid`, `IoBus`, `GameState`, and `Placement`
- Core scripts
  - `scripts/sim_clock.gd`: fixed‐tick clock (100 ms) that calls `tick(dt_ms)` on nodes in group `tickables`
  - `scripts/io_bus.gd`: per‑cell arrays of items `{ kind, dir, progress }`; belts advance deterministically; items move when `progress >= 1`
  - `scripts/grid.gd`: grid occupancy, placement, coordinate transforms; emits `cell_size_changed` and refits to the viewport; all placed nodes are children of `Grid` and store `grid_cell` + `grid_size` so they remain locked on resize
  - `scripts/machine.gd`: base for machines (`cycle_ms`, `inputs`, `outputs`, `footprint`, `facing`); handles sprite creation/scaling and tick cycle
  - `scripts/machines/*.gd`: `mill`, `mixer`, `oven`, `seller` (consumes outputs and reports to `GameState`)
  - `scripts/conveyor.gd`: belt with 4‑way rotation; on tick advances items; sprite rotates with direction and scales to cell size
  - `scripts/game_state.gd`: constraints (budget/power), produced counts, `stats_changed` and `level_passed` signals
  - `scripts/placement.gd`: input → placement with ghost preview (green/red validity), rotation, erase
  - `scripts/level_loader.gd`: loads level JSON, configures Grid/IoBus/GameState, spawns a starter chain, wires UI
  - `scripts/ui.gd` + `scenes/UI.tscn`: play/pause/reset, level select, budget/power bars, win banner

Data & content
- Levels: `content/levels/level_001.json` etc. Schema:
  - `grid: { cols, rows }`
  - `constraints: { budget, power_watts, max_tiles }`
  - `target: { croissant: 12 }`
- Building stats: `content/buildings.yml` documents footprints and cycle_ms; machines currently read stats from their scripts for simplicity but keep values aligned with the YAML.

Visuals
- Sprites live in `assets/cards/` (copied from the main repo’s art). Machines and belts scale to the current cell size with linear filtering for clean visuals. The ghost preview uses the same textures and scales per cell.

Determinism
- All simulation logic runs on fixed ticks and reads `dt_ms` from `SimClock`. Do not use frame `delta` in sim code. Keep randomness out (or seed a private RNG from level `seed` and never touch globals).

Code conventions
- Tabs for indentation (no mixed spaces), strict types where inference would produce `Variant`. Keep nodes that should follow the grid under the `Grid` node.

Extending
- To add a new machine: create `scripts/machines/new_machine.gd` extending `machine.gd`, set `id`, `cycle_ms`, `inputs`, `outputs`, `footprint`, call `_add_sprite()` in `_ready()`. Add a build entry in `scripts/build_catalog.gd` and a key binding in `placement.gd` (or surface it via UI).
- To add a new level: drop a JSON in `content/levels/` and add a button in `UI.tscn` or call `LevelLoader.load_level_path(path)` from code.
