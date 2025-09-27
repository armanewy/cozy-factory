import argparse, json, collections

CARDS = "assets/meta/cards.json"

def load_cards():
    with open(CARDS, "r", encoding="utf-8") as f:
        return json.load(f)

def tick(cards, stocks):
    # Each card applies consumes then produces. No capacities yet; clamp at zero.
    # Returns note of potential bottlenecks (when consumes > available).
    notes=[]
    # consume
    for c in cards:
        for cc in c.get("consumes", []):
            res, rate = cc["resource"], cc["rate"]
            have = stocks.get(res, 0)
            if have < rate:
                notes.append(f"{c['id']}: lacking {res} ({have}/{rate})")
                rate = min(rate, have)
            stocks[res] = have - rate
    # produce
    for c in cards:
        for pp in c.get("produces", []):
            res, rate = pp["resource"], pp["rate"]
            stocks[res] = stocks.get(res, 0) + rate
    return notes

def main(ticks: int, verbose: bool):
    cards = load_cards()
    # Start with 0 of everything, except seed a little wheat so the chain moves
    stocks = collections.defaultdict(float, {"wheat": 4.0})
    bottlenecks=collections.Counter()
    for t in range(1, ticks+1):
        notes = tick(cards, stocks)
        for n in notes: bottlenecks[n.split(":")[0]] += 1
        if verbose or t in (1, ticks):
            print(f"[t={t}] stocks:", dict(sorted(stocks.items())))
            if notes: print("  bottlenecks:", notes)
    if bottlenecks:
        worst = bottlenecks.most_common(1)[0]
        print(f"\n🔎 Likely bottleneck: {worst[0]} ({worst[1]} ticks limited)")
    else:
        print("\n✅ No bottlenecks detected at this scale")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    main(args.ticks, args.verbose)
