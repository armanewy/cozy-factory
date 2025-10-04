ids = ['wheat_field_001','mill_001','bakery_001','market_001','lamp_post_001','signboard_001','water_pump_001','farm_001','cow_001','dairy_001','sugarcane_001','sugar_mill_001','pastry_001']
for cid in ids:
    val = int.from_bytes(cid.encode('utf-8'),'big') & 0xFFFFFFFF
    print(cid, val)
