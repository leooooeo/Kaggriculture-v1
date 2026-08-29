"""Canonical vocabularies, action tables and normalization constants for the
Kaggriculture imitation-learning pipeline.

Everything the feature encoder, action codec, model and runtime agent need to
agree on lives here so the training-time and inference-time representations can
never drift apart.
"""

# ---------------------------------------------------------------------------
# Board / game constants (defaults; overridden from obs/config when available)
# ---------------------------------------------------------------------------
BOARD_SIZE = 10
TURNS_PER_DAY = 24
SEASON_DAYS = 30
EPISODE_STEPS = 720
MAX_MARKET_ORDERS = 10
I0 = 10000  # market starting inventory per resource

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
# Products that have a market price / inventory (order matters, kept stable).
MARKET_ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER"]
# Everything that can sit in the shed / an inventory.
SHED_ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]
# Items a unit can PICKUP / PLACE (superset used for the arg head).
UNIT_ITEMS = SHED_ITEMS
SHOP_TYPES = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
              "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]
QUADRANTS = ["NW", "NE", "SW", "SE"]

# Per-resource base price (for price normalization).
BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

# Buyable-from-market products (BUY_PRODUCT) and buyable animals.
BUY_PRODUCTS = ["WHEAT", "FERTILIZER"]
BUY_ANIMALS = ANIMALS

# ---------------------------------------------------------------------------
# Unit (farmer / hand) action table
# ---------------------------------------------------------------------------
# index -> action verb.  Args (crop / item / count) are predicted by separate
# heads and only consumed for the verbs that need them.
UNIT_ACTIONS = [
    "PASS",              # 0
    "NORTH",             # 1
    "SOUTH",             # 2
    "EAST",              # 3
    "WEST",              # 4
    "PLANT",             # 5  -> crop head
    "WATER",             # 6
    "HARVEST",           # 7
    "FERTILIZE",         # 8
    "FEED",              # 9
    "CARE",              # 10
    "COLLECT_FERTILIZER",# 11
    "PICKUP",            # 12 -> item head + count head
    "DROP",              # 13
    "PLACE",             # 14 -> item head + count head
    "DIG",               # 15
    "BUILD_COOP",        # 16
    "BUILD_PASTURE",     # 17
]
UNIT_ACTION_IDX = {a: i for i, a in enumerate(UNIT_ACTIONS)}
N_UNIT_ACTIONS = len(UNIT_ACTIONS)

# Quantity buckets shared by count head and market quantity heads.
QTY_BUCKETS = [1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50]


def qty_to_bucket(n):
    """Nearest-bucket index for an integer quantity (>=1)."""
    n = max(1, int(n))
    best, bi = 10 ** 9, 0
    for i, b in enumerate(QTY_BUCKETS):
        d = abs(b - n)
        if d < best:
            best, bi = d, i
    return bi


def bucket_to_qty(i):
    i = max(0, min(len(QTY_BUCKETS) - 1, int(i)))
    return QTY_BUCKETS[i]


# ---------------------------------------------------------------------------
# Market head layout
# ---------------------------------------------------------------------------
# The market action is factored into independent decisions, each with a
# presence (binary) label and, where relevant, a quantity-bucket label.
#   - SELL <item>          for every MARKET_ITEM
#   - BUY_PRODUCT <item>   for WHEAT, FERTILIZER
#   - BUY_SEED <crop>      for every CROP
#   - BUY_ANIMAL <animal>  for every ANIMAL
#   - HIRE                 count categorical 0..MAX_HIRE
#   - BUY_LAND             binary
MAX_HIRE = 14  # categorical 0..MAX_HIRE inclusive

# Fixed decode priority: cash-in first (SELL), then land/hire (capacity),
# then restock (buys).  Within a group, follow the list order.
SELL_SLOTS = list(MARKET_ITEMS)
BUY_PRODUCT_SLOTS = list(BUY_PRODUCTS)
BUY_SEED_SLOTS = list(CROPS)
BUY_ANIMAL_SLOTS = list(ANIMALS)

# Decode ordering of order *kinds* when assembling the capped market list.
MARKET_DECODE_ORDER = ["SELL", "BUY_LAND", "HIRE",
                       "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT"]
