import requests

TAG = "ENCHANTED_GOLD"

hyp = requests.get("https://api.hypixel.net/v2/skyblock/bazaar", timeout=30).json()
product = hyp["products"][TAG]

print("Hypixel quick_status buyPrice :", product["quick_status"]["buyPrice"])
print("Hypixel quick_status sellPrice:", product["quick_status"]["sellPrice"])

print("\nHypixel sell_summary (first 5 orders):")
for order in product["sell_summary"][:5]:
    print(" ", order)

print("\nHypixel buy_summary (first 5 orders):")
for order in product["buy_summary"][:5]:
    print(" ", order)

cofl = requests.get(f"https://sky.coflnet.com/api/bazaar/{TAG}/snapshot", timeout=30).json()
print("\nCoflnet buyPrice :", cofl.get("buyPrice"))
print("Coflnet sellPrice:", cofl.get("sellPrice"))