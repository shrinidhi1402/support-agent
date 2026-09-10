import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

brands = [
    "SpotifyCares",
    "TMobileHelp",
    "comcastcares",
    "AmericanAir",
    "Delta",
    "AmazonHelp"
]

for brand in brands:

    brand_tweets = df[df["author_id"] == brand]

    # Customer tweets directly replying to this brand
    customer_replies = df[
        df["in_response_to_tweet_id"].isin(brand_tweets["tweet_id"])
        & (df["inbound"] == True)
    ]

    # Brand responses that have a customer reply
    brand_tweets_with_replies = brand_tweets[
        brand_tweets["response_tweet_id"].notna()
    ]

    print("\n" + "=" * 50)
    print("BRAND:", brand)
    print("Brand tweets:", len(brand_tweets))
    print("Customer tweets replying to brand:", len(customer_replies))
    print(
        "Brand tweets with response links:",
        len(brand_tweets_with_replies)
    )

    print("\nCustomer examples:")
    for text in customer_replies["text"].dropna().head(5):
        print("-", text[:250])