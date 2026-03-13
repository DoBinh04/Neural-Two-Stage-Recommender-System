def build_interactions(df):

    weights = {
        "view": 0.02,
        "addtocart": 0.3,
        "transaction": 1.0
    }

    df["weight"] = df["event"].map(weights)

    interactions = df[["user_id","item_id","timestamp","weight", "event"]]

    return interactions