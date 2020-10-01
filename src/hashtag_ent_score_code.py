from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


# generating features from Entities
all_entities = list(df.entities.values)

actual_text = []
entity_text = []

count = 0
for line in all_entities:
    total_str_act = ""
    total_str_ent = ""
    count += 1
    # print("line is", line)
    if not "null;" in line:
        for entity in line.split(";"):
            # print("each entity in line is", entity)
            if len(entity) > 0:
                act_text = entity.split(":")[0]
                ent_text = entity.split(":")[1]
                act_text = re.sub("[^A-Za-z]+", " ", act_text)
                ent_text = re.sub("[^A-Za-z]+", " ", ent_text)
                act_text = act_text.strip().lower()
                ent_text = ent_text.strip().lower()
                total_str_act = total_str_act + " " + act_text
                total_str_ent = total_str_ent + " " + ent_text
        actual_text.append(total_str_act)
        entity_text.append(total_str_ent)
    else:
        actual_text.append("null")
        entity_text.append("null")

df["tweet_act_text"] = actual_text
df["tweet_ent_text"] = entity_text

df["tweet_act_text"] = df["tweet_act_text"].fillna("")
df["tweet_ent_text"] = df["tweet_ent_text"].fillna("")


df["hashtags"] = np.where(df["hashtags"] == "null;", "", df["hashtags"])
df["hashtags"] = np.where(df["hashtags"].isnull(), "", df["hashtags"])
grouped_hashtags = (
    df.groupby("Date")["hashtags"].apply(lambda x: " ".join(x)).reset_index()
)

# vectorizer = CountVectorizer(token_pattern='(?u)[^ ]+',max_features=2500)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(grouped_hashtags["hashtags"])
hashtag_age = X.todense()

hashtag_age = np.where(hashtag_age > 0, 1, 0)
hastag_life = csr_matrix(hashtag_age.sum(axis=0))
for i in range(1, len(hashtag_age)):
    temp_old = hashtag_age[i - 1]
    temp = hashtag_age[i]
    new_temp = np.where(temp_old > 0, temp_old + 1, 0)
    new_temp = np.where((temp_old == 0) & (temp == 1), 1, new_temp)
    hashtag_age[i] = new_temp

weight_hashtag = np.where(hashtag_age > 0, 1 / np.log1p(hashtag_age), 0)

hashtag_age = csr_matrix(hashtag_age)
weight_hashtag = csr_matrix(weight_hashtag)
Y = vectorizer.transform(df["hashtags"])
Y_bool = Y.astype(bool)
df["avg_score"] = 0
df["max_score"] = 0
df["count_for_score"] = 0
df["avg_weighted_score"] = 0
df["max_weighted_score"] = 0
df["avg_hashtag_age"] = 0
df["max_hashtag_age"] = 0
df["min_hashtag_age"] = 0
df["max_hashtag_life"] = 0
df["avg_hashtag_life"] = 0

for date in df["Date"].unique():
    #     print(date)
    row_index = df["Date"] == date
    val_index = grouped_hashtags["Date"] == date
    Z = Y_bool[row_index.values].multiply(X[val_index.values])
    Z_weighted = Z.multiply(weight_hashtag[val_index.values])
    weights = Y_bool[row_index.values].multiply(weight_hashtag[val_index.values])
    Z_hashtag = Y_bool[row_index.values].multiply(hashtag_age[val_index.values])
    Z_hastag_life = Y_bool[row_index.values].multiply(hastag_life)
    #     (x,y,z)=scipy.sparse.find(Z)
    #     countings=np.bincount(x)
    #     sums=np.bincount(x,weights=z)
    #     averages=sums/countings
    sums = Z.sum(axis=1).A1
    sum_weighted = Z_weighted.sum(axis=1).A1
    weight = weights.sum(axis=1).A1
    counts = np.diff(Z.indptr)
    averages = sums / counts
    df["avg_score"].iloc[row_index.values] = averages
    df["max_score"].iloc[row_index.values] = Z.max(axis=1).A.ravel()
    df["avg_weighted_score"].iloc[row_index.values] = sum_weighted / weight
    df["max_weighted_score"].iloc[row_index.values] = Z_weighted.max(axis=1).A.ravel()
    df["avg_hashtag_age"].iloc[row_index.values] = Z_hashtag.sum(axis=1).A1 / np.diff(
        Z_hashtag.indptr
    )
    df["max_hashtag_age"].iloc[row_index.values] = Z_hashtag.max(axis=1).A.ravel()
    df["min_hashtag_age"].iloc[row_index.values] = weights.max(axis=1).A.ravel()
    df["max_hashtag_life"].iloc[row_index.values] = Z_hastag_life.max(axis=1).A.ravel()
    df["avg_hashtag_life"].iloc[row_index.values] = Z_hastag_life.sum(
        axis=1
    ).A1 / np.diff(Z_hastag_life.indptr)
    #     df['min_score'].iloc[row_index.values] = Z.min(axis=1).A.ravel()
    df["count_for_score"].iloc[row_index.values] = counts
df["avg_score"] = np.where(df["avg_score"].isnull(), 0, df["avg_score"])
df["max_score"] = np.where(df["max_score"].isnull(), 0, df["max_score"])
df["count_for_score"] = np.where(
    df["count_for_score"].isnull(), 0, df["count_for_score"]
)
df["avg_weighted_score"] = np.where(
    df["avg_weighted_score"].isnull(), 0, df["avg_weighted_score"]
)
df["max_weighted_score"] = np.where(
    df["max_weighted_score"].isnull(), 0, df["max_weighted_score"]
)
df["avg_hashtag_age"] = np.where(
    df["avg_hashtag_age"].isnull(), 0, df["avg_hashtag_age"]
)
df["max_hashtag_age"] = np.where(
    df["max_hashtag_age"].isnull(), 0, df["max_hashtag_age"]
)
df["min_hashtag_age"] = np.where(
    df["min_hashtag_age"].isnull(), 0, df["min_hashtag_age"]
)
df["max_hashtag_life"] = np.where(
    df["max_hashtag_life"].isnull(), 0, df["max_hashtag_life"]
)
df["avg_hashtag_life"] = np.where(
    df["avg_hashtag_life"].isnull(), 0, df["avg_hashtag_life"]
)


# df['tweet_ent_text'] = df['tweet_ent_text'].str.findall('\w{2,}').str.join(' ')
df["tweet_ent_text"] = np.where(
    df["tweet_ent_text"] == "null", "", df["tweet_ent_text"]
)
df["tweet_ent_text"] = np.where(df["tweet_ent_text"].isnull(), "", df["tweet_ent_text"])
# df['tweet_ent_text'] = df['tweet_ent_text'].str.findall('\w{2,}').str.join(' ')
grouped_hashtags = (
    df.groupby("Date")["tweet_ent_text"].apply(lambda x: " ".join(x)).reset_index()
)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(grouped_hashtags["tweet_ent_text"])
ent_age = X.todense()

ent_age = np.where(ent_age > 0, 1, 0)

for i in range(1, len(ent_age)):
    temp_old = ent_age[i - 1]
    temp = ent_age[i]
    new_temp = np.where(temp_old > 0, temp_old + 1, 0)
    new_temp = np.where((temp_old == 0) & (temp == 1), 1, new_temp)
    ent_age[i] = new_temp

weight_ent = np.where(ent_age > 0, 1 / np.log1p(ent_age), 0)

ent_age = csr_matrix(ent_age)
weight_ent = csr_matrix(weight_ent)
# B = X.todense()

Y = vectorizer.transform(df["tweet_ent_text"])
Y_bool = Y.astype(bool)

df["avg_score_tweet_ent_text"] = 0
df["max_score_tweet_ent_text"] = 0
df["count_for_score_tweet_ent_text"] = 0
df["avg_weighted_score_ent_text"] = 0
df["max_weighted_score_ent_text"] = 0
df["avg_hashtag_age_ent_text"] = 0
df["max_hashtag_age_ent_text"] = 0
df["min_hashtag_age_ent_text"] = 0

for date in df["Date"].unique():
    #     print(date)
    row_index = df["Date"] == date
    val_index = grouped_hashtags["Date"] == date
    Z = Y[row_index.values].multiply(X[val_index.values])
    Z = Y_bool[row_index.values].multiply(X[val_index.values])
    Z_weighted = Z.multiply(weight_ent[val_index.values])
    weights = Y_bool[row_index.values].multiply(weight_ent[val_index.values])
    Z_hashtag = Y_bool[row_index.values].multiply(ent_age[val_index.values])

    #     (x,y,z)=scipy.sparse.find(Z)
    #     countings=np.bincount(x)
    #     sums=np.bincount(x,weights=z)
    #     averages=sums/countings
    sums = Z.sum(axis=1).A1
    sum_weighted = Z_weighted.sum(axis=1).A1
    weight = weights.sum(axis=1).A1
    counts = np.diff(Z.indptr)
    averages = sums / counts

    df["avg_score_tweet_ent_text"].iloc[row_index.values] = averages
    df["max_score_tweet_ent_text"].iloc[row_index.values] = Z.max(axis=1).A.ravel()
    df["avg_weighted_score_ent_text"].iloc[row_index.values] = sum_weighted / weight
    df["max_weighted_score_ent_text"].iloc[row_index.values] = Z_weighted.max(
        axis=1
    ).A.ravel()
    df["avg_hashtag_age_ent_text"].iloc[row_index.values] = Z_hashtag.sum(
        axis=1
    ).A1 / np.diff(Z_hashtag.indptr)
    df["max_hashtag_age_ent_text"].iloc[row_index.values] = Z_hashtag.max(
        axis=1
    ).A.ravel()
    df["min_hashtag_age_ent_text"].iloc[row_index.values] = weights.max(
        axis=1
    ).A.ravel()
    #     df['min_score'].iloc[row_index.values] = Z.min(axis=1).A.ravel()
    df["count_for_score_tweet_ent_text"].iloc[row_index.values] = counts
df["avg_score_tweet_ent_text"] = np.where(
    df["avg_score_tweet_ent_text"].isnull(), 0, df["avg_score_tweet_ent_text"]
)
df["max_score_tweet_ent_text"] = np.where(
    df["max_score_tweet_ent_text"].isnull(), 0, df["max_score_tweet_ent_text"]
)
df["count_for_score_tweet_ent_text"] = np.where(
    df["count_for_score_tweet_ent_text"].isnull(),
    0,
    df["count_for_score_tweet_ent_text"],
)
df["avg_weighted_score_ent_text"] = np.where(
    df["avg_weighted_score_ent_text"].isnull(), 0, df["avg_weighted_score_ent_text"]
)
df["max_weighted_score_ent_text"] = np.where(
    df["max_weighted_score_ent_text"].isnull(), 0, df["max_weighted_score_ent_text"]
)
df["avg_hashtag_age_ent_text"] = np.where(
    df["avg_hashtag_age_ent_text"].isnull(), 0, df["avg_hashtag_age_ent_text"]
)
df["max_hashtag_age_ent_text"] = np.where(
    df["max_hashtag_age_ent_text"].isnull(), 0, df["max_hashtag_age_ent_text"]
)
df["min_hashtag_age_ent_text"] = np.where(
    df["min_hashtag_age_ent_text"].isnull(), 0, df["min_hashtag_age_ent_text"]
)

del Z, Z_weighted, weights, Z_hashtag, ent_age, weight_ent, Z_hastag_life, Y, Y_boolool


feat_to_use = [
    "avg_weighted_score",
    "max_weighted_score",
    "avg_hashtag_age",
    "max_hashtag_age",
    "min_hashtag_age",
    "max_hashtag_life",
    "avg_hashtag_life",
    #            'avg_weighted_score_ent_text', 'max_weighted_score_ent_text', 'avg_hashtag_age_ent_text',
    #        'max_hashtag_age_ent_text', 'min_hashtag_age_ent_text',
    #            'fav_hashtag_avg',
    #        'fav_hashtag_mx', 'fav_hashtag_avg_div', 'fav_hashtag_max_div',
    "max_score",
    "count_for_score",
    "avg_score",
]
