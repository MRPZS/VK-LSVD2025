import os
import polars as pl
import numpy as np
from tqdm import tqdm
import faiss
from collections import defaultdict
import lightgbm as lgb
from typing import Dict, Tuple, List
import gc

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'data_dir': './VK-LSVD',
    'train_files': [f'subsamples/up0.01_ip0.01/train/week_{i:02}.parquet' for i in range(0, 24)],
    'val_files': ['subsamples/up0.01_ip0.01/validation/week_25.parquet'],
    'meta_users': 'metadata/users_metadata.parquet',
    'meta_items': 'metadata/items_metadata.parquet',
    'emb_file': 'metadata/item_embeddings.npz',
    'submission_file': 'metadata/submission.parquet',
}

EMBEDDING_DIM = 64
CANDIDATE_POOL_SIZE = 500  # Number of candidates from FAISS for re-ranking
FINAL_TOP_K = 100  # Final number of users per item
MAX_USER_APPEARANCES = 100  # Anti-spam constraint

# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================
def load_all_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[int, np.ndarray], pl.DataFrame, pl.DataFrame, np.ndarray]:
    """
    Load all data files and return them.

    Returns:
        train_interactions: Training data for building user profiles (weeks 0-23)
        val_interactions: Validation data for training LightGBM ranker (week 25)
        users_metadata: User demographic information
        item_embeddings_map: Dict mapping item_id to embedding vector
        items_metadata: Item metadata
        submission_df: Submission template with target item_ids
        item_embeddings_matrix: Raw embedding matrix
    """
    print("--- Loading data ---")

    data_dir = CONFIG['data_dir']

    # Load training interactions (weeks 0-23 for user profiles)
    train_files = [os.path.join(data_dir, f) for f in CONFIG['train_files']]
    existing_train_files = [f for f in train_files if os.path.exists(f)]

    if not existing_train_files:
        raise FileNotFoundError(f"No training files found. Expected files in: {data_dir}/subsamples/up0.01_ip0.01/train/")

    print(f"Loading {len(existing_train_files)} training files...")
    train_interactions = pl.concat([pl.read_parquet(f) for f in tqdm(existing_train_files, desc="Loading training data")])

    # Load validation interactions (week 25 for LightGBM training)
    val_files = [os.path.join(data_dir, f) for f in CONFIG['val_files']]
    existing_val_files = [f for f in val_files if os.path.exists(f)]

    if existing_val_files:
        print(f"Loading {len(existing_val_files)} validation files...")
        val_interactions = pl.concat([pl.read_parquet(f) for f in tqdm(existing_val_files, desc="Loading validation data")])
    else:
        print("Warning: No validation files found, using last portion of training data")
        val_interactions = train_interactions

    # Load embeddings
    embedding_file = os.path.join(data_dir, CONFIG['emb_file'])
    embeddings_data = np.load(embedding_file)
    item_ids = embeddings_data['item_id']
    item_embeddings_matrix = embeddings_data['embedding'].astype(np.float32)
    item_embeddings_map = {int(id_): emb for id_, emb in zip(item_ids, item_embeddings_matrix)}

    # Load user metadata
    users_metadata = pl.read_parquet(os.path.join(data_dir, CONFIG['meta_users']))

    # Load item metadata
    items_metadata = pl.read_parquet(os.path.join(data_dir, CONFIG['meta_items']))

    # Load submission template
    submission_df = pl.read_parquet(os.path.join(data_dir, CONFIG['submission_file']), columns=['item_id'])

    print(f"Loaded {len(train_interactions)} training interactions")
    print(f"Loaded {len(val_interactions)} validation interactions")
    print(f"Loaded {len(item_embeddings_map)} item embeddings")
    print(f"Loaded {len(users_metadata)} user profiles")
    print(f"Loaded {len(items_metadata)} item profiles")
    print(f"Target items for submission: {len(submission_df)}")

    return train_interactions, val_interactions, users_metadata, item_embeddings_map, items_metadata, submission_df, item_embeddings_matrix


# ==============================================================================
# SECTION 2: USER PROFILE BUILDING
# ==============================================================================
def build_user_profiles(interactions: pl.DataFrame, embeddings_map: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create user profiles as weighted average of item embeddings they interacted with.
    Weight is based on timespent.
    """
    print("\n--- Step 2: Building user profiles (weighted average method) ---")

    # Get top 100 interactions per user by timespent
    top_interactions = interactions.sort("timespent", descending=True).group_by("user_id").head(100)

    # Group to collect items and timespent for each user
    user_top_content = top_interactions.group_by('user_id').agg(
        pl.struct(['item_id', 'timespent']).alias('top_content')
    )

    user_profiles = []
    profiled_user_ids = []

    for row in tqdm(user_top_content.iter_rows(named=True), desc="Computing weighted embeddings", total=len(user_top_content)):
        embeddings_to_average = []
        weights = []

        for interaction in row['top_content']:
            item_id = interaction['item_id']
            timespent = interaction['timespent']

            if item_id in embeddings_map:
                embeddings_to_average.append(embeddings_map[item_id])
                weights.append(timespent + 1)  # +1 to avoid zero weights

        if embeddings_to_average:
            profile_emb = np.average(
                np.array(embeddings_to_average, dtype=np.float32),
                axis=0,
                weights=np.array(weights, dtype=np.float32)
            )
            user_profiles.append(profile_emb)
            profiled_user_ids.append(row['user_id'])

    print(f"Created {len(profiled_user_ids)} user profiles.")
    return np.array(profiled_user_ids, dtype=np.uint32), np.array(user_profiles, dtype=np.float32)


# ==============================================================================
# SECTION 3: USER STATISTICS COMPUTATION
# ==============================================================================
def compute_user_statistics(interactions: pl.DataFrame) -> pl.DataFrame:
    """Compute aggregate statistics for each user."""
    print("\n--- Computing user statistics ---")

    user_stats = interactions.group_by('user_id').agg([
        pl.col('timespent').mean().alias('avg_timespent'),
        pl.col('timespent').max().alias('max_timespent'),
        pl.col('timespent').sum().alias('total_timespent'),
        pl.col('like').mean().alias('like_rate'),
        pl.col('dislike').mean().alias('dislike_rate'),
        pl.col('share').mean().alias('share_rate'),
        pl.col('bookmark').mean().alias('bookmark_rate'),
        pl.col('click_on_author').mean().alias('click_author_rate'),
        pl.col('open_comments').mean().alias('open_comments_rate'),
        pl.len().alias('interaction_count'),
        pl.col('place').n_unique().alias('place_diversity'),
        pl.col('platform').n_unique().alias('platform_diversity'),
    ])

    return user_stats


def compute_item_statistics(interactions: pl.DataFrame) -> pl.DataFrame:
    """Compute aggregate statistics for each item."""
    print("--- Computing item statistics ---")

    item_stats = interactions.group_by('item_id').agg([
        pl.col('timespent').mean().alias('item_avg_timespent'),
        pl.col('like').mean().alias('item_like_rate'),
        pl.col('dislike').mean().alias('item_dislike_rate'),
        pl.col('share').mean().alias('item_share_rate'),
        pl.col('bookmark').mean().alias('item_bookmark_rate'),
        pl.len().alias('item_interaction_count'),
    ])

    return item_stats


# ==============================================================================
# SECTION 4: FEATURE ENGINEERING FOR LIGHTGBM
# ==============================================================================
def create_training_features(
    interactions: pl.DataFrame,
    user_profiles_map: Dict[int, np.ndarray],
    item_embeddings_map: Dict[int, np.ndarray],
    users_metadata: pl.DataFrame,
    items_metadata: pl.DataFrame,
    user_stats: pl.DataFrame,
    item_stats: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training features for LightGBM ranker.

    Features include:
    - Cosine similarity between user profile and item embedding
    - User demographic features
    - User engagement statistics
    - Item characteristics
    """
    print("\n--- Creating training features ---")

    # Sample interactions for training (to avoid memory issues)
    # Focus on positive examples (high engagement) and negative examples
    positive_interactions = interactions.filter(
        (pl.col('timespent') >= 30) |
        pl.col('like') |
        pl.col('bookmark') |
        pl.col('share')
    ).sample(fraction=min(1.0, 500000 / len(interactions)), seed=42)

    negative_interactions = interactions.filter(
        (pl.col('timespent') < 10) &
        ~pl.col('like') &
        ~pl.col('dislike')
    ).sample(fraction=min(1.0, 500000 / len(interactions)), seed=42)

    training_data = pl.concat([positive_interactions, negative_interactions])

    # Join with metadata
    training_data = training_data.join(users_metadata, on='user_id', how='left')
    training_data = training_data.join(items_metadata.select(['item_id', 'duration', 'author_id']), on='item_id', how='left')
    training_data = training_data.join(user_stats, on='user_id', how='left')
    training_data = training_data.join(item_stats, on='item_id', how='left')

    print(f"Training data size: {len(training_data)}")

    features_list = []
    labels_list = []
    groups_list = []

    # Group by item_id for ranking
    grouped = training_data.group_by('item_id').agg(pl.all())

    for row in tqdm(grouped.iter_rows(named=True), desc="Creating features", total=len(grouped)):
        item_id = row['item_id']

        if item_id not in item_embeddings_map:
            continue

        item_emb = item_embeddings_map[item_id]
        item_emb_norm = item_emb / (np.linalg.norm(item_emb) + 1e-8)

        user_ids = row['user_id']
        timespents = row['timespent']
        likes = row['like']
        bookmarks = row['bookmark']
        shares = row['share']

        # User features from metadata
        ages = row['age'] if 'age' in row else [0] * len(user_ids)
        genders = row['gender'] if 'gender' in row else [0] * len(user_ids)
        geos = row['geo'] if 'geo' in row else [0] * len(user_ids)

        # User statistics
        avg_timespents = row['avg_timespent'] if 'avg_timespent' in row else [0] * len(user_ids)
        like_rates = row['like_rate'] if 'like_rate' in row else [0] * len(user_ids)
        bookmark_rates = row['bookmark_rate'] if 'bookmark_rate' in row else [0] * len(user_ids)
        interaction_counts = row['interaction_count'] if 'interaction_count' in row else [0] * len(user_ids)

        # Item features
        durations = row['duration'] if 'duration' in row else [0] * len(user_ids)
        item_like_rates = row['item_like_rate'] if 'item_like_rate' in row else [0] * len(user_ids)

        group_features = []
        group_labels = []

        for i, user_id in enumerate(user_ids):
            if user_id not in user_profiles_map:
                continue

            user_profile = user_profiles_map[user_id]
            user_profile_norm = user_profile / (np.linalg.norm(user_profile) + 1e-8)

            # Compute cosine similarity
            cosine_sim = np.dot(user_profile_norm, item_emb_norm)

            # Create feature vector
            feature = [
                cosine_sim,
                ages[i] if ages[i] is not None else 0,
                genders[i] if genders[i] is not None else 0,
                geos[i] if geos[i] is not None else 0,
                avg_timespents[i] if avg_timespents[i] is not None else 0,
                like_rates[i] if like_rates[i] is not None else 0,
                bookmark_rates[i] if bookmark_rates[i] is not None else 0,
                interaction_counts[i] if interaction_counts[i] is not None else 0,
                durations[i] if durations[i] is not None else 0,
                item_like_rates[i] if item_like_rates[i] is not None else 0,
            ]

            # Create label (relevance score)
            # Higher weight for explicit positive actions
            label = 0
            if likes[i]:
                label += 3
            if bookmarks[i]:
                label += 2
            if shares[i]:
                label += 2
            if timespents[i] >= 30:
                label += 1
            elif timespents[i] >= 60:
                label += 2

            group_features.append(feature)
            group_labels.append(label)

        if len(group_features) >= 2:  # Need at least 2 samples per group for ranking
            features_list.extend(group_features)
            labels_list.extend(group_labels)
            groups_list.append(len(group_features))

    print(f"Created {len(features_list)} training samples in {len(groups_list)} groups")

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32), np.array(groups_list, dtype=np.int32)


# ==============================================================================
# SECTION 5: LIGHTGBM RANKER TRAINING
# ==============================================================================
def train_lightgbm_ranker(X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray) -> lgb.Booster:
    """Train a LightGBM ranker model."""
    print("\n--- Training LightGBM Ranker ---")

    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train, group=groups)

    # LightGBM ranker parameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 50, 100],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'num_threads': -1,
        'verbose': -1,
    }

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(period=50)]
    )

    print("LightGBM Ranker training complete.")
    return model


# ==============================================================================
# SECTION 6: CANDIDATE GENERATION AND RE-RANKING
# ==============================================================================
def generate_candidates_faiss(
    target_item_ids: List[int],
    item_embeddings_map: Dict[int, np.ndarray],
    user_ids: np.ndarray,
    user_profiles: np.ndarray,
    top_k: int = CANDIDATE_POOL_SIZE
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Use FAISS to generate candidate users for each target item.
    Returns a dict mapping item_id to list of (user_id, similarity_score) tuples.
    """
    print("\n--- Generating candidates using FAISS ---")

    # Build FAISS index with user profiles
    embedding_dim = user_profiles.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)

    # Normalize user profiles for cosine similarity
    user_profiles_normalized = user_profiles.copy()
    faiss.normalize_L2(user_profiles_normalized)
    index.add(user_profiles_normalized)

    # Prepare target item embeddings
    target_embeddings = []
    valid_target_ids = []

    for item_id in target_item_ids:
        if item_id in item_embeddings_map:
            target_embeddings.append(item_embeddings_map[item_id])
            valid_target_ids.append(item_id)
        else:
            target_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            valid_target_ids.append(item_id)

    target_embeddings = np.array(target_embeddings, dtype=np.float32)
    faiss.normalize_L2(target_embeddings)

    # Search for nearest users
    print(f"Searching for top-{top_k} candidates per item...")
    similarities, indices = index.search(target_embeddings, top_k)

    # Create candidate dict
    candidates = {}
    for i, item_id in enumerate(valid_target_ids):
        candidates[item_id] = [
            (int(user_ids[idx]), float(sim))
            for idx, sim in zip(indices[i], similarities[i])
        ]

    return candidates


def rerank_candidates_lightgbm(
    candidates: Dict[int, List[Tuple[int, float]]],
    model: lgb.Booster,
    user_profiles_map: Dict[int, np.ndarray],
    item_embeddings_map: Dict[int, np.ndarray],
    users_metadata: pl.DataFrame,
    user_stats: pl.DataFrame,
    items_metadata: pl.DataFrame,
    item_stats: pl.DataFrame
) -> Dict[int, List[int]]:
    """
    Re-rank candidates using the LightGBM ranker.
    """
    print("\n--- Re-ranking candidates with LightGBM ---")

    # Create lookup dicts for metadata
    user_meta_dict = {row['user_id']: row for row in users_metadata.iter_rows(named=True)}
    user_stats_dict = {row['user_id']: row for row in user_stats.iter_rows(named=True)}
    item_meta_dict = {row['item_id']: row for row in items_metadata.iter_rows(named=True)}
    item_stats_dict = {row['item_id']: row for row in item_stats.iter_rows(named=True)}

    reranked = {}

    for item_id in tqdm(candidates.keys(), desc="Re-ranking"):
        item_candidates = candidates[item_id]

        if item_id not in item_embeddings_map:
            # Fallback to original order
            reranked[item_id] = [uid for uid, _ in item_candidates]
            continue

        item_emb = item_embeddings_map[item_id]
        item_emb_norm = item_emb / (np.linalg.norm(item_emb) + 1e-8)

        # Get item metadata
        item_meta = item_meta_dict.get(item_id, {})
        item_stat = item_stats_dict.get(item_id, {})

        features = []
        valid_users = []

        for user_id, cosine_sim in item_candidates:
            if user_id not in user_profiles_map:
                continue

            user_profile = user_profiles_map[user_id]
            user_profile_norm = user_profile / (np.linalg.norm(user_profile) + 1e-8)

            # Recompute cosine similarity for accuracy
            cosine_sim = np.dot(user_profile_norm, item_emb_norm)

            # Get user metadata and stats
            user_meta = user_meta_dict.get(user_id, {})
            user_stat = user_stats_dict.get(user_id, {})

            feature = [
                cosine_sim,
                user_meta.get('age', 0) or 0,
                user_meta.get('gender', 0) or 0,
                user_meta.get('geo', 0) or 0,
                user_stat.get('avg_timespent', 0) or 0,
                user_stat.get('like_rate', 0) or 0,
                user_stat.get('bookmark_rate', 0) or 0,
                user_stat.get('interaction_count', 0) or 0,
                item_meta.get('duration', 0) or 0,
                item_stat.get('item_like_rate', 0) or 0,
            ]

            features.append(feature)
            valid_users.append(user_id)

        if features:
            features_array = np.array(features, dtype=np.float32)
            scores = model.predict(features_array)

            # Sort by predicted score
            sorted_indices = np.argsort(-scores)
            reranked[item_id] = [valid_users[i] for i in sorted_indices]
        else:
            reranked[item_id] = [uid for uid, _ in item_candidates]

    return reranked


# ==============================================================================
# SECTION 7: ANTI-SPAM CONSTRAINT APPLICATION
# ==============================================================================
def apply_antispam_constraints(
    reranked_candidates: Dict[int, List[int]],
    target_item_ids: List[int],
    fallback_users: np.ndarray,
    top_k: int = FINAL_TOP_K,
    max_appearances: int = MAX_USER_APPEARANCES
) -> List[List[int]]:
    """
    Apply anti-spam constraints: each user can appear at most max_appearances times.
    Uses fair iterative distribution.
    """
    print("\n--- Applying anti-spam constraints (fair distribution) ---")

    user_counts = defaultdict(int)
    num_items = len(target_item_ids)
    final_predictions = [[] for _ in range(num_items)]
    final_predictions_sets = [set() for _ in range(num_items)]

    # Get candidate lists for each item
    candidate_lists = []
    for item_id in target_item_ids:
        candidates = reranked_candidates.get(item_id, [])
        candidate_lists.append(candidates)

    # Find max candidate list length
    max_candidates = max(len(c) for c in candidate_lists) if candidate_lists else 0

    # Iterative fair distribution
    for rank in tqdm(range(max_candidates), desc="Distributing by rank"):
        for item_idx in range(num_items):
            if len(final_predictions[item_idx]) >= top_k:
                continue

            if rank < len(candidate_lists[item_idx]):
                candidate_user = candidate_lists[item_idx][rank]
                if user_counts[candidate_user] < max_appearances and candidate_user not in final_predictions_sets[item_idx]:
                    final_predictions[item_idx].append(candidate_user)
                    final_predictions_sets[item_idx].add(candidate_user)
                    user_counts[candidate_user] += 1

    # Fallback for remaining slots
    print("--- Applying fallback for remaining slots ---")
    fallback_pointer = 0

    for i in tqdm(range(num_items), desc="Fallback"):
        while len(final_predictions[i]) < top_k:
            if fallback_pointer >= len(fallback_users):
                fallback_pointer = 0

            candidate_user = int(fallback_users[fallback_pointer])
            fallback_pointer += 1

            if user_counts[candidate_user] < max_appearances and candidate_user not in final_predictions_sets[i]:
                final_predictions[i].append(candidate_user)
                final_predictions_sets[i].add(candidate_user)
                user_counts[candidate_user] += 1

    return final_predictions


# ==============================================================================
# SECTION 8: SUBMISSION CREATION
# ==============================================================================
def create_submission(
    submission_df: pl.DataFrame,
    final_predictions: List[List[int]],
    output_filename: str = 'content_based_submission.parquet'
):
    """Create and save the final submission file."""
    print("\n--- Creating submission file ---")

    final_predictions_np = np.array(final_predictions, dtype=np.uint32)
    submission_result = submission_df.with_columns(
        pl.Series(name='user_id', values=final_predictions_np.tolist())
    )

    submission_result.write_parquet(output_filename)

    print(f"\nSubmission file '{output_filename}' created successfully!")
    print(f"Shape: {submission_result.shape}")
    print(f"Schema: {submission_result.schema}")

    return submission_result


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("VIDEO RECOMMENDATION SYSTEM - LightGBM Ranker Pipeline")
    print("=" * 70)

    # Step 1: Load all data
    train_interactions, val_interactions, users_metadata, item_embeddings_map, items_metadata, submission_df, item_embeddings_matrix = load_all_data()

    # Step 2: Build user profiles from training data (weeks 0-23)
    profiled_user_ids, user_profiles_matrix = build_user_profiles(train_interactions, item_embeddings_map)
    user_profiles_map = {int(uid): profile for uid, profile in zip(profiled_user_ids, user_profiles_matrix)}

    # Step 3: Compute statistics from training data
    user_stats = compute_user_statistics(train_interactions)
    item_stats = compute_item_statistics(train_interactions)

    # Step 4: Create training features for LightGBM using validation data (week 25)
    # This simulates the "future" data that the model should predict
    X_train, y_train, groups = create_training_features(
        val_interactions,  # Use validation data for LightGBM training
        user_profiles_map,
        item_embeddings_map,
        users_metadata,
        items_metadata,
        user_stats,
        item_stats
    )

    # Step 5: Train LightGBM ranker
    ranker_model = train_lightgbm_ranker(X_train, y_train, groups)

    # Clean up training data
    del X_train, y_train, groups
    gc.collect()

    # Step 6: Generate candidates using FAISS
    target_item_ids = submission_df['item_id'].to_list()
    candidates = generate_candidates_faiss(
        target_item_ids,
        item_embeddings_map,
        profiled_user_ids,
        user_profiles_matrix,
        top_k=CANDIDATE_POOL_SIZE
    )

    # Step 7: Re-rank candidates using LightGBM
    reranked_candidates = rerank_candidates_lightgbm(
        candidates,
        ranker_model,
        user_profiles_map,
        item_embeddings_map,
        users_metadata,
        user_stats,
        items_metadata,
        item_stats
    )

    # Step 8: Apply anti-spam constraints
    # Use all interactions for popularity fallback
    all_interactions = pl.concat([train_interactions, val_interactions])
    popular_users = all_interactions.group_by('user_id').count().sort('count', descending=True).get_column('user_id').to_numpy()
    final_predictions = apply_antispam_constraints(
        reranked_candidates,
        target_item_ids,
        popular_users,
        top_k=FINAL_TOP_K,
        max_appearances=MAX_USER_APPEARANCES
    )

    # Step 9: Create submission
    create_submission(submission_df, final_predictions)

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
