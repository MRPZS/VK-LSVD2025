import os
import polars as pl
import numpy as np
from tqdm import tqdm
import faiss
from collections import defaultdict
from typing import Dict, Tuple, List
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

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

# Model hyperparameters - OPTIMIZED for speed
MODEL_CONFIG = {
    'embedding_dim': 64,
    'hidden_dim': 256,
    'dropout': 0.1,
    'learning_rate': 3e-4,
    'batch_size': 1024,            # Smaller batch = easier learning
    'num_epochs': 10,
    'max_train_samples': 300000,
    'temperature': 0.1,            # Higher = more stable gradients
    'item_emb_dim': 64,
}

CANDIDATE_POOL_SIZE = 1000
FINAL_TOP_K = 100
MAX_USER_APPEARANCES = 100

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================
def load_all_data():
    """Load all data files."""
    print("--- Loading data ---")
    data_dir = CONFIG['data_dir']

    train_files = [os.path.join(data_dir, f) for f in CONFIG['train_files']]
    existing_train_files = [f for f in train_files if os.path.exists(f)]
    if not existing_train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    print(f"Loading {len(existing_train_files)} training files...")
    train_interactions = pl.concat([pl.read_parquet(f) for f in tqdm(existing_train_files, desc="Loading training")])

    val_files = [os.path.join(data_dir, f) for f in CONFIG['val_files']]
    existing_val_files = [f for f in val_files if os.path.exists(f)]
    if existing_val_files:
        val_interactions = pl.concat([pl.read_parquet(f) for f in existing_val_files])
    else:
        val_interactions = train_interactions

    embedding_file = os.path.join(data_dir, CONFIG['emb_file'])
    embeddings_data = np.load(embedding_file)
    item_ids = embeddings_data['item_id']
    item_embeddings_matrix = embeddings_data['embedding'].astype(np.float32)
    item_embeddings_map = {int(id_): emb for id_, emb in zip(item_ids, item_embeddings_matrix)}

    users_metadata = pl.read_parquet(os.path.join(data_dir, CONFIG['meta_users']))
    items_metadata = pl.read_parquet(os.path.join(data_dir, CONFIG['meta_items']))
    submission_df = pl.read_parquet(os.path.join(data_dir, CONFIG['submission_file']), columns=['item_id'])

    print(f"Train: {len(train_interactions)}, Val: {len(val_interactions)}")
    print(f"Embeddings: {len(item_embeddings_map)}, Users: {len(users_metadata)}, Items: {len(items_metadata)}")

    return train_interactions, val_interactions, users_metadata, item_embeddings_map, items_metadata, submission_df


# ==============================================================================
# SECTION 2: FEATURE PREPARATION (Pre-tensorized)
# ==============================================================================
class FeatureStore:
    """Pre-compute and store all features as tensors for fast access."""

    def __init__(self, interactions: pl.DataFrame, users_metadata: pl.DataFrame,
                 items_metadata: pl.DataFrame, item_embeddings_map: Dict[int, np.ndarray]):
        print("--- Building feature store ---")

        # Get unique users and items
        all_user_ids = interactions['user_id'].unique().to_list()
        all_item_ids = interactions['item_id'].unique().to_list()

        # Create ID mappings (0 is reserved for padding/unknown)
        self.user_id_to_idx = {uid: idx + 1 for idx, uid in enumerate(all_user_ids)}
        self.item_id_to_idx = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}

        self.num_users = len(all_user_ids)
        self.num_items = len(all_item_ids)

        print(f"Users: {self.num_users}, Items: {self.num_items}")

        # Compute user behavioral stats
        print("Computing user stats...")
        user_stats = interactions.group_by('user_id').agg([
            pl.col('timespent').mean().alias('avg_timespent'),
            pl.col('like').mean().alias('like_rate'),
            pl.col('bookmark').mean().alias('bookmark_rate'),
            pl.col('share').mean().alias('share_rate'),
            pl.len().alias('n_interactions'),
        ]).join(users_metadata, on='user_id', how='left').fill_null(0)

        # Compute item stats
        print("Computing item stats...")
        item_stats = interactions.group_by('item_id').agg([
            pl.col('timespent').mean().alias('item_avg_timespent'),
            pl.col('like').mean().alias('item_like_rate'),
            pl.len().alias('item_n_interactions'),
        ]).join(items_metadata.select(['item_id', 'author_id', 'duration']), on='item_id', how='left').fill_null(0)

        # Build author mapping
        author_ids = item_stats['author_id'].unique().to_list()
        self.author_id_to_idx = {aid: idx + 1 for idx, aid in enumerate(author_ids) if aid is not None}
        self.num_authors = len(self.author_id_to_idx)

        # Pre-allocate tensors
        print("Building user tensors...")
        self._build_user_tensors(user_stats)

        print("Building item tensors...")
        self._build_item_tensors(item_stats, item_embeddings_map)

        print("Feature store ready!")

    def _build_user_tensors(self, user_stats: pl.DataFrame):
        """Pre-build all user feature tensors."""
        n = self.num_users + 1

        self.user_age = torch.zeros(n, dtype=torch.float32)
        self.user_gender = torch.zeros(n, dtype=torch.long)
        self.user_geo = torch.zeros(n, dtype=torch.long)
        self.user_features = torch.zeros(n, 5, dtype=torch.float32)  # behavioral features

        # Track max values for embedding sizes
        max_gender, max_geo = 0, 0

        for row in user_stats.iter_rows(named=True):
            idx = self.user_id_to_idx.get(row['user_id'], 0)
            if idx == 0:
                continue

            age = row.get('age', 0) or 0
            gender = int(row.get('gender', 0) or 0)
            geo = int(row.get('geo', 0) or 0)

            self.user_age[idx] = age / 70.0
            self.user_gender[idx] = gender
            self.user_geo[idx] = geo

            max_gender = max(max_gender, gender)
            max_geo = max(max_geo, geo)

            # Behavioral features (will normalize later)
            self.user_features[idx, 0] = row.get('avg_timespent', 0) or 0
            self.user_features[idx, 1] = row.get('like_rate', 0) or 0
            self.user_features[idx, 2] = row.get('bookmark_rate', 0) or 0
            self.user_features[idx, 3] = row.get('share_rate', 0) or 0
            self.user_features[idx, 4] = row.get('n_interactions', 0) or 0

        self.num_genders = max_gender + 2
        self.num_geos = max_geo + 2

        # Normalize features
        mean = self.user_features[1:].mean(dim=0)
        std = self.user_features[1:].std(dim=0) + 1e-8
        self.user_features = (self.user_features - mean) / std

    def _build_item_tensors(self, item_stats: pl.DataFrame, item_embeddings_map: Dict[int, np.ndarray]):
        """Pre-build all item feature tensors."""
        n = self.num_items + 1
        emb_dim = MODEL_CONFIG['item_emb_dim']

        self.item_author = torch.zeros(n, dtype=torch.long)
        self.item_duration = torch.zeros(n, dtype=torch.float32)
        self.item_features = torch.zeros(n, 3, dtype=torch.float32)  # popularity features
        self.item_pretrained = torch.zeros(n, emb_dim, dtype=torch.float32)

        for row in item_stats.iter_rows(named=True):
            item_id = row['item_id']
            idx = self.item_id_to_idx.get(item_id, 0)
            if idx == 0:
                continue

            author_id = row.get('author_id', 0) or 0
            self.item_author[idx] = self.author_id_to_idx.get(author_id, 0)
            self.item_duration[idx] = (row.get('duration', 0) or 0) / 255.0

            self.item_features[idx, 0] = row.get('item_avg_timespent', 0) or 0
            self.item_features[idx, 1] = row.get('item_like_rate', 0) or 0
            self.item_features[idx, 2] = row.get('item_n_interactions', 0) or 0

            if item_id in item_embeddings_map:
                self.item_pretrained[idx] = torch.from_numpy(item_embeddings_map[item_id])

        # Normalize features
        mean = self.item_features[1:].mean(dim=0)
        std = self.item_features[1:].std(dim=0) + 1e-8
        self.item_features = (self.item_features - mean) / std


# ==============================================================================
# SECTION 3: TWO-TOWER MODEL (Cold-Start Friendly)
# ==============================================================================
class UserTower(nn.Module):
    """User tower: uses user_id + demographics + behavioral features."""
    def __init__(self, num_users, num_genders, num_geos, embedding_dim, hidden_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, 32, padding_idx=0)
        self.gender_emb = nn.Embedding(num_genders, 8, padding_idx=0)
        self.geo_emb = nn.Embedding(num_geos, 16, padding_idx=0)

        # Input: user(32) + gender(8) + geo(16) + age(1) + features(5) = 62
        self.mlp = nn.Sequential(
            nn.Linear(62, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout']),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

    def forward(self, user_idx, gender, geo, age, features):
        x = torch.cat([
            self.user_emb(user_idx),
            self.gender_emb(gender),
            self.geo_emb(geo),
            age.unsqueeze(1),
            features
        ], dim=1)
        return F.normalize(self.mlp(x), p=2, dim=1)


class ItemTower(nn.Module):
    """
    Item tower for COLD-START items.
    Does NOT use item_id or author_id embeddings (they're 0 for new items).
    Relies ONLY on pretrained content embeddings + duration.
    """
    def __init__(self, pretrained_dim, embedding_dim, hidden_dim):
        super().__init__()
        # Project pretrained embeddings - this is the KEY for cold-start
        # Input: pretrained(64) + duration(1) = 65
        self.mlp = nn.Sequential(
            nn.Linear(pretrained_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout']),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

    def forward(self, pretrained, duration):
        x = torch.cat([pretrained, duration.unsqueeze(1)], dim=1)
        return F.normalize(self.mlp(x), p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, feature_store: FeatureStore):
        super().__init__()
        emb_dim = MODEL_CONFIG['embedding_dim']
        hidden_dim = MODEL_CONFIG['hidden_dim']

        self.user_tower = UserTower(
            feature_store.num_users, feature_store.num_genders,
            feature_store.num_geos, emb_dim, hidden_dim
        )
        # Item tower only uses pretrained embeddings + duration (cold-start friendly)
        self.item_tower = ItemTower(
            MODEL_CONFIG['item_emb_dim'], emb_dim, hidden_dim
        )

    def get_user_emb(self, user_idx, feature_store):
        return self.user_tower(
            user_idx,
            feature_store.user_gender[user_idx],
            feature_store.user_geo[user_idx],
            feature_store.user_age[user_idx],
            feature_store.user_features[user_idx]
        )

    def get_item_emb(self, item_idx, feature_store):
        # Only use pretrained embeddings and duration - works for new items!
        return self.item_tower(
            feature_store.item_pretrained[item_idx],
            feature_store.item_duration[item_idx]
        )


# ==============================================================================
# SECTION 4: EFFICIENT DATASET WITH IN-BATCH NEGATIVES
# ==============================================================================
class TwoTowerDataset(Dataset):
    """Efficient dataset - returns (user_idx, item_idx) pairs."""

    def __init__(self, interactions: pl.DataFrame, feature_store: FeatureStore, max_samples: int):
        # Sample positive interactions
        print(f"Sampling up to {max_samples} training pairs...")

        # Weight by engagement for sampling
        interactions = interactions.with_columns([
            (pl.col('timespent') / 255.0 +
             pl.col('like').cast(pl.Float32) * 2.0 +
             pl.col('bookmark').cast(pl.Float32) * 1.5 +
             pl.col('share').cast(pl.Float32) * 1.5).alias('weight')
        ])

        # Sample with probability proportional to weight
        if len(interactions) > max_samples:
            interactions = interactions.sample(n=max_samples, seed=42)

        self.pairs = []
        for row in interactions.iter_rows(named=True):
            user_idx = feature_store.user_id_to_idx.get(row['user_id'], 0)
            item_idx = feature_store.item_id_to_idx.get(row['item_id'], 0)
            if user_idx > 0 and item_idx > 0:
                self.pairs.append((user_idx, item_idx))

        print(f"Created {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    """Simple collate - just stack indices."""
    user_idx = torch.LongTensor([b[0] for b in batch])
    item_idx = torch.LongTensor([b[1] for b in batch])
    return user_idx, item_idx


# ==============================================================================
# SECTION 5: TRAINING WITH IN-BATCH NEGATIVES
# ==============================================================================
def train_model(model: TwoTowerModel, dataset: TwoTowerDataset,
                feature_store: FeatureStore, num_epochs: int):
    """Train with in-batch negatives (much faster than explicit sampling)."""
    print(f"\n--- Training Two-Tower Model ({num_epochs} epochs) ---")

    # Move feature tensors to device
    feature_store.user_age = feature_store.user_age.to(DEVICE)
    feature_store.user_gender = feature_store.user_gender.to(DEVICE)
    feature_store.user_geo = feature_store.user_geo.to(DEVICE)
    feature_store.user_features = feature_store.user_features.to(DEVICE)
    feature_store.item_author = feature_store.item_author.to(DEVICE)
    feature_store.item_duration = feature_store.item_duration.to(DEVICE)
    feature_store.item_features = feature_store.item_features.to(DEVICE)
    feature_store.item_pretrained = feature_store.item_pretrained.to(DEVICE)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'], weight_decay=0.01)

    dataloader = DataLoader(
        dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * num_epochs
    warmup_steps = len(dataloader)  # 1 epoch warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    temperature = MODEL_CONFIG['temperature']
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for user_idx, item_idx in pbar:
            user_idx = user_idx.to(DEVICE)
            item_idx = item_idx.to(DEVICE)

            # Get embeddings
            user_emb = model.get_user_emb(user_idx, feature_store)
            item_emb = model.get_item_emb(item_idx, feature_store)

            # In-batch negatives: all items in batch are negatives for each user
            # Scores: (batch_size, batch_size)
            scores = torch.mm(user_emb, item_emb.T) / temperature

            # Labels: diagonal is positive (user i matches item i)
            labels = torch.arange(scores.size(0), device=DEVICE)

            # Cross-entropy loss
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

        avg_loss = total_loss / num_batches
        best_loss = min(best_loss, avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} (best: {best_loss:.4f})")

    return model


# ==============================================================================
# SECTION 6: INFERENCE
# ==============================================================================
def compute_all_embeddings(model: TwoTowerModel, feature_store: FeatureStore,
                           target_item_ids: List[int],
                           item_embeddings_map: Dict[int, np.ndarray],
                           items_metadata: pl.DataFrame):
    """
    Compute all user and target item embeddings.
    For items: use pretrained embeddings directly (works for cold-start items).
    """
    print("\n--- Computing embeddings ---")
    model.eval()

    # User embeddings
    user_indices = list(range(1, feature_store.num_users + 1))
    user_embeddings = []

    with torch.no_grad():
        batch_size = 8192
        for i in tqdm(range(0, len(user_indices), batch_size), desc="User embeddings"):
            batch_idx = torch.LongTensor(user_indices[i:i+batch_size]).to(DEVICE)
            emb = model.get_user_emb(batch_idx, feature_store)
            user_embeddings.append(emb.cpu().numpy())

    user_embeddings = np.vstack(user_embeddings).astype(np.float32)
    user_ids = np.array([feature_store.idx_to_user_id[idx] for idx in user_indices], dtype=np.uint32)

    # Item embeddings for submission targets - use pretrained embeddings directly!
    print("Preparing item features for cold-start items...")

    # Get duration for all items from metadata
    item_duration_map = {}
    for row in items_metadata.iter_rows(named=True):
        item_duration_map[row['item_id']] = (row.get('duration', 0) or 0) / 255.0

    # Prepare pretrained embeddings and durations for target items
    emb_dim = MODEL_CONFIG['item_emb_dim']
    target_pretrained = []
    target_durations = []

    for item_id in target_item_ids:
        if item_id in item_embeddings_map:
            target_pretrained.append(item_embeddings_map[item_id])
        else:
            target_pretrained.append(np.zeros(emb_dim, dtype=np.float32))
        target_durations.append(item_duration_map.get(item_id, 0.0))

    target_pretrained = np.array(target_pretrained, dtype=np.float32)
    target_durations = np.array(target_durations, dtype=np.float32)

    # Compute item embeddings through the model
    item_embeddings = []
    with torch.no_grad():
        batch_size = 4096
        for i in tqdm(range(0, len(target_item_ids), batch_size), desc="Item embeddings"):
            batch_pretrained = torch.FloatTensor(target_pretrained[i:i+batch_size]).to(DEVICE)
            batch_duration = torch.FloatTensor(target_durations[i:i+batch_size]).to(DEVICE)

            # Item tower only needs pretrained embeddings and duration
            emb = model.item_tower(batch_pretrained, batch_duration)
            item_embeddings.append(emb.cpu().numpy())

    item_embeddings = np.vstack(item_embeddings).astype(np.float32)

    return user_ids, user_embeddings, item_embeddings


def faiss_search(user_embeddings: np.ndarray, item_embeddings: np.ndarray, top_k: int):
    """FAISS search for top-k users per item."""
    print(f"\n--- FAISS search (top-{top_k}) ---")

    faiss.normalize_L2(user_embeddings)
    faiss.normalize_L2(item_embeddings)

    index = faiss.IndexFlatIP(user_embeddings.shape[1])
    index.add(user_embeddings)

    _, indices = index.search(item_embeddings, top_k)
    return indices


# ==============================================================================
# SECTION 7: ANTI-SPAM AND SUBMISSION
# ==============================================================================
def apply_antispam(candidate_indices: np.ndarray, user_ids: np.ndarray,
                   target_item_ids: List[int], fallback_users: np.ndarray):
    """Apply anti-spam constraints."""
    print("\n--- Applying anti-spam constraints ---")

    user_counts = defaultdict(int)
    num_items = len(target_item_ids)
    final_predictions = [[] for _ in range(num_items)]
    final_sets = [set() for _ in range(num_items)]

    # Candidate lists
    candidate_lists = []
    for i in range(num_items):
        candidates = [int(user_ids[idx]) for idx in candidate_indices[i] if idx < len(user_ids)]
        candidate_lists.append(candidates)

    max_candidates = max(len(c) for c in candidate_lists)

    # Fair distribution
    for rank in tqdm(range(max_candidates), desc="Distributing"):
        for item_idx in range(num_items):
            if len(final_predictions[item_idx]) >= FINAL_TOP_K:
                continue
            if rank < len(candidate_lists[item_idx]):
                user = candidate_lists[item_idx][rank]
                if user_counts[user] < MAX_USER_APPEARANCES and user not in final_sets[item_idx]:
                    final_predictions[item_idx].append(user)
                    final_sets[item_idx].add(user)
                    user_counts[user] += 1

    # Fallback
    fallback_ptr = 0
    for i in tqdm(range(num_items), desc="Fallback"):
        while len(final_predictions[i]) < FINAL_TOP_K:
            if fallback_ptr >= len(fallback_users):
                fallback_ptr = 0
            user = int(fallback_users[fallback_ptr])
            fallback_ptr += 1
            if user_counts[user] < MAX_USER_APPEARANCES and user not in final_sets[i]:
                final_predictions[i].append(user)
                final_sets[i].add(user)
                user_counts[user] += 1

    return final_predictions


def create_submission(submission_df: pl.DataFrame, predictions: List[List[int]],
                      filename: str = 'two_tower_submission.parquet'):
    """Create submission file."""
    print("\n--- Creating submission ---")

    predictions_np = np.array(predictions, dtype=np.uint32)
    result = submission_df.with_columns(
        pl.Series(name='user_id', values=predictions_np).cast(pl.Array(pl.UInt32, 100))
    )
    result.write_parquet(filename)
    print(f"Saved: {filename}")
    print(f"Shape: {result.shape}, Schema: {result.schema}")
    return result


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("Two-Tower Model Pipeline (Optimized)")
    print("=" * 60)

    # Load data
    train_int, val_int, users_meta, item_emb_map, items_meta, submission_df = load_all_data()
    all_interactions = pl.concat([train_int, val_int])

    # Build feature store
    feature_store = FeatureStore(all_interactions, users_meta, items_meta, item_emb_map)

    # Create dataset
    dataset = TwoTowerDataset(all_interactions, feature_store, MODEL_CONFIG['max_train_samples'])

    # Train model
    model = TwoTowerModel(feature_store)
    model = train_model(model, dataset, feature_store, MODEL_CONFIG['num_epochs'])

    # Compute embeddings (pass item_emb_map and items_meta for cold-start items)
    target_item_ids = submission_df['item_id'].to_list()
    user_ids, user_embs, item_embs = compute_all_embeddings(
        model, feature_store, target_item_ids, item_emb_map, items_meta
    )

    # FAISS search
    candidate_indices = faiss_search(user_embs, item_embs, CANDIDATE_POOL_SIZE)

    # Anti-spam
    popular_users = all_interactions.group_by('user_id').count().sort('count', descending=True)['user_id'].to_numpy()
    predictions = apply_antispam(candidate_indices, user_ids, target_item_ids, popular_users)

    # Create submission
    create_submission(submission_df, predictions)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
