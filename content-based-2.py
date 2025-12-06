import os
import polars as pl
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import faiss
from collections import defaultdict

# ==============================================================================
# РАЗДЕЛ 1: ЗАГРУЗКА ДАННЫХ (без изменений)
# ==============================================================================
def load_all_data():
    print("--- Шаг 1: Загрузка данных ---")
    data_dir = 'VK-LSVD'
    interaction_files = [f'C:\\Users\\user\\Desktop\\vk_recsys\\VK-LSVD\\train\\week_{i:02}.parquet' for i in range(23, 25)]
    embedding_file = 'metadata/item_embeddings.npz'
    #for file in tqdm(interaction_files c, desc="Скачивание файлов"):
    #    hf_hub_download('deepvk/VK-LSVD', file, local_dir=data_dir, repo_type='dataset', force_filename=file)
    interactions_df = pl.concat([pl.read_parquet(os.path.join(data_dir, f)) for f in interaction_files])
    embeddings_data = np.load(os.path.join(data_dir, embedding_file))
    item_ids = embeddings_data['item_id']
    item_embeddings_matrix = embeddings_data['embedding']
    item_embeddings_map = {id: emb for id, emb in zip(item_ids, item_embeddings_matrix)}
    submission_df = pl.read_parquet(r'C:\Users\user\Desktop\vk_recsys\VK-LSVD\metadata\submission.parquet', columns=['item_id'])
    print("Данные успешно загружены.")
    return interactions_df, item_embeddings_map, submission_df

# ==============================================================================
# РАЗДЕЛ 2: СОЗДАНИЕ ПРОФИЛЕЙ ПОЛЬЗОВАТЕЛЕЙ (С ВЗВЕШЕННЫМ УСРЕДНЕНИЕМ)
# ==============================================================================

def build_user_profiles(interactions, embeddings_map):
    """
    Для каждого пользователя находит топ-100 айтемов по времени просмотра
    и создает его "профиль" как ВЗВЕШЕННОЕ среднее их эмбеддингов,
    где вес - это timespent.
    """
    print("\n--- Шаг 2: Создание профилей пользователей (метод взвешенного среднего) ---")
    
    # --- ИЗМЕНЕНИЕ: Теперь мы берем не просто item_id, а пару (item_id, timespent) ---
    # 1. Сортируем по timespent и берем топ-100
    top_interactions = interactions.sort("timespent", descending=True).group_by("user_id").head(100)
    
    # 2. Группируем, чтобы собрать все айтемы и их timespent для каждого юзера
    user_top_content = top_interactions.group_by('user_id').agg(
        # Собираем айтемы и время просмотра в структуры, чтобы сохранить их связь
        pl.struct(['item_id', 'timespent']).alias('top_content')
    )
    
    user_profiles = []
    profiled_user_ids = []
    
    # 3. Для каждого юзера вычисляем взвешенное среднее
    for row in tqdm(user_top_content.iter_rows(named=True), desc="Вычисление взвешенных эмбеддингов", total=len(user_top_content)):
        user_id = row['user_id']
        
        embeddings_to_average = []
        weights = []
        
        # row['top_content'] - это список словарей, например [{'item_id': 1, 'timespent': 200}, ...]
        for interaction in row['top_content']:
            item_id = interaction['item_id']
            timespent = interaction['timespent']
            
            # Добавляем эмбеддинг и его вес, только если эмбеддинг существует
            if item_id in embeddings_map:
                embeddings_to_average.append(embeddings_map[item_id])
                # Добавляем 1, чтобы избежать деления на ноль, если timespent=0
                weights.append(timespent + 1)
        
        # Если для пользователя нашлись эмбеддинги, вычисляем взвешенное среднее
        if embeddings_to_average:
            # np.average - идеальная функция для этого
            profile_emb = np.average(
                np.array(embeddings_to_average, dtype=np.float32), 
                axis=0, 
                weights=np.array(weights, dtype=np.float32)
            )
            user_profiles.append(profile_emb)
            profiled_user_ids.append(user_id)
            
    print(f"Создано {len(profiled_user_ids)} профилей пользователей.")
    return np.array(profiled_user_ids, dtype=np.uint32), np.array(user_profiles, dtype=np.float32)

def build_user_profiles(interactions, embeddings_map):
    print("\n--- Шаг 2: Создание профилей пользователей (метод взвешенного среднего) ---")
    top_interactions = interactions.sort("timespent", descending=True).group_by("user_id").head(100)
    user_top_content = top_interactions.group_by('user_id').agg(
        pl.struct(['item_id', 'timespent']).alias('top_content')
    )
    user_profiles, profiled_user_ids = [], []
    for row in tqdm(user_top_content.iter_rows(named=True), desc="Вычисление взвешенных эмбеддингов", total=len(user_top_content)):
        embeddings_to_average, weights = [], []
        for interaction in row['top_content']:
            item_id, timespent = interaction['item_id'], interaction['timespent']
            if item_id in embeddings_map:
                embeddings_to_average.append(embeddings_map[item_id])
                weights.append(timespent + 1)
        if embeddings_to_average:
            profile_emb = np.average(
                np.array(embeddings_to_average, dtype=np.float32), 
                axis=0, 
                weights=np.array(weights, dtype=np.float32)
            )
            user_profiles.append(profile_emb)
            profiled_user_ids.append(row['user_id'])
    print(f"Создано {len(profiled_user_ids)} профилей пользователей.")
    return np.array(profiled_user_ids, dtype=np.uint32), np.array(user_profiles, dtype=np.float32)

# ==============================================================================
# РАЗДЕЛ 3: ПОИСК И ФОРМИРОВАНИЕ САБМИТА (С ИТЕРАТИВНЫМ РАСПРЕДЕЛЕНИЕМ)
# ==============================================================================
def find_neighbors_and_create_submission(submission, interactions, embeddings_map, user_ids, user_profiles):
    """
    Генерирует предсказания и применяет справедливое итеративное распределение
    для соблюдения анти-спам правила.
    """
    print("\n--- Шаг 3: Генерация кандидатских списков ---")
    embedding_dim = user_profiles.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    faiss.normalize_L2(user_profiles)
    index.add(user_profiles)
    target_item_ids = submission['item_id'].to_list()
    target_embeddings = []
    for item_id in tqdm(target_item_ids, desc="Подготовка эмбеддингов айтемов"):
        emb = embeddings_map.get(item_id, np.zeros(embedding_dim, dtype=np.float32))
        target_embeddings.append(emb)
    target_embeddings = np.array(target_embeddings, dtype=np.float32)
    faiss.normalize_L2(target_embeddings)
    print("Выполняется поиск ближайших соседей...")
    _, neighbor_indices = index.search(target_embeddings, 100)
    candidate_matrix = user_ids[neighbor_indices]

    print("\n--- Шаг 4: Справедливое итеративное распределение кандидатов ---")
    user_counts = defaultdict(int)
    num_items = len(target_item_ids)
    final_predictions = [[] for _ in range(num_items)]
    final_predictions_sets = [set() for _ in range(num_items)]

    for rank in tqdm(range(100), desc="Распределение по рангам"):
        for item_idx in range(num_items):
            if len(final_predictions[item_idx]) < 100:
                candidate_user = candidate_matrix[item_idx, rank]
                if user_counts[candidate_user] < 100 and candidate_user not in final_predictions_sets[item_idx]:
                    final_predictions[item_idx].append(candidate_user)
                    final_predictions_sets[item_idx].add(candidate_user)
                    user_counts[candidate_user] += 1

    print("\n--- Шаг 5: Финальный фолбэк для оставшихся пустых слотов ---")
    # Расчет фолбэка делаем прямо здесь, чтобы он был свежим
    popular_users_fallback = interactions.group_by('user_id').count().sort('count', descending=True).get_column('user_id').to_numpy()
    fallback_pointer = 0
    
    for i in tqdm(range(num_items), desc="Финальный фолбэк"):
        while len(final_predictions[i]) < 100:
            if fallback_pointer >= len(popular_users_fallback):
                fallback_pointer = 0
            candidate_user = popular_users_fallback[fallback_pointer]
            fallback_pointer += 1
            if user_counts[candidate_user] < 100 and candidate_user not in final_predictions_sets[i]:
                final_predictions[i].append(candidate_user)
                final_predictions_sets[i].add(candidate_user)
                user_counts[candidate_user] += 1

    print("\n--- Шаг 6: Сборка и сохранение итогового файла ---")
    final_predictions_np = np.array(final_predictions, dtype=np.uint32)
    submission_df = submission.with_columns(user_id=final_predictions_np)
    output_filename = 'content_based_submission.parquet'
    submission_df.write_parquet(output_filename)
    
    print(f"\nФайл '{output_filename}' успешно создан!")
    print("Информация об итоговом DataFrame:")
    print(f"Форма (shape): {submission_df.shape}")
    print(f"Схема (schema) по версии Polars: {submission_df.schema}")

# ==============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ==============================================================================
if __name__ == '__main__':
    interactions_df, item_embeddings_map, submission_df = load_all_data()
    profiled_user_ids, user_profiles_matrix = build_user_profiles(interactions_df, item_embeddings_map)
    find_neighbors_and_create_submission(submission_df, interactions_df, item_embeddings_map, profiled_user_ids, user_profiles_matrix)