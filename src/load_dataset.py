from datasets import load_dataset


def explore_dataset(train_path, test_path, format="parquet"):
    dataset = get_dataset(train_path, test_path, format)
    
    print("📊 Информация о датасете:")
    print(f"Размер тренировочной выборки: {len(dataset['train']):,} примеров")
    print(f"Размер тестовой выборки: {len(dataset['test']):,} примеров")
    
    train_labels = dataset['train']['label']
    test_labels = dataset['test']['label']
    
    print(f"\n🏷️ Распределение меток в тренировочной выборке:")
    print(f"Инъекции (1): {sum(train_labels):,}")
    print(f"Нормальные (0): {len(train_labels) - sum(train_labels):,}")
    
    print(f"\n🏷️ Распределение меток в тестовой выборке:")
    print(f"Инъекции (1): {sum(test_labels):,}")
    print(f"Нормальные (0): {len(test_labels) - sum(test_labels):,}")
    
    print("\n🔍 Примеры инъекций:")
    injection_examples = [ex for ex in dataset['train'] if ex['label'] == 1][:3]
    for i, ex in enumerate(injection_examples):
        print(f"{i+1}. {ex['text'][:200]}...")
    
    print("\n🔍 Примеры нормальных запросов:")
    normal_examples = [ex for ex in dataset['train'] if ex['label'] == 0][:3]
    for i, ex in enumerate(normal_examples):
        print(f"{i+1}. {ex['text'][:200]}...")
    
    return dataset


def get_dataset(train_path, test_path, format="parquet"):
    dataset = load_dataset(format, data_files={
        'train': train_path,
        'test': test_path
    })
    
    return dataset