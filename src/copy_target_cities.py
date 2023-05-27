import pathlib
import pickle 
import shutil
from typing import Tuple

def get_target_cities() -> Tuple[list[pathlib.Path], list[pathlib.Path]]:
    planned_cities_ids, not_planned_cities_ids = get_sampled_cities_ids()

    GRAPH_DATA_FOLDER = pathlib.Path("../data/graphml/")
    print(f"Data folder: {GRAPH_DATA_FOLDER}")
    
    all_graph_data_files = GRAPH_DATA_FOLDER.glob("*/*.graphml")
    planned_cities_paths = list()
    not_planned_cities_paths = list()

    num_planned_cities = len(planned_cities_ids)
    num_not_planned_cities = len(not_planned_cities_ids)

    for file_path in all_graph_data_files:
        city_id = get_city_id_from_file_name(file_path)

        if city_id in planned_cities_ids:
            planned_cities_paths.append(file_path)
        elif city_id in not_planned_cities_ids:
            not_planned_cities_paths.append(file_path)
        
        if len(planned_cities_paths) == num_planned_cities and len(not_planned_cities_paths) == num_not_planned_cities:
            break

    return planned_cities_paths, not_planned_cities_paths

def get_sampled_cities_ids():
    planned_cities_ids_file_path = pathlib.Path(
        "../data/POC2_data/random_planned_cities_id.pkl"
        )
    not_planned_cities_ids_file_path = pathlib.Path(
        "../data/POC2_data/random_not_planned_cities_id.pkl"
        )

    planned_cities_ids = get_pickle_file_content(planned_cities_ids_file_path)
    not_planned_cities_ids = get_pickle_file_content(not_planned_cities_ids_file_path)
    return planned_cities_ids,not_planned_cities_ids

def get_pickle_file_content(file_path:pathlib.Path):
    with open(file_path, 'rb') as my_file:
        data = pickle.load(my_file)
    
    return data

def get_city_id_from_file_name(file_path):
    city_name_and_id = file_path.stem
    city_id = int(city_name_and_id.split('-')[1])
    return city_id

def get_total_size_of_files(files):
    total = 0
    for city_file in files:
        total += pathlib.Path(city_file).stat().st_size
    
    return total

def copy_to_target_folder(files, target_folder):
    for city_file in files:
        target_file = target_folder / city_file.name
        shutil.copy(city_file, target_file)

if __name__ == "__main__":
    planned_cities_files, not_planned_cities_files = get_target_cities()

    planned_total_size = get_total_size_of_files(planned_cities_files)
    not_planned_total_size = get_total_size_of_files(not_planned_cities_files)
    
    print(f"Planned total (MB): {(planned_total_size / 1024) / 1024}")
    print(f"Not Planned total (MB): {(not_planned_total_size / 1024) / 1024}")
    print(f"Total (MB): {((planned_total_size + not_planned_total_size) / 1024) / 1024}")

    target_folder = pathlib.Path("../data/POC2_data/sampled_cities/")
    target_folder.mkdir(parents=True, exist_ok=True)

    copy_to_target_folder(planned_cities_files, target_folder)
    copy_to_target_folder(not_planned_cities_files, target_folder)