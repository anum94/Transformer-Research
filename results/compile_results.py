import os
import json
import pandas as pd


def calculate_average_metrics(metrics_dict):
    rouge1 = metrics_dict["rouge1"]
    rouge2 = metrics_dict["rouge2"]
    rougeL = metrics_dict["rougeL"]

    average_rouge = (rouge1+rouge2+rougeL) / 3


    return average_rouge, rouge1, rouge2, rougeL


def process_folders(root_folder):
    results = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        metrics_file = os.path.join(folder_path, "metrics.json")

        if not os.path.isfile(metrics_file):
            continue

        with open(metrics_file) as f:
            metrics_data = json.load(f)

        print (folder_name)
        if 'rouge' in metrics_data:
            average_metrics = calculate_average_metrics(metrics_data['rouge'])
        else:
            average_metrics = calculate_average_metrics(metrics_data)

        result = {
            "Folder": folder_name,
            "RougeAvg": average_metrics[0],
            "Rouge1": average_metrics[1],
            "Rouge2": average_metrics[2],
            "RougeL": average_metrics[3]
        }
        results.append(result)

    return results


def write_to_excel(results, output_file):
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)


root_folder = "."
output_file = "result.xlsx"

results = process_folders(root_folder)
write_to_excel(results, output_file)
