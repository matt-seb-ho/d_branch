import json
import logging
from datetime import datetime

import yaml
from model_utils import LM
from module import Node, calculate_mc_score, perform_rollouts, process_annotations
from tqdm import tqdm

from gen_data import load_config, load_json_file, setup_logging


def main():
    # Load configuration
    config = load_config("config.yaml")

    # Get parameters from config
    json_file_path = config["input"]["json_file_path"]
    log_file_path = config["output"]["log_file_path"]
    file_prefix = config["output"]["file_prefix"]
    num_rollouts = config["processing"]["num_rollouts"]
    initial_rollouts = config["processing"]["initial_rollouts"]
    max_iterations = config["processing"]["max_iterations"]
    example_limit = config["input"]["example_limit"]

    lm_model = LM(
        model_type=config["model"]["model_type"],
        model_name=config["model"]["model_name"],
        num_rollouts=num_rollouts,
        **config["model"]["model_args"],
    )

    # Set up logging
    setup_logging(log_file_path)

    # Start the process and log it
    logging.info("Started processing the JSON file.")

    # Load the JSON data
    full_data = load_json_file(json_file_path)
    data = full_data[:example_limit]

    # Process each problem and its final answer
    for i, item in tqdm(enumerate(data), total=len(data)):
        problem = item.get("problem", "No problem found")
        final_answer = item.get("final_answer", "No answer found")

        # Log each problem and answer
        logging.info(f"Processed Problem {i + 1}: {problem}")
        logging.info(f"Final Answer: {final_answer}")

        # Initialize the root node and perform rollouts
        nodes = []
        root_node = Node(problem, "", final_answer)
        rollouts, correctness_flags = perform_rollouts(
            root_node, lm_model, initial_rollouts, single_pass_mode=False
        )
        mc_score = calculate_mc_score(root_node)
        root_node.mc_score = mc_score

        nodes.append(root_node)

        # Check if further processing is needed
        if 0 < sum(correctness_flags) < initial_rollouts:
            print("Processing annotations ...\n")
            filename = f"{file_prefix}_{i+1}_nodes_data.json"
            process_annotations(
                problem,
                nodes,
                lm_model,
                filename,
                max_iterations,
                single_pass_mode=True,
            )

    # Log completion
    logging.info("Finished processing the JSON file.")


if __name__ == "__main__":
    main()
