import argparse
import json
import os

import jinja2
from jinja2 import Environment, FileSystemLoader


def clean_queue_paths(username: str) -> None:
    """Cleans the queue paths."""
    queue_paths = [
        os.path.join(os.getcwd(), "assets/configs/queue"),
        os.path.join(os.getcwd(), "job_queue"),
    ]
    for queue_path in queue_paths:
        if os.path.exists(queue_path):
            for file in os.listdir(queue_path):
                if file.startswith(username):
                    os.remove(os.path.join(queue_path, file))


def look_for_jobs_in_queue(username: str) -> None:
    """Looks for jobs in the queue. If there are no jobs it cleans the queue paths."""

    queue = os.popen("squeue").read().strip()
    jobs_in_queue = False
    for line in queue.split("\n"):
        if username in line:
            jobs_in_queue = True
            break

    if not jobs_in_queue:
        clean_queue_paths(username)


def read_and_modify_json(args: dict) -> dict:
    """Reads the base config and modifies it according to the arguments passed."""
    with open(os.path.join(os.getcwd(), "assets/configs/base.json"), "r") as f:
        config = json.load(f)
    # config["model"]["kwargs"]["block_activations"] = args["block_activations"]
    # config["model"]["kwargs"]["reprojector_activation"] = args["reprojector_activation"]
    return config


def get_last_number_in_queue(queue_path: str, username: str) -> int:
    """Checks if the queue path exists and returns the last number in the queue."""
    if not os.path.exists(queue_path):
        # If path does not exist, create it
        os.makedirs(queue_path)
        current_queue_number = 1
    else:
        # If path exists, get the last config in the queue and increment the number
        user_config_queue = []
        for file in os.listdir(queue_path):
            if file.startswith(username):
                user_config_queue.append(file)

        if user_config_queue:
            # If user has configs in the queue, get the last config and increment the number
            last_config_in_queue = sorted(user_config_queue)[-1]
            current_queue_number = (
                int(last_config_in_queue.split("_")[-1].split(".")[0]) + 1
            )
        else:
            # If user does not have configs in the queue, start from 1
            current_queue_number = 1
    return current_queue_number


def write_json(config: dict, username: str) -> str:
    """Writes the config to the queue path and returns the path to the config."""
    queue_path = os.path.join(os.getcwd(), "assets/configs/queue")
    current_queue_number = get_last_number_in_queue(queue_path, username)

    config_name = f"{username}_config_{current_queue_number}.json"
    config_path = os.path.join(queue_path, config_name)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config_path


def read_jinja_template() -> jinja2.Template:
    """Reads the jinja template and returns the template."""
    template_path = os.path.join(os.getcwd(), "templates/")
    environment = Environment(loader=FileSystemLoader(template_path))
    template = environment.get_template("train_template.sh")
    return template


def write_template(template: jinja2.Template, config_path: str, username: str) -> str:
    """Writes the template with config_path and returns the path of the shell script."""
    rendered_template = template.render(
        config_path=config_path, email=f"{username}@eafit.edu.co"
    )

    job_queue_path = os.path.join(os.getcwd(), "job_queue")
    current_queue_number = get_last_number_in_queue(job_queue_path, username)

    train_script_name = f"{username}_train_{current_queue_number}.sh"
    train_script_path = os.path.join(job_queue_path, train_script_name)

    with open(train_script_path, "w") as f:
        f.write(rendered_template)

    return train_script_path


def run_script(script_path: str) -> None:
    """Runs the script."""
    os.system(f"chmod 774 {script_path}")
    os.system(f"sbatch {script_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block_activations",
        type=str,
        nargs="+",
        default=[],
        help="Activations to use in the model.",
    )
    parser.add_argument(
        "--reprojector_activation",
        type=str,
        default="",
        help="Activation to use in the reprojector.",
    )
    args = parser.parse_args()

    username = os.popen("whoami").read().strip()
    look_for_jobs_in_queue(username)
    config = read_and_modify_json(vars(args))
    config_path = write_json(config, username)
    template = read_jinja_template()
    train_script_path = write_template(template, config_path, username)
    run_script(train_script_path)
